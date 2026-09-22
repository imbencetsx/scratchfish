"""Negamax search with alpha-beta, quiescence and iterative deepening.

The evaluator scores from White's perspective in centipawns; this module
converts to side-to-move internally. Mate scores dominate any eval.
"""

import time
from collections.abc import Callable

import chess

from evaluation import PIECE_VALUES

MATE = 100_000
MAX_QDEPTH = 4  # quiescence plies (captures only)

# eval from White's perspective, in centipawns
EvalFn = Callable[[chess.Board], int]


class _Timeout(Exception):
    pass


def _tt_key(board: chess.Board):
    """Stable transposition key across python-chess versions."""
    k = getattr(board, "transposition_key", None)
    if callable(k):
        return k()
    k = getattr(board, "_transposition_key", None)
    if callable(k):
        return k()
    return board.fen()


class Searcher:
    def __init__(self, eval_white_cp: EvalFn, time_limit: float = 0.8):
        self.eval_white_cp = eval_white_cp
        self.time_limit = time_limit
        self.nodes = 0
        self.tt_hits = 0
        self.eval_hits = 0
        self._deadline = 0.0
        self._tt: dict[int, tuple[int, int, int, chess.Move | None]] = {}
        self._eval_cache: dict[int, int] = {}

    # ----------------------------
    # Public
    # ----------------------------

    def best_move(
        self, board: chess.Board, max_depth: int
    ) -> tuple[chess.Move | None, dict]:
        """Iterative deepening. Returns (move, info with depth/nodes/score)."""
        legal = list(board.legal_moves)
        if not legal:
            return None, {"depth": 0, "nodes": 0, "score_cp": 0}
        if len(legal) == 1:
            return legal[0], {"depth": 0, "nodes": 1, "score_cp": 0}

        self.nodes = 0
        self.tt_hits = 0
        self.eval_hits = 0
        self._tt = {}
        self._eval_cache = {}
        self._deadline = time.monotonic() + self.time_limit

        best: chess.Move | None = None
        best_score = 0
        completed_depth = 0
        try:
            for depth in range(1, max_depth + 1):
                scored = self._ordered(board, legal, first=best)
                alpha = -MATE
                candidate = None
                candidate_score = 0
                for move in scored:
                    board.push(move)
                    score = -self._negamax(board, depth - 1, -MATE, -alpha, 1)
                    board.pop()
                    if score > alpha:
                        alpha = score
                        candidate = move
                        candidate_score = score
                best, best_score = candidate, candidate_score
                completed_depth = depth
        except _Timeout:
            pass
        return best, {
            "depth": completed_depth,
            "nodes": self.nodes,
            "score_cp": best_score,
            "tt_hits": self.tt_hits,
            "eval_hits": self.eval_hits,
        }

    # ----------------------------
    # Core
    # ----------------------------

    def _eval_stm(self, board: chess.Board) -> int:
        key = _tt_key(board)
        cached = self._eval_cache.get(key)
        if cached is not None:
            self.eval_hits += 1
            return cached
        ev = self.eval_white_cp(board)
        stm = ev if board.turn == chess.WHITE else -ev
        self._eval_cache[key] = stm
        return stm

    def _negamax(
        self, board: chess.Board, depth: int, alpha: int, beta: int, ply: int
    ) -> int:
        self.nodes += 1
        if self.nodes % 2048 == 0 and time.monotonic() > self._deadline:
            raise _Timeout

        if board.is_checkmate():
            return -MATE + ply  # prefer faster mates
        if (
            board.is_stalemate()
            or board.is_insufficient_material()
            or board.is_seventyfive_moves()
        ):
            return 0

        # Transposition table probe.
        key = _tt_key(board)
        tt = self._tt.get(key)
        tt_move: chess.Move | None = None
        if tt is not None:
            tt_depth, tt_score, tt_flag, tt_move = tt
            if tt_depth >= depth:
                self.tt_hits += 1
                if tt_flag == 0:  # exact
                    return tt_score
                if tt_flag == 1 and tt_score >= beta:  # lower bound
                    return beta
                if tt_flag == 2 and tt_score <= alpha:  # upper bound
                    return alpha

        if depth <= 0:
            return self._quiesce(board, alpha, beta, ply, MAX_QDEPTH)

        orig_alpha = alpha
        best_move: chess.Move | None = None
        for move in self._ordered(board, board.legal_moves, first=tt_move):
            board.push(move)
            score = -self._negamax(board, depth - 1, -beta, -alpha, ply + 1)
            board.pop()
            if score >= beta:
                self._tt[key] = (depth, beta, 1, move)
                return beta
            if score > alpha:
                alpha = score
                best_move = move
        # Store exact / upper bound.
        flag = 0 if best_move is not None else 2
        if best_move is None:
            flag = 2 if alpha == orig_alpha else 0
        self._tt[key] = (depth, alpha, flag, best_move)
        return alpha

    def _quiesce(
        self, board: chess.Board, alpha: int, beta: int, ply: int, left: int
    ) -> int:
        if board.is_checkmate():
            return -MATE + ply
        if board.is_stalemate() or board.is_insufficient_material():
            return 0

        stand_pat = self._eval_stm(board)
        if stand_pat >= beta:
            return beta
        if stand_pat > alpha:
            alpha = stand_pat
        if left <= 0:
            return alpha

        tactics = [m for m in board.legal_moves if board.is_capture(m)]
        # Queen promotions are tactical too, even when quiet.
        tactics += [
            m
            for m in board.legal_moves
            if m.promotion == chess.QUEEN and not board.is_capture(m)
        ]
        for move in self._ordered(board, tactics):
            board.push(move)
            score = -self._quiesce(board, -beta, -alpha, ply + 1, left - 1)
            board.pop()
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
        return alpha

    # ----------------------------
    # Move ordering (MVV-LVA + promotions)
    # ----------------------------

    @staticmethod
    def _move_score(board: chess.Board, move: chess.Move) -> int:
        score = 0
        if board.is_capture(move):
            victim = board.piece_at(move.to_square)
            attacker = board.piece_at(move.from_square)
            victim_v = PIECE_VALUES[victim.piece_type] if victim else 0
            # En passant: captured pawn isn't on the target square.
            if board.is_en_passant(move):
                victim_v = PIECE_VALUES[chess.PAWN]
            attacker_v = (
                PIECE_VALUES[attacker.piece_type] if attacker else 0
            )
            score = 10 * victim_v - attacker_v + 10_000
        if move.promotion:
            score += PIECE_VALUES.get(move.promotion, 0) + 8_000
        return score

    def _ordered(
        self,
        board: chess.Board,
        moves,
        first: chess.Move | None = None,
    ) -> list[chess.Move]:
        scored = [(self._move_score(board, m), m) for m in moves]
        scored.sort(key=lambda t: t[0], reverse=True)
        ordered = [m for _, m in scored]
        if first is not None and first in ordered:
            ordered.remove(first)
            ordered.insert(0, first)
        return ordered


def best_move(
    board: chess.Board,
    eval_white_cp: EvalFn,
    max_depth: int = 3,
    time_limit: float = 0.8,
) -> tuple[chess.Move | None, dict]:
    return Searcher(eval_white_cp, time_limit).best_move(board, max_depth)
