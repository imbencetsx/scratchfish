"""Time-bounded alpha-beta search; all scores are side-to-move centipawns."""

import time
from collections import Counter
from collections.abc import Callable

import chess

from evaluation import PIECE_VALUES

MATE = 100_000
MATE_THRESHOLD = MATE - 1000
MAX_QDEPTH = 6
MAX_PLY = 96
EvalFn = Callable[[chess.Board], int]


class _Timeout(Exception):
    pass


def _tt_key(board: chess.Board):
    return board._transposition_key()


def _pack_mate(score: int, ply: int) -> int:
    return score + ply if score > MATE_THRESHOLD else score - ply if score < -MATE_THRESHOLD else score


def _unpack_mate(score: int, ply: int) -> int:
    return score - ply if score > MATE_THRESHOLD else score + ply if score < -MATE_THRESHOLD else score


class Searcher:
    def __init__(self, eval_white_cp: EvalFn, time_limit: float = 0.35,
                 node_limit: int | None = None, stop_event=None, root_eval_white_cp: EvalFn | None = None):
        self.eval_white_cp = eval_white_cp
        self.root_eval_white_cp = root_eval_white_cp
        self.time_limit = max(0.0, time_limit)
        self.node_limit = node_limit
        self.stop_event = stop_event
        self.nodes = self.tt_hits = self.eval_hits = 0
        self._deadline = 0.0
        self._tt = {}
        self._eval_cache = {}
        self._killers = [[None, None] for _ in range(MAX_PLY)]
        self._history = {}
        self._positions = Counter()
        self._repeated = 0

    def _check_time(self):
        if ((self.node_limit is not None and self.nodes >= self.node_limit)
                or time.monotonic() >= self._deadline
                or (self.stop_event is not None and self.stop_event.is_set())):
            raise _Timeout

    def _visit(self):
        self.nodes += 1
        if self.nodes % 16 == 0 or (self.node_limit is not None and self.nodes >= self.node_limit):
            self._check_time()

    def _push(self, board, move):
        board.push(move)
        key = _tt_key(board)
        self._positions[key] += 1
        if self._positions[key] == 2:
            self._repeated += 1

    def _pop(self, board):
        key = _tt_key(board)
        if self._positions[key] == 2:
            self._repeated -= 1
        self._positions[key] -= 1
        board.pop()

    def _draw(self, board):
        return (board.is_insufficient_material() or board.halfmove_clock >= 100
                or self._positions[_tt_key(board)] >= 3)

    def best_move(self, board: chess.Board, max_depth: int = 8) -> tuple[chess.Move | None, dict]:
        started = time.monotonic()
        self.nodes = self.tt_hits = self.eval_hits = 0
        self._tt.clear()
        self._eval_cache.clear()
        self._history.clear()
        self._killers = [[None, None] for _ in range(MAX_PLY)]
        self._deadline = started + self.time_limit
        # Include actual game history, not just the position at the search root.
        history = board.copy()
        self._positions = Counter([_tt_key(history)])
        while history.move_stack:
            previous = history.pop()
            if history.is_irreversible(previous):
                break
            self._positions[_tt_key(history)] += 1
        self._repeated = sum(n >= 2 for n in self._positions.values())
        legal = self._ordered(board, board.legal_moves)
        if self.root_eval_white_cp is not None and legal:
            root_scores = {}
            for move in legal:
                if time.monotonic() >= self._deadline:
                    break
                board.push(move)
                try:
                    value = self.root_eval_white_cp(board)
                    root_scores[move] = value if not board.turn else -value
                finally:
                    board.pop()
            legal.sort(key=lambda move: root_scores.get(move, -MATE), reverse=True)
        best = legal[0] if legal else None
        score = -MATE if not legal and board.is_check() else 0
        completed_depth = 0
        try:
            if best is not None:
                for depth in range(1, max_depth + 1):
                    self._check_time()
                    alpha, candidate = -MATE, best
                    for index, move in enumerate(self._ordered(board, legal, first=best)):
                        self._check_time()
                        self._push(board, move)
                        try:
                            if index == 0 or depth == 1:
                                value = -self._negamax(board, depth - 1, -MATE, -alpha, 1)
                            else:
                                value = -self._negamax(board, depth - 1, -alpha - 1, -alpha, 1)
                                if value > alpha:
                                    value = -self._negamax(board, depth - 1, -MATE, -alpha, 1)
                        finally:
                            self._pop(board)
                        if value > alpha:
                            alpha, candidate = value, move
                            # Even an interrupted first iteration has a searched legal fallback.
                            if completed_depth == 0:
                                best, score = candidate, alpha
                    best, score, completed_depth = candidate, alpha, depth
                    if abs(score) > MATE_THRESHOLD or len(legal) == 1:
                        break
        except _Timeout:
            pass
        elapsed = time.monotonic() - started
        return best, {"depth": completed_depth, "nodes": self.nodes,
                      "score_cp": score, "tt_hits": self.tt_hits,
                      "eval_hits": self.eval_hits, "elapsed": elapsed,
                      "nps": int(self.nodes / max(elapsed, 1e-9))}

    def _eval_stm(self, board):
        key = (_tt_key(board), board.halfmove_clock)
        cached = self._eval_cache.get(key)
        if cached is not None:
            self.eval_hits += 1
            return cached
        value = self.eval_white_cp(board)
        value = value if board.turn else -value
        self._eval_cache[key] = value
        return value

    def _negamax(self, board, depth, alpha, beta, ply):
        if depth <= 0:
            return self._quiesce(board, alpha, beta, ply, MAX_QDEPTH)
        self._visit()
        in_check = board.is_check()
        if self._draw(board):
            return -MATE + ply if in_check and board.is_checkmate() else 0
        if ply >= MAX_PLY - 1:
            return self._eval_stm(board)
        key = (_tt_key(board), board.halfmove_clock)
        # Repetition-dependent scores cannot safely be reused on another history.
        tt = self._tt.get(key) if self._repeated == 0 else None
        tt_move = None
        if tt is not None:
            tt_depth, value, flag, tt_move = tt
            value = _unpack_mate(value, ply)
            if tt_depth >= depth:
                self.tt_hits += 1
                if flag == 0 or (flag == 1 and value >= beta) or (flag == 2 and value <= alpha):
                    return value
        moves = self._ordered(board, board.legal_moves, first=tt_move, ply=ply)
        if not moves:
            return -MATE + ply if in_check else 0
        if (depth >= 3 and not in_check and abs(beta) < MATE_THRESHOLD
                and (not board.move_stack or board.peek() != chess.Move.null())
                and board.halfmove_clock < 90
                and board.occupied_co[board.turn] & ~(board.pawns | board.kings)
                and self._eval_stm(board) >= beta):
            self._push(board, chess.Move.null())
            try:
                null_value = -self._negamax(board, depth - 3, -beta, -beta + 1, ply + 1)
            finally:
                self._pop(board)
            if null_value >= beta:
                return beta
        original_alpha, best_value, best = alpha, -MATE, moves[0]
        for index, move in enumerate(moves):
            quiet = not board.is_capture(move) and not move.promotion
            color = board.turn
            self._push(board, move)
            try:
                if index == 0:
                    value = -self._negamax(board, depth - 1, -beta, -alpha, ply + 1)
                else:
                    # Reduce late quiet moves, then verify every alpha improvement at full depth.
                    reduction = int(depth >= 3 and index >= 4 and quiet and not in_check and not board.is_check())
                    value = -self._negamax(board, depth - 1 - reduction, -alpha - 1, -alpha, ply + 1)
                    if reduction and value > alpha:
                        value = -self._negamax(board, depth - 1, -alpha - 1, -alpha, ply + 1)
                    if alpha < value < beta:
                        value = -self._negamax(board, depth - 1, -beta, -alpha, ply + 1)
            finally:
                self._pop(board)
            if value > best_value:
                best_value, best = value, move
            alpha = max(alpha, value)
            if alpha >= beta:
                if quiet:
                    killers = self._killers[ply]
                    if move != killers[0]:
                        killers[1], killers[0] = killers[0], move
                    hk = (color, move.from_square, move.to_square)
                    self._history[hk] = min(8000, self._history.get(hk, 0) + depth * depth)
                break
        if self._repeated == 0:
            flag = 1 if best_value >= beta else 2 if best_value <= original_alpha else 0
            self._tt[key] = (depth, _pack_mate(best_value, ply), flag, best)
        return best_value

    @staticmethod
    def _see(board, move):
        """Swap-off value using x-ray attackers; used only to prune quieting captures."""
        target = move.to_square
        victim = board.piece_type_at(target) or chess.PAWN
        attacker = board.piece_type_at(move.from_square)
        gains = [PIECE_VALUES[victim]]
        occupied = board.occupied ^ chess.BB_SQUARES[move.from_square]
        if board.is_en_passant(move):
            occupied ^= chess.BB_SQUARES[target - 8 if board.turn else target + 8]
        occupied |= chess.BB_SQUARES[target]
        value = PIECE_VALUES[move.promotion or attacker]
        if move.promotion:
            gains[0] += value - PIECE_VALUES[chess.PAWN]
        side = not board.turn
        for _ in range(30):
            attackers = board.attackers_mask(side, target, occupied) & occupied
            chosen = 0
            for piece in chess.PIECE_TYPES:
                candidates = attackers & board.pieces_mask(piece, side)
                if candidates:
                    # Don't count a pinned defender as a free recapture.
                    for sq in chess.scan_forward(candidates):
                        if board.pin_mask(side, sq) & chess.BB_SQUARES[target]:
                            chosen = chess.BB_SQUARES[sq]
                            break
                    if chosen:
                        break
            if not chosen:
                break
            if piece == chess.KING and board.attackers_mask(not side, target, occupied ^ chosen) & (occupied ^ chosen):
                break
            gains.append(value - gains[-1])
            occupied ^= chosen
            value = 20_000 if piece == chess.KING else PIECE_VALUES[piece]
            side = not side
        for index in range(len(gains) - 1, 0, -1):
            gains[index - 1] = min(gains[index - 1], -gains[index])
        return gains[0]

    def _quiesce(self, board, alpha, beta, ply, left):
        self._visit()
        in_check = board.is_check()
        if self._draw(board):
            return -MATE + ply if in_check and board.is_checkmate() else 0
        if ply >= MAX_PLY - 1:
            return -MATE + ply if in_check and board.is_checkmate() else self._eval_stm(board)
        if in_check:
            # No stand pat in check: every legal evasion must be considered, even at the q-depth cap.
            moves = list(board.legal_moves)
            if not moves:
                return -MATE + ply
        else:
            if not any(board.generate_legal_moves()):
                return 0
            stand = self._eval_stm(board)
            if stand >= beta:
                return stand
            alpha = max(alpha, stand)
            if left <= 0:
                return alpha
            moves = list(board.generate_legal_captures())
            moves.extend(m for m in board.generate_legal_moves(
                from_mask=board.pawns & board.occupied_co[board.turn],
                to_mask=chess.BB_BACKRANKS) if m.promotion and not board.is_capture(m))
        for move in self._ordered(board, moves, ply=ply):
            if not in_check and not move.promotion:
                victim = board.piece_type_at(move.to_square) or chess.PAWN
                # Checks and promotions may justify sacrificing material; never prune those.
                if (stand + PIECE_VALUES[victim] + 180 < alpha
                        or self._see(board, move) < 0) and not board.gives_check(move):
                    continue
            self._push(board, move)
            try:
                value = -self._quiesce(board, -beta, -alpha, ply + 1, left - 1)
            finally:
                self._pop(board)
            if value >= beta:
                return value
            alpha = max(alpha, value)
        return alpha

    @staticmethod
    def _move_score(board, move):
        score = 0
        if board.is_capture(move):
            victim = board.piece_type_at(move.to_square) or chess.PAWN
            attacker = board.piece_type_at(move.from_square)
            score = 10_000 + 10 * PIECE_VALUES[victim] - PIECE_VALUES[attacker]
        if move.promotion:
            score += 8000 + PIECE_VALUES[move.promotion]
        return score

    def _ordered(self, board, moves, first=None, ply=0):
        def priority(move):
            if move == first:
                return 1_000_000
            score = self._move_score(board, move)
            if score:
                return score
            killers = self._killers[min(ply, MAX_PLY - 1)]
            if move == killers[0]:
                return 9000
            if move == killers[1]:
                return 8500
            return self._history.get((board.turn, move.from_square, move.to_square), 0)
        return sorted(moves, key=priority, reverse=True)


def best_move(board, eval_white_cp, max_depth=8, time_limit=0.35):
    return Searcher(eval_white_cp, time_limit).best_move(board, max_depth)
