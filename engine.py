import chess

from evaluation import evaluate_white_cp
from neural import WEIGHTS_PATH, evaluate_white_nn, try_load_net
from search import Searcher
from runlog import EventLog, new_run_directory, save_game


class ChessEngine:
    """Owns the game state. The GUI calls into this; it never draws."""

    def __init__(
        self,
        bot_is_white: bool = False,
        depth: int = 8,
        time_limit: float = 0.35,
        use_nn: bool = True,
        weights=WEIGHTS_PATH,
        search_mode: str | None = None,
        log_dir=None,
    ):
        self.board = chess.Board()
        self.bot_is_white = bot_is_white
        self.game_over = False
        self.depth = depth
        self.time_limit = time_limit

        # A compact model guides root ordering; tactical leaves stay inexpensive.
        self.net = try_load_net(weights) if use_nn else None
        self.search_mode = search_mode or getattr(self.net, "metadata", {}).get("search_mode", "root")
        if self.search_mode not in ("root", "compiled", "projected"):
            raise ValueError("search_mode must be root, compiled or projected")
        self.use_nn = self.net is not None
        self.last_search_info = {}
        self._log_dir = new_run_directory(log_dir, 'play') if log_dir else None
        self._game_number = 0
        self._game_log = None
        self._logged_end = False
        self._start_game_log()
        if self._log_dir:
            print(f'Game logs: {self._log_dir.resolve()}')
        if use_nn and self.net is None:
            print(f"No compact {WEIGHTS_PATH.name} loaded; using classical evaluation.")

    # ----------------------------
    # Game control
    # ----------------------------

    def new_game(self) -> None:
        self.close_log('new_game')
        self.board = chess.Board()
        self.game_over = False
        self.last_search_info = {}
        self._start_game_log()

    def _start_game_log(self):
        if self._log_dir:
            self._game_number += 1
            self._logged_end = False
            self._game_log = EventLog(self._log_dir / f'game-{self._game_number:06d}.jsonl')
            self._game_log.event('game_start', bot_color=self.bot_color_name(),
                                 weights_loaded=self.use_nn, search_mode=self.search_mode,
                                 depth=self.depth, seconds=self.time_limit)
            self._save_pgn()

    def _save_pgn(self):
        if self._game_log:
            save_game(self.board, self._log_dir / f'game-{self._game_number:06d}.pgn',
                      Event='Scratchfish play', Round=self._game_number,
                      White='Scratchfish' if self.bot_is_white else 'Player',
                      Black='Player' if self.bot_is_white else 'Scratchfish')

    def _record_move(self, move, actor, info=None):
        if self._game_log:
            self._game_log.event('move', ply=len(self.board.move_stack), fen=self.board.fen(),
                                 move=move.uci(), san=self.board.san(move), actor=actor, search=info or {})

    def _record_position(self):
        if self._game_log:
            self._save_pgn()
            if self.game_over and not self._logged_end:
                outcome = self.board.outcome(claim_draw=True)
                self._game_log.event('game_end', result=self.board.result(claim_draw=True),
                                     reason=outcome.termination.name.lower() if outcome else 'game_over',
                                     final_fen=self.board.fen())
                self._logged_end = True

    def close_log(self, reason='quit'):
        if self._game_log:
            if not self._logged_end:
                self._game_log.event('game_end', result=self.board.result(claim_draw=True),
                                     reason=reason, final_fen=self.board.fen())
            self._save_pgn()
            self._game_log.close()
            self._game_log = None

    def switch_side(self) -> None:
        self.close_log('switch_side')
        self.bot_is_white = not self.bot_is_white
        self.new_game()

    # ----------------------------
    # Queries (used by GUI)
    # ----------------------------

    def bot_turn(self) -> bool:
        return self.board.turn == self.bot_is_white and not self.game_over

    def legal_targets(self, square: chess.Square) -> list[chess.Square]:
        return [
            move.to_square
            for move in self.board.legal_moves
            if move.from_square == square
        ]

    def status_text(self) -> str:
        if self.game_over:
            if self.board.is_checkmate():
                outcome = self.board.outcome(claim_draw=True)
                winner = "White" if outcome and outcome.winner else "Black"
                return f"Checkmate - {winner} wins"
            if self.board.is_stalemate():
                return "Stalemate"
            return "Game over"
        turn = "White" if self.board.turn == chess.WHITE else "Black"
        status = f"{turn}'s turn"
        if self.board.is_check():
            status += " - CHECK!"
        return status

    def bot_color_name(self) -> str:
        return "White" if self.bot_is_white else "Black"

    def brain_name(self) -> str:
        if self.use_nn:
            return f"hybrid ({self.time_limit:g}s)"
        return f"classical ({self.time_limit:g}s)"

    def _eval_white_cp(self, board: chess.Board) -> int:
        if self.net is not None:
            return evaluate_white_nn(self.net, board)
        return evaluate_white_cp(board)

    # ----------------------------
    # Moves
    # ----------------------------

    def try_move(self, from_square: chess.Square, to_square: chess.Square) -> bool:
        """Attempt a player move (auto-queens on promotion). Returns True if played."""
        piece = self.board.piece_at(from_square)
        promotion = None
        if (
            piece
            and piece.piece_type == chess.PAWN
            and chess.square_rank(to_square) in (0, 7)
        ):
            promotion = chess.QUEEN

        move = chess.Move(from_square, to_square, promotion=promotion)
        if move in self.board.legal_moves:
            self._record_move(move, 'player')
            self.board.push(move)
            if self.board.is_game_over(claim_draw=True):
                self.game_over = True
            self._record_position()
            return True
        return False

    def find_bot_move(self, board: chess.Board, stop_event=None):
        """Search a private board; safe to call in the GUI worker thread."""
        evaluator = evaluate_white_cp
        if self.net is not None and self.search_mode != 'root' and hasattr(self.net, 'for_search'):
            evaluator = self.net.for_search(board, self.search_mode)
        return Searcher(evaluator, time_limit=self.time_limit, stop_event=stop_event,
                        root_eval_white_cp=self._eval_white_cp if self.net is not None else None).best_move(board, self.depth)

    def apply_bot_move(self, move, info) -> bool:
        """Apply a completed search result on the main thread."""
        self.last_search_info = info
        if move is None or move not in self.board.legal_moves:
            self.game_over = self.board.is_game_over(claim_draw=True)
            return False
        self._record_move(move, 'bot', info)
        self.board.push(move)
        self.game_over = self.board.is_game_over(claim_draw=True)
        self._record_position()
        return True

    def bot_move(self) -> bool:
        """Search-driven bot. Returns True if a move was played."""
        if self.board.is_game_over(claim_draw=True):
            self.game_over = True
            return False
        move, info = self.find_bot_move(self.board.copy())
        return self.apply_bot_move(move, info)
