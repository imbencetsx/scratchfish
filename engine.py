import chess

from evaluation import evaluate_white_cp
from neural import WEIGHTS_PATH, evaluate_white_nn, try_load_net
from search import Searcher


class ChessEngine:
    """Owns the game state. The GUI calls into this; it never draws."""

    def __init__(
        self,
        bot_is_white: bool = False,
        depth: int = 8,
        time_limit: float = 0.35,
        use_nn: bool = True,
    ):
        self.board = chess.Board()
        self.bot_is_white = bot_is_white
        self.game_over = False
        self.depth = depth
        self.time_limit = time_limit

        # A compact model guides root ordering; tactical leaves stay inexpensive.
        self.net = try_load_net() if use_nn else None
        self.use_nn = self.net is not None
        self.last_search_info = {}
        if use_nn and self.net is None:
            print(f"No compact {WEIGHTS_PATH.name} loaded; using classical evaluation.")

    # ----------------------------
    # Game control
    # ----------------------------

    def new_game(self) -> None:
        self.board = chess.Board()
        self.game_over = False
        self.last_search_info = {}

    def switch_side(self) -> None:
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
            self.board.push(move)
            if self.board.is_game_over(claim_draw=True):
                self.game_over = True
            return True
        return False

    def find_bot_move(self, board: chess.Board, stop_event=None):
        """Search a private board; safe to call in the GUI worker thread."""
        return Searcher(evaluate_white_cp, time_limit=self.time_limit,
                        stop_event=stop_event,
                        root_eval_white_cp=self._eval_white_cp if self.net is not None else None).best_move(board, self.depth)

    def apply_bot_move(self, move, info) -> bool:
        """Apply a completed search result on the main thread."""
        self.last_search_info = info
        if move is None or move not in self.board.legal_moves:
            self.game_over = self.board.is_game_over(claim_draw=True)
            return False
        self.board.push(move)
        self.game_over = self.board.is_game_over(claim_draw=True)
        return True

    def bot_move(self) -> bool:
        """Search-driven bot. Returns True if a move was played."""
        if self.board.is_game_over(claim_draw=True):
            self.game_over = True
            return False
        move, info = self.find_bot_move(self.board.copy())
        return self.apply_bot_move(move, info)
