import chess

from evaluation import evaluate_white_cp
from neural import WEIGHTS_PATH, evaluate_white_nn, try_load_net
from search import Searcher


class ChessEngine:
    """Owns the game state. The GUI calls into this; it never draws."""

    def __init__(
        self,
        bot_is_white: bool = False,
        depth: int = 3,
        time_limit: float = 0.8,
        use_nn: bool = True,
    ):
        self.board = chess.Board()
        self.bot_is_white = bot_is_white
        self.game_over = False
        self.depth = depth
        self.time_limit = time_limit

        # Neural brain if weights exist, else classical fallback.
        self.net = try_load_net() if use_nn else None
        self.use_nn = self.net is not None
        if use_nn and self.net is None:
            print(f"No {WEIGHTS_PATH.name} found - using classical eval. Run train.py.")
    # ----------------------------
    # Game control
    # ----------------------------

    def new_game(self) -> None:
        self.board = chess.Board()
        self.game_over = False

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
                outcome = self.board.outcome()
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
            return f"NN (depth {self.depth})"
        return f"classical (depth {self.depth})"

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
            if self.board.is_game_over():
                self.game_over = True
            return True
        return False

    def bot_move(self) -> bool:
        """Search-driven bot. Returns True if a move was played."""
        if self.board.is_game_over():
            self.game_over = True
            return False
        move, info = Searcher(
            self._eval_white_cp, time_limit=self.time_limit
        ).best_move(self.board, self.depth)
        if move is None:
            self.game_over = True
            return False
        self.board.push(move)
        if self.board.is_game_over():
            self.game_over = True
        return True
