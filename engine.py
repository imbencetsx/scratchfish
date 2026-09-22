import random
import chess

class ChessEngine:
    """Owns the game state. The GUI calls into this; it never draws."""

    def __init__(self, bot_is_white: bool = False):
        self.board = chess.Board()
        self.bot_is_white = bot_is_white
        self.game_over = False

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
        """Temporary random bot. Returns True if a move was played."""
        if self.board.is_game_over():
            self.game_over = True
            return False
        moves = list(self.board.legal_moves)
        if not moves:
            self.game_over = True
            return False
        self.board.push(random.choice(moves))
        if self.board.is_game_over():
            self.game_over = True
        return True
