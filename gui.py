"""Pygame GUI: draws the board and handles input. Game rules live in engine.py."""

import chess
import pygame

from engine import ChessEngine

# ----------------------------
# Settings
# ----------------------------

WIDTH = 720
HEIGHT = 780
BOARD_SIZE = 720
SQUARE_SIZE = BOARD_SIZE // 8

LIGHT = (240, 217, 181)
DARK = (181, 136, 99)
HIGHLIGHT = (246, 246, 105)
MOVE_DOT = (80, 80, 80)
TEXT = (30, 30, 30)
BACKGROUND = (235, 235, 235)

WHITE_PIECE = (250, 250, 250)
WHITE_OUTLINE = (20, 20, 20)
BLACK_PIECE = (20, 20, 20)
BLACK_OUTLINE = (250, 250, 250)

# Use the six solid glyphs and color them by side, instead of relying on
# the font's white-vs-black glyph variants (which many fonts lack or
# render inconsistently / as tofu boxes).
GLYPHS = {
    chess.PAWN: "\u265f",
    chess.KNIGHT: "\u265e",
    chess.BISHOP: "\u265d",
    chess.ROOK: "\u265c",
    chess.QUEEN: "\u265b",
    chess.KING: "\u265a",
}

# Fonts known to contain chess glyphs, best first.
# Arial alone often renders these as tofu, so it is last resort.
_PIECE_FONT_CANDIDATES = [
    "arialunicode",
    "applesymbols",  # macOS
    "dejavusans",  # Linux / bundled with many apps
    "freeserif",
    "segoeuisymbol",  # Windows
    "symbol",
    "arial",
]


def load_piece_font(size: int) -> pygame.font.Font:
    """Pick the first system font that actually contains chess glyphs."""
    for name in _PIECE_FONT_CANDIDATES:
        try:
            font = pygame.font.SysFont(name, size)
            metrics = font.metrics(GLYPHS[chess.KNIGHT])
            if metrics and metrics[0] is not None:
                return font
        except Exception:
            continue
    # Fallback: default pygame font (may still be tofu, but won't crash).
    return pygame.font.SysFont(None, size)


def render_with_outline(
    font: pygame.font.Font,
    glyph: str,
    fill: tuple[int, int, int],
    outline: tuple[int, int, int],
    outline_px: int = 2,
) -> pygame.Surface:
    """Render glyph in `fill` with an `outline` halo so it reads on any square."""
    base = font.render(glyph, True, fill)
    halo = font.render(glyph, True, outline)
    w, h = base.get_size()
    pad = outline_px * 2
    surf = pygame.Surface((w + pad, h + pad), pygame.SRCALPHA)
    cx, cy = pad // 2, pad // 2
    for dx in range(-outline_px, outline_px + 1):
        for dy in range(-outline_px, outline_px + 1):
            if dx == 0 and dy == 0:
                continue
            surf.blit(halo, (cx + dx, cy + dy))
    surf.blit(base, (cx, cy))
    return surf


class ChessGUI:
    def __init__(self, engine: ChessEngine):
        self.engine = engine
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Chess Bot")

        self.piece_font = load_piece_font(56)
        self.small_font = pygame.font.SysFont("arial", 22)
        self.button_font = pygame.font.SysFont("arial", 20)
        self.clock = pygame.time.Clock()

        self.selected_square: chess.Square | None = None
        self.new_game_rect = pygame.Rect(490, 728, 100, 38)
        self.switch_rect = pygame.Rect(600, 728, 100, 38)

    # ----------------------------
    # Helpers
    # ----------------------------

    @staticmethod
    def square_from_mouse(pos: tuple[int, int]) -> chess.Square | None:
        x, y = pos
        if y >= BOARD_SIZE:
            return None
        file = x // SQUARE_SIZE
        rank = 7 - (y // SQUARE_SIZE)
        return chess.square(file, rank)

    # ----------------------------
    # Drawing
    # ----------------------------

    def draw_board(self) -> None:
        for rank in range(8):
            for file in range(8):
                x = file * SQUARE_SIZE
                y = (7 - rank) * SQUARE_SIZE
                color = LIGHT if (file + rank) % 2 == 0 else DARK
                pygame.draw.rect(
                    self.screen, color, (x, y, SQUARE_SIZE, SQUARE_SIZE)
                )

    def draw_pieces(self) -> None:
        for square, piece in self.engine.board.piece_map().items():
            file = chess.square_file(square)
            rank = chess.square_rank(square)
            x = file * SQUARE_SIZE + SQUARE_SIZE // 2
            y = (7 - rank) * SQUARE_SIZE + SQUARE_SIZE // 2

            glyph = GLYPHS[piece.piece_type]
            if piece.color == chess.WHITE:
                surf = render_with_outline(
                    self.piece_font, glyph, WHITE_PIECE, WHITE_OUTLINE
                )
            else:
                surf = render_with_outline(
                    self.piece_font, glyph, BLACK_PIECE, BLACK_OUTLINE
                )
            rect = surf.get_rect(center=(x, y))
            self.screen.blit(surf, rect)

    def draw_selection(self) -> None:
        if self.selected_square is None:
            return
        file = chess.square_file(self.selected_square)
        rank = chess.square_rank(self.selected_square)
        x = file * SQUARE_SIZE
        y = (7 - rank) * SQUARE_SIZE
        pygame.draw.rect(
            self.screen, HIGHLIGHT, (x, y, SQUARE_SIZE, SQUARE_SIZE), 5
        )
        for target in self.engine.legal_targets(self.selected_square):
            cx = chess.square_file(target) * SQUARE_SIZE + SQUARE_SIZE // 2
            cy = (7 - chess.square_rank(target)) * SQUARE_SIZE + SQUARE_SIZE // 2
            pygame.draw.circle(self.screen, MOVE_DOT, (cx, cy), 10)

    def draw_status(self) -> None:
        pygame.draw.rect(
            self.screen, BACKGROUND, (0, BOARD_SIZE, WIDTH, HEIGHT - BOARD_SIZE)
        )
        text = self.small_font.render(self.engine.status_text(), True, TEXT)
        self.screen.blit(text, (20, 735))
        bot_text = self.small_font.render(
            f"Bot: {self.engine.bot_color_name()}", True, TEXT
        )
        self.screen.blit(bot_text, (250, 735))

    def draw_buttons(self) -> None:
        for rect, label in (
            (self.new_game_rect, "New Game"),
            (self.switch_rect, "Switch"),
        ):
            pygame.draw.rect(self.screen, (210, 210, 210), rect)
            text = self.button_font.render(label, True, TEXT)
            self.screen.blit(text, text.get_rect(center=rect.center))

    # ----------------------------
    # Input
    # ----------------------------

    def handle_click(self, pos: tuple[int, int]) -> None:
        x, y = pos
        if self.new_game_rect.collidepoint(x, y):
            self.engine.new_game()
            self.selected_square = None
            return
        if self.switch_rect.collidepoint(x, y):
            self.engine.switch_side()
            self.selected_square = None
            return

        if self.engine.bot_turn() or self.engine.game_over:
            return

        square = self.square_from_mouse(pos)
        if square is None:
            return

        if self.selected_square is None:
            piece = self.engine.board.piece_at(square)
            if piece and piece.color == self.engine.board.turn:
                self.selected_square = square
        else:
            self.engine.try_move(self.selected_square, square)
            self.selected_square = None

    # ----------------------------
    # Main loop
    # ----------------------------

    def run(self) -> None:
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_click(event.pos)

            if not self.engine.game_over and self.engine.bot_turn():
                self.engine.bot_move()

            self.screen.fill(BACKGROUND)
            self.draw_board()
            self.draw_selection()
            self.draw_pieces()
            self.draw_status()
            self.draw_buttons()
            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()
