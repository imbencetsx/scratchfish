"""Fast bitboard evaluation: material, tapered king placement and pawn structure."""

from functools import lru_cache

import chess

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}

# Tables below are written rank-8-first (a8..h8, ..., a1..h1),
# from White's perspective, in centipawns.
_PAWN = [
    0, 0, 0, 0, 0, 0, 0, 0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
    5, 5, 10, 25, 25, 10, 5, 5,
    0, 0, 0, 20, 20, 0, 0, 0,
    5, -5, -10, 0, 0, -10, -5, 5,
    5, 10, 10, -20, -20, 10, 10, 5,
    0, 0, 0, 0, 0, 0, 0, 0,
]
_KNIGHT = [
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20, 0, 0, 0, 0, -20, -40,
    -30, 0, 10, 15, 15, 10, 0, -30,
    -30, 5, 15, 20, 20, 15, 5, -30,
    -30, 0, 15, 20, 20, 15, 0, -30,
    -30, 5, 10, 15, 15, 10, 5, -30,
    -40, -20, 0, 5, 5, 0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50,
]
_BISHOP = [
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -10, 0, 5, 10, 10, 5, 0, -10,
    -10, 5, 5, 10, 10, 5, 5, -10,
    -10, 0, 10, 10, 10, 10, 0, -10,
    -10, 10, 10, 10, 10, 10, 10, -10,
    -10, 5, 0, 0, 0, 0, 5, -10,
    -20, -10, -10, -10, -10, -10, -10, -20,
]
_ROOK = [
    0, 0, 0, 0, 0, 0, 0, 0,
    5, 10, 10, 10, 10, 10, 10, 5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    0, 0, 0, 5, 5, 0, 0, 0,
]
_QUEEN = [
    -20, -10, -10, -5, -5, -10, -10, -20,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -10, 0, 5, 5, 5, 5, 0, -10,
    -5, 0, 5, 5, 5, 5, 0, -5,
    0, 0, 5, 5, 5, 5, 0, -5,
    -10, 5, 5, 5, 5, 5, 0, -10,
    -10, 0, 5, 0, 0, 0, 0, -10,
    -20, -10, -10, -5, -5, -10, -10, -20,
]
_KING = [
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -10, -20, -20, -20, -20, -20, -20, -10,
    20, 20, 0, 0, 0, 0, 20, 20,
    20, 30, 10, 0, 0, 10, 30, 20,
]

PST = {
    chess.PAWN: _PAWN,
    chess.KNIGHT: _KNIGHT,
    chess.BISHOP: _BISHOP,
    chess.ROOK: _ROOK,
    chess.QUEEN: _QUEEN,
    chess.KING: _KING,
}


def _pst_value(piece_type: int, square: chess.Square, color: bool) -> int:
    table = PST[piece_type]
    rank = chess.square_rank(square)
    file = chess.square_file(square)
    if color == chess.WHITE:
        return table[(7 - rank) * 8 + file]
    return table[rank * 8 + file]


# Precompute signed square values; avoid constructing Piece objects at every leaf.
_SQUARE_VALUES = {
    (color, piece): tuple(PIECE_VALUES[piece] + _pst_value(piece, sq, color)
                          for sq in chess.SQUARES)
    for color in chess.COLORS for piece in chess.PIECE_TYPES
}
_PHASE = {chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 1,
          chess.ROOK: 2, chess.QUEEN: 4, chess.KING: 0}
_KING_END = tuple(int(35 - 12 * (abs(chess.square_file(sq) - 3.5)
                                + abs(chess.square_rank(sq) - 3.5)))
                  for sq in chess.SQUARES)
_PASSED = (0, 5, 10, 20, 35, 60, 100, 0)


@lru_cache(maxsize=8192)
def _pawn_structure(white_pawns: int, black_pawns: int) -> int:
    pawns = [black_pawns, white_pawns]
    score = 0
    for color in chess.COLORS:
        own_pawns, enemy_pawns = pawns[color], pawns[not color]
        subtotal = 0
        for file in range(8):
            file_pawns = own_pawns & chess.BB_FILES[file]
            count = file_pawns.bit_count()
            if not count:
                continue
            subtotal -= 12 * (count - 1)
            neighbors = (chess.BB_FILES[file - 1] if file else 0) | (chess.BB_FILES[file + 1] if file < 7 else 0)
            if not own_pawns & neighbors:
                subtotal -= 10 * count
            for sq in chess.scan_forward(file_pawns):
                rank = chess.square_rank(sq)
                ahead = (chess.BB_ALL << ((rank + 1) * 8)) & chess.BB_ALL if color else (1 << (rank * 8)) - 1
                if not enemy_pawns & ahead & (chess.BB_FILES[file] | neighbors):
                    subtotal += _PASSED[rank if color else 7 - rank]
        score += subtotal if color else -subtotal
    return score


def evaluate_white_cp(board: chess.Board) -> int:
    """White centipawns; search handles mate/draws. No neural model is needed."""
    score = _pawn_structure(board.pawns & board.occupied_co[chess.WHITE],
                            board.pawns & board.occupied_co[chess.BLACK])
    phase = min(24, board.knights.bit_count() + board.bishops.bit_count()
                + 2 * board.rooks.bit_count() + 4 * board.queens.bit_count())
    pawns = [board.pawns & board.occupied_co[color] for color in chess.COLORS]
    for color in chess.COLORS:
        ours = board.occupied_co[color]
        sign = 1 if color else -1
        subtotal = 0
        for piece in chess.PIECE_TYPES:
            mask = board.pieces_mask(piece, color)
            table = _SQUARE_VALUES[color, piece]
            if piece == chess.KING:
                for sq in chess.scan_forward(mask):
                    subtotal += (table[sq] * phase + _KING_END[sq] * (24 - phase)) // 24
            else:
                subtotal += sum(table[sq] for sq in chess.scan_forward(mask))
        if (board.bishops & ours).bit_count() >= 2:
            subtotal += 30
        own_pawns, enemy_pawns = pawns[color], pawns[not color]
        for sq in chess.scan_forward(board.rooks & ours):
            file_mask = chess.BB_FILES[chess.square_file(sq)]
            if not own_pawns & file_mask:
                subtotal += 12 if enemy_pawns & file_mask else 22
        # Pawn shelter matters with queens/minor pieces, but fades in endings.
        king = board.king(color)
        if king is not None and phase >= 12:
            shelter = chess.BB_KING_ATTACKS[king] & own_pawns
            subtotal += shelter.bit_count() * phase // 3
        score += sign * subtotal
    # In pawnless winning endings, bring our king closer and drive theirs to the edge.
    if not board.pawns and abs(score) > 250:
        wk, bk = board.king(chess.WHITE), board.king(chess.BLACK)
        if wk is not None and bk is not None:
            loser = bk if score > 0 else wk
            edge = max(abs(chess.square_file(loser) - 3.5), abs(chess.square_rank(loser) - 3.5))
            bonus = int(10 * edge + 8 * (7 - chess.square_distance(wk, bk)))
            score += bonus if score > 0 else -bonus
    return score + (10 if board.turn else -10)
