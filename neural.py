"""Neural eval: board encoding + MLP (torch).

Input: 13 planes x 64 squares = 832 floats + 8 aux floats = 840.
  planes 0-5:  white P N B R Q K
  planes 6-11: black P N B R Q K
  plane 12:    side to move (all 1.0 if White to move else 0.0)
  aux: white-K, white-Q, black-K, black-Q castling rights,
       side-to-move in check, en-passant available,
       en-passant file (/7), halfmove clock (/100).
Output: tanh in [-1, 1], White's perspective (1 = White winning).

Trained in train.py by distilling the classical eval, then improved
via self-play later. Search (search.py) converts to side-to-move cp.
"""

from pathlib import Path

import chess
import torch
import torch.nn as nn

N_PLANES = 13
N_SQUARES = 64
N_AUX = 8
INPUT_SIZE = N_PLANES * N_SQUARES + N_AUX  # 840

WEIGHTS_PATH = Path(__file__).with_name("weights.pt")

# Order must match planes 0-5 / 6-11.
_PIECE_ORDER = [
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
    chess.KING,
]


def board_to_tensor(board: chess.Board) -> torch.Tensor:
    """Encode board as flat FloatTensor of shape (840,).

    Squares: index = plane*64 + square. Aux features appended last.
    """
    t = torch.zeros(INPUT_SIZE, dtype=torch.float32)
    for square, piece in board.piece_map().items():
        plane = _PIECE_ORDER.index(piece.piece_type)
        if piece.color == chess.BLACK:
            plane += 6
        t[plane * N_SQUARES + square] = 1.0
    if board.turn == chess.WHITE:
        t[12 * N_SQUARES : 13 * N_SQUARES] = 1.0
    base = N_PLANES * N_SQUARES
    t[base + 0] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    t[base + 1] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    t[base + 2] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    t[base + 3] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0
    t[base + 4] = 1.0 if board.is_check() else 0.0
    t[base + 5] = 1.0 if board.has_legal_en_passant() else 0.0
    # En-passant file (0-7 normalized) + halfmove clock (/100). Zero when n/a.
    if board.ep_square is not None:
        t[base + 6] = chess.square_file(board.ep_square) / 7.0
    t[base + 7] = min(board.halfmove_clock, 100) / 100.0
    return t


def cp_to_target(cp: int) -> float:
    """Map teacher centipawns to [-1, 1] training target."""
    cp = max(-1500, min(1500, cp))
    return cp / 1500.0


def target_to_cp(score: float) -> int:
    """Map net output [-1, 1] back to centipawns (White perspective)."""
    return int(max(-1.0, min(1.0, score)) * 1500)


class EvalNet(nn.Module):
    def __init__(self, input_size: int = INPUT_SIZE) -> None:
        super().__init__()
        self.input_size = input_size
        self.net = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@torch.no_grad()
def evaluate_white_nn(net: nn.Module, board: chess.Board) -> int:
    """NN eval in centipawns, White's perspective."""
    net.eval()
    score = net(board_to_tensor(board).unsqueeze(0)).item()
    return target_to_cp(score)


@torch.no_grad()
def evaluate_batch_nn(net: nn.Module, boards: list[chess.Board]) -> list[int]:
    """Batched NN eval (much faster inside search). Returns cp list."""
    net.eval()
    if not boards:
        return []
    x = torch.stack([board_to_tensor(b) for b in boards])
    scores = net(x).squeeze(1).tolist()
    if isinstance(scores, float):
        scores = [scores]
    return [target_to_cp(s) for s in scores]


def try_load_net(weights: Path = WEIGHTS_PATH) -> EvalNet | None:
    """Load trained weights. Returns None if missing or shape-mismatched.

    Shape mismatch happens after architecture upgrades (e.g. 832 -> 840
    inputs). In that case we warn and return None so the caller falls back
    to classical eval until you retrain.
    """
    if not weights.exists():
        return None
    try:
        state = torch.load(weights, map_location="cpu", weights_only=True)
        first_w = state.get("net.0.weight")
        if first_w is not None and first_w.shape[1] != INPUT_SIZE:
            print(
                f"Weights expect input {first_w.shape[1]}, "
                f"code wants {INPUT_SIZE} - retrain needed. Run train.py."
            )
            return None
        net = EvalNet()
        net.load_state_dict(state)
        net.eval()
        return net
    except Exception as e:  # corrupt / partial file
        print(f"Could not load {weights.name}: {e} - using classical eval.")
        return None
