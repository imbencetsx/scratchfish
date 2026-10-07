"""Compact residual evaluator, with sparse NumPy inference for root move ordering.

Material/position scoring stays exact. The learned correction is bounded to
200cp, so a small or poorly trained model cannot erase an extra queen.
The old weights.pt remains available for experiments with the legacy ResNet.
"""

from pathlib import Path

import chess
import numpy as np
import torch
import torch.nn as nn

from evaluation import evaluate_white_cp, _SQUARE_VALUES

N_PLANES = 20
BOARD_HW = 8
N_BLOCKS = 8
N_FILTERS = 256
INPUT_SIZE = 12 * 64 + 8
HIDDEN_SIZE = 64
RESIDUAL_CP = 200
WEIGHTS_PATH = Path(__file__).with_name("fast_weights.pt")
LEGACY_WEIGHTS_PATH = Path(__file__).with_name("weights.pt")


def sparse_features(board: chess.Board) -> tuple[list[int], np.ndarray]:
    indices = []
    for color in chess.COLORS:
        for piece in chess.PIECE_TYPES:
            offset = ((piece - 1) + (0 if color else 6)) * 64
            indices.extend(offset + sq for sq in chess.scan_forward(board.pieces_mask(piece, color)))
    aux = np.array([float(board.turn), float(board.is_check()),
                    float(board.has_kingside_castling_rights(chess.WHITE)),
                    float(board.has_queenside_castling_rights(chess.WHITE)),
                    float(board.has_kingside_castling_rights(chess.BLACK)),
                    float(board.has_queenside_castling_rights(chess.BLACK)),
                    (board.ep_square + 1) / 64 if board.ep_square is not None else 0,
                    min(board.halfmove_clock, 100) / 100], dtype=np.float32)
    return indices, aux


def board_to_tensor(board: chess.Board) -> torch.Tensor:
    indices, aux = sparse_features(board)
    values = np.zeros(INPUT_SIZE, dtype=np.float32)
    values[indices] = 1
    values[768:] = aux
    return torch.from_numpy(values)


def cp_to_target(cp: int) -> float:
    """Legacy teacher score mapping; compact training uses residual targets."""
    return max(-1500, min(1500, cp)) / 1500.0


def target_to_cp(score: float) -> int:
    return int(max(-1.0, min(1.0, score)) * 1500)


class EvalNet(nn.Module):
    """52k-parameter MLP predicting a bounded correction to classical eval."""
    def __init__(self):
        super().__init__()
        self.input_size = INPUT_SIZE
        self.net = nn.Sequential(nn.Linear(INPUT_SIZE, HIDDEN_SIZE), nn.ReLU(),
                                 nn.Linear(HIDDEN_SIZE, 32), nn.ReLU(),
                                 nn.Linear(32, 1), nn.Tanh())
        # An untrained model adds zero, preserving the handcrafted baseline.
        nn.init.zeros_(self.net[4].weight)
        nn.init.zeros_(self.net[4].bias)

    def forward(self, x):
        return self.net(x)


class FastEvaluator:
    """Fold a trained MLP into sparse CPU inference without per-node Torch calls."""
    def __init__(self, net: EvalNet, metadata=None):
        self.metadata = metadata or {}
        layers = [net.net[i] for i in (0, 2, 4)]
        self.weights = [layer.weight.detach().cpu().numpy().T.copy() for layer in layers]
        self.biases = [layer.bias.detach().cpu().numpy().copy() for layer in layers]

    def residual(self, board):
        indices, aux = sparse_features(board)
        # Sum only occupied piece features (~32), not a dense 776x64 multiply.
        w0, w1, w2 = self.weights
        b0, b1, b2 = self.biases
        x = np.maximum(w0[indices].sum(axis=0) + aux @ w0[768:] + b0, 0)
        x = np.maximum(x @ w1 + b1, 0)
        return float(np.tanh(x @ w2 + b2)[0])

    def _gradient(self, board):
        """Analytic input derivative of the tiny MLP; only computed at the root."""
        indices, aux = sparse_features(board)
        w0, w1, w2 = self.weights
        b0, b1, b2 = self.biases
        pre0 = w0[indices].sum(axis=0) + aux @ w0[768:] + b0
        pre1 = np.maximum(pre0, 0) @ w1 + b1
        score = float(np.tanh(np.maximum(pre1, 0) @ w2 + b2)[0])
        grad = w0 @ ((pre0 > 0) * (w1 @ ((pre1 > 0) * w2[:, 0])))
        return score, grad * (1 - score * score)

    def for_search(self, board, mode='compiled'):
        projected = ProjectedEvaluator(self, board)
        return projected if mode == 'projected' else CompiledEvaluator(projected)

    def __call__(self, board):
        # Color/vertical symmetry prevents a learned preference for one color.
        correction = (self.residual(board) - self.residual(board.mirror())) * 0.5
        return evaluate_white_cp(board) + int(RESIDUAL_CP * correction)


class ProjectedEvaluator:
    """Root-conditioned learned square values, with no neural calls at leaves.

    This is a local linear approximation, not exact NN inference. Build once
    per move; keep the root fixed for all iterative-deepening passes.
    """
    def __init__(self, evaluator, root):
        score, gradient = evaluator._gradient(root)
        mirrored_score, mirrored_gradient = evaluator._gradient(root.mirror())
        mirrored = np.empty(INPUT_SIZE, dtype=np.float32)
        for plane in range(12):
            source = (plane + 6) % 12
            mirrored[plane * 64:(plane + 1) * 64] = mirrored_gradient[
                source * 64 + (np.arange(64) ^ 56)]
        mirrored[768:] = mirrored_gradient[[768, 769, 772, 773, 770, 771, 774, 775]]
        mirrored[768] *= -1  # mirroring flips side-to-move
        gradient = (gradient - mirrored) * (RESIDUAL_CP / 2)
        # Ordinal en-passant encoding has no global linear mirror transform.
        gradient[774] = 0
        self.aux = tuple(float(value) for value in gradient[768:])
        self.tables = {
            (color, piece): tuple(float(value) for value in gradient[
                ((piece - 1) + (0 if color else 6)) * 64:
                ((piece - 1) + (0 if color else 6) + 1) * 64])
            for color in chess.COLORS for piece in chess.PIECE_TYPES
        }
        indices, aux = sparse_features(root)
        self.bias = (score - mirrored_score) * RESIDUAL_CP / 2 - float(
            gradient[indices].sum() + gradient[768:] @ aux)

    def __call__(self, board):
        rights = board.clean_castling_rights()
        values = (float(board.turn), float(board.is_check()),
                  float(bool(rights & chess.BB_H1)), float(bool(rights & chess.BB_A1)),
                  float(bool(rights & chess.BB_H8)), float(bool(rights & chess.BB_A8)), 0,
                  min(board.halfmove_clock, 100) / 100)
        bias = self.bias + sum(value * weight for value, weight in zip(values, self.aux))
        return evaluate_white_cp(board, correction_tables=self.tables,
                                 correction_bias=bias, correction_limit=RESIDUAL_CP)


class CompiledEvaluator:
    """Conservative integer projection fused into the existing square tables.

    Cap learned changes at 2cp per piece/square. Full NN still guides root
    ordering. Preparation is per root; leaf evaluation uses ordinary Python
    integer table lookups without a second feature-encoding pass.
    """
    def __init__(self, projected):
        self.tables = {}
        self.king_bonus = {}
        for (color, piece), weights in projected.tables.items():
            sign = 1 if color else -1
            adjustment = tuple(sign * int(round(max(-2, min(2, value)))) for value in weights)
            base = _SQUARE_VALUES[color, piece]
            if piece == chess.KING:
                self.tables[color, piece] = base
                self.king_bonus[color] = adjustment
            else:
                self.tables[color, piece] = tuple(value + change for value, change in zip(base, adjustment))
        self.bias = int(round(max(-20, min(20, projected.bias))))
        self.tempo = int(round(max(-2, min(2, projected.aux[0]))))

    def __call__(self, board):
        return evaluate_white_cp(board, square_tables=self.tables,
                                 king_bonus=self.king_bonus,
                                 extra=self.bias + (self.tempo if board.turn else 0))


@torch.inference_mode()
def evaluate_white_nn(net, board):
    if isinstance(net, FastEvaluator):
        return net(board)
    if isinstance(net, EvalNet):
        device = next(net.parameters()).device
        net.eval()
        boards = [board, board.mirror()]
        x = torch.stack([board_to_tensor(b) for b in boards]).to(device)
        scores = net(x).flatten().cpu().tolist()
        return evaluate_white_cp(board) + int(RESIDUAL_CP * (scores[0] - scores[1]) / 2)
    net.eval()
    device = next(net.parameters()).device
    return target_to_cp(net(board_to_planes(board).unsqueeze(0).to(device)).item())


def evaluate_batch_nn(net, boards):
    return [evaluate_white_nn(net, board) for board in boards]


def checkpoint(net, **metadata):
    return {"format_version": 1, "architecture": "compact-residual-64x32",
            "state_dict": net.state_dict(), "metadata": metadata}


def try_load_net(weights: Path = WEIGHTS_PATH):
    weights = Path(weights)
    if not weights.exists():
        return None
    try:
        state = torch.load(weights, map_location="cpu", weights_only=True)
        if state.get("architecture") == "compact-residual-64x32":
            net = EvalNet()
            net.load_state_dict(state["state_dict"])
            net.eval()
            return FastEvaluator(net, state.get("metadata", {}))
        # Explicit legacy loads are supported; gameplay never silently loads the big CNN.
        if "stem.0.weight" in state:
            filters = state["stem.0.weight"].shape[0]
            blocks = len({key.split('.')[1] for key in state if key.startswith("tower.")})
            net = LegacyEvalNet(blocks=blocks, filters=filters)
            net.load_state_dict(state)
            return net.eval()
        raise ValueError("unrecognized checkpoint architecture")
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        print(f"Could not load {weights.name}: {exc}; using classical evaluation.")
        return None


# Legacy encoding and architecture are retained only for explicit old-checkpoint comparisons.
def board_to_planes(board):
    values = np.zeros((20, 8, 8), dtype=np.float32)
    for color in chess.COLORS:
        for piece in chess.PIECE_TYPES:
            plane = piece - 1 + (0 if color else 6)
            for sq in chess.scan_forward(board.pieces_mask(piece, color)):
                values[plane, 7 - chess.square_rank(sq), chess.square_file(sq)] = 1
    _, aux = sparse_features(board)
    for plane, value in zip(range(12, 18), aux[:6]):
        values[plane] = value
    if board.ep_square is not None:
        values[18, 7 - chess.square_rank(board.ep_square), chess.square_file(board.ep_square)] = 1
    values[19] = aux[7]
    return torch.from_numpy(values)


class ResBlock(nn.Module):
    def __init__(self, filters=N_FILTERS):
        super().__init__()
        self.conv1 = nn.Conv2d(filters, filters, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(filters)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x))))) + x)


class LegacyEvalNet(nn.Module):
    def __init__(self, planes=N_PLANES, blocks=N_BLOCKS, filters=N_FILTERS):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(planes, filters, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(filters), nn.ReLU(inplace=True))
        self.tower = nn.Sequential(*[ResBlock(filters) for _ in range(blocks)])
        self.head = nn.Sequential(nn.Conv2d(filters, 32, 1, bias=False),
                                  nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.Flatten(),
                                  nn.Linear(2048, 512), nn.ReLU(inplace=True),
                                  nn.Linear(512, 256), nn.ReLU(inplace=True),
                                  nn.Linear(256, 1), nn.Tanh())

    def forward(self, x):
        return self.head(self.tower(self.stem(x)))
