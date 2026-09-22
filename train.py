"""Train the NN eval by distilling the classical eval.

Self-play with an epsilon-greedy teacher produces realistic positions;
the classical eval labels them. Result: a net that plays like the
classical eval (~fast beginner level), which self-play can improve later.

Upgrades in v2:
  - richer input (840 floats: pieces + castling/check/ep/clock)
  - bigger net (1024-512-256 + LayerNorm)
  - stronger teacher (--teacher-depth 1 or 2, blends eval + game outcome)
  - GPU/MPS auto, AdamW + cosine schedule, best-checkpoint saving
  - --resume to fine-tune existing weights

Usage:
    uv run python train.py --games 300 --epochs 40
    uv run python train.py --games 2000 --teacher-depth 2 --epochs 60  # overnight, stronger
    uv run python train.py --help
"""

import argparse
import random
from pathlib import Path

import chess
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, random_split, DataLoader

from evaluation import evaluate_white_cp
from neural import EvalNet, WEIGHTS_PATH, board_to_tensor, cp_to_target


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def greedy_move(board: chess.Board, depth: int = 1) -> chess.Move:
    """Teacher move maximizing side-to-move classical eval.

    depth=1: 1-ply greedy (fast, weak). depth=2: 2-ply minimax
    (slower, notably stronger - sees hanging pieces / simple tactics).
    """
    if depth <= 1:
        return _greedy_1ply(board)
    # 2-ply minimax: we move, opponent replies minimizing our score.
    # Our perspective = side to move at entry.
    we_are_white = board.turn == chess.WHITE
    best, best_score = None, None
    for move in board.legal_moves:
        board.push(move)
        if board.is_checkmate():
            score = 100_000
        elif board.is_stalemate() or board.is_insufficient_material():
            score = 0
        else:
            our_score = None
            for reply in board.legal_moves:
                board.push(reply)
                if board.is_checkmate():
                    s = -100_000  # we got mated
                elif board.is_stalemate() or board.is_insufficient_material():
                    s = 0
                else:
                    ev = evaluate_white_cp(board)
                    s = ev if we_are_white else -ev
                board.pop()
                our_score = s if our_score is None else min(our_score, s)
            score = our_score if our_score is not None else 0
        board.pop()
        if best_score is None or score > best_score:
            best, best_score = move, score
    assert best is not None
    return best


def _greedy_1ply(board: chess.Board) -> chess.Move:
    """1-ply teacher: move maximizing side-to-move classical eval."""
    best, best_score = None, None
    for move in board.legal_moves:
        board.push(move)
        if board.is_checkmate():
            score = 100_000
        elif board.is_stalemate() or board.is_insufficient_material():
            score = 0
        else:
            ev = evaluate_white_cp(board)
            # ev is post-move White perspective; the side that just
            # moved is the opposite of board.turn now.
            just_moved_white = board.turn == chess.BLACK
            score = ev if just_moved_white else -ev
        board.pop()
        if best_score is None or score > best_score:
            best, best_score = move, score
    assert best is not None
    return best


def play_game(eps: float, max_plies: int, teacher_depth: int) -> list[tuple[str, float]]:
    """One epsilon-greedy self-play game. Returns (fen, target) samples."""
    board = chess.Board()
    fens: list[str] = []
    plies = 0
    while not board.is_game_over() and plies < max_plies:
        fens.append(board.fen())
        moves = list(board.legal_moves)
        if random.random() < eps:
            board.push(random.choice(moves))
        else:
            board.push(greedy_move(board, depth=teacher_depth))
        plies += 1

    if board.is_checkmate():
        winner = chess.WHITE if board.turn == chess.BLACK else chess.BLACK
        terminal = 1.0 if winner == chess.WHITE else -1.0
    else:
        terminal = 0.0  # draw / cap

    # Label with classical eval, blended toward the true game outcome near
    # the end (last 10 plies ramp 0 -> 1). This teaches mating + drawing
    # technique instead of just copying the teacher's static eval.
    samples = []
    n = len(fens)
    for i, fen in enumerate(fens):
        b = chess.Board(fen)
        static = cp_to_target(evaluate_white_cp(b))
        dist_from_end = n - 1 - i
        w = max(0.0, 1.0 - dist_from_end / 10.0) if n else 1.0
        target = (1 - w) * static + w * terminal
        samples.append((fen, target))
    return samples


def build_dataset(games: int, eps: float, max_plies: int, teacher_depth: int, seed: int):
    random.seed(seed)
    samples: list[tuple[str, float]] = []
    for i in range(games):
        samples.extend(play_game(eps, max_plies, teacher_depth))
        if (i + 1) % 50 == 0:
            print(f"  generated {i + 1}/{games} games ({len(samples)} positions)")
    X = torch.stack([board_to_tensor(chess.Board(f)) for f, _ in samples])
    y = torch.tensor([t for _, t in samples], dtype=torch.float32).unsqueeze(1)
    return TensorDataset(X, y)


def train(args) -> None:
    device = pick_device(args.device)
    print(f"device={device}  games={args.games} teacher_depth={args.teacher_depth} eps={args.eps}")
    print(f"Generating {args.games} games (eps={args.eps})...")
    ds = build_dataset(args.games, args.eps, args.max_plies, args.teacher_depth, args.seed)
    print(f"Dataset: {len(ds)} positions")
    n_val = max(1, int(0.1 * len(ds)))
    train_ds, val_ds = random_split(
        ds, [len(ds) - n_val, n_val], generator=torch.Generator().manual_seed(0)
    )
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=2048)

    net = EvalNet().to(device)
    if args.resume and Path(args.resume).exists():
        try:
            state = torch.load(args.resume, map_location="cpu", weights_only=True)
            if state.get("net.0.weight") is not None and \
                    state["net.0.weight"].shape[1] == net.input_size:
                net.load_state_dict(state)
                print(f"Resumed from {args.resume} (fine-tuning).")
            else:
                print(f"{args.resume} has old architecture - training from scratch.")
        except Exception as e:
            print(f"Could not resume ({e}) - training from scratch.")

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.MSELoss()

    best_vl = float("inf")
    for epoch in range(1, args.epochs + 1):
        net.train()
        for X, y in train_dl:
            X, y = X.to(device), y.to(device)
            opt.zero_grad()
            loss_fn(net(X), y).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            tot, mae = 0.0, 0.0
            for X, y in val_dl:
                X, y = X.to(device), y.to(device)
                out = net(X)
                tot += loss_fn(out, y).item() * len(X)
                mae += (out - y).abs().sum().item()
            vl = tot / len(val_ds)
            mae /= len(val_ds)
        print(f"epoch {epoch:2d}/{args.epochs}  val_mse={vl:.5f} val_mae={mae:.4f} lr={sched.get_last_lr()[0]:.2e}")
        if vl < best_vl:
            best_vl = vl
            torch.save(net.state_dict(), args.out)
            print(f"  -> new best, saved {args.out}")

    print(f"Best val_mse={best_vl:.5f} -> {args.out}")

    # Sanity: startpos ~ drawish, e4-d5 ~ equal, mate-in-1 position ~ winning.
    net.eval()
    with torch.no_grad():
        s0 = net(board_to_tensor(chess.Board()).unsqueeze(0).to(device)).item()
        b = chess.Board("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
        s1 = net(board_to_tensor(b).unsqueeze(0).to(device)).item()
        mate = chess.Board("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1")
        s2 = net(board_to_tensor(mate).unsqueeze(0).to(device)).item()
    print(f"sanity  startpos={s0:+.3f} (want ~0), after-1.e4-d5={s1:+.3f}, rook-up={s2:+.3f} (want >+0.5)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=int, default=300)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--eps", type=float, default=0.25)
    p.add_argument("--max-plies", type=int, default=120)
    p.add_argument("--teacher-depth", type=int, default=1, choices=[1, 2],
                   help="1=fast greedy, 2=2-ply minimax (slower, stronger)")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", help="auto|cpu|mps|cuda")
    p.add_argument("--resume", default=str(WEIGHTS_PATH),
                   help="path to existing weights to fine-tune ('' to disable)")
    p.add_argument("--out", default=str(WEIGHTS_PATH))
    train(p.parse_args())


if __name__ == "__main__":
    main()
