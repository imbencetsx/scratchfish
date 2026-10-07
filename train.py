"""Train a small residual evaluator from time/node-bounded self-play search.

No Stockfish required. Reuse --dataset to skip game generation on later runs.
Existing large weights.pt is never overwritten by the default output.
"""

import argparse
import contextlib
import multiprocessing as mp
import os
import random
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import chess
import chess.engine
import torch
import torch.nn as nn

from evaluation import evaluate_white_cp
from neural import EvalNet, WEIGHTS_PATH, RESIDUAL_CP, INPUT_SIZE, board_to_tensor, checkpoint
from search import Searcher


class BudgetExpired(Exception):
    pass


def atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def pick_device(name):
    if name != 'auto':
        return torch.device(name)
    # Tiny dense layers usually cost less on CPU than dispatching Metal kernels.
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def greedy_move(board, depth=1):
    move, _ = Searcher(evaluate_white_cp, .02, node_limit=512).best_move(board, depth)
    return move


def _generate_game(job):
    game_id, config, deadline = job
    rng = random.Random(config['seed'] + game_id * 7919)
    board = chess.Board()
    samples = []
    teacher = config['teacher']
    sf = None
    if time.monotonic() >= deadline:
        return []
    try:
        if teacher == 'stockfish':
            sf = chess.engine.SimpleEngine.popen_uci(config['sf_path'])
            sf.configure({'Threads': 1, 'Hash': 16})
        searcher = Searcher(evaluate_white_cp, config['teacher_ms'] / 1000,
                            node_limit=config['teacher_nodes'])
        for ply in range(config['max_plies']):
            if time.monotonic() >= deadline or board.is_game_over(claim_draw=True):
                break
            # Randomized opening avoids dozens of copies of the same starting line.
            if ply < 4:
                board.push(rng.choice(list(board.legal_moves)))
                continue
            sampled = (ply - 4) % config['sample_every'] == 0
            if sf is not None and sampled:
                limit = chess.engine.Limit(depth=config['sf_depth'],
                                           time=min(config['sf_ms'] / 1000,
                                                    max(.001, deadline - time.monotonic())))
                info = sf.analyse(board, limit)
                white_cp = info['score'].white().score(mate_score=100_000)
                move = info['pv'][0] if info.get('pv') else greedy_move(board)
                searched = True
            else:
                searcher.time_limit = min(config['teacher_ms'] / 1000,
                                          max(0, deadline - time.monotonic()))
                move, info = searcher.best_move(board, config['teacher_depth'])
                searched = info['depth'] > 0
                white_cp = info['score_cp'] * (1 if board.turn else -1)
                if teacher == 'classical':
                    white_cp = evaluate_white_cp(board)
                    searched = True
            if move is None:
                break
            if sampled and searched:
                # Search labels include tactics. Capped/unfinished games are never called draws.
                # Keep mate targets bounded; search itself always handles exact mates.
                base = evaluate_white_cp(board)
                target = max(-1.0, min(1.0, (white_cp - base) / RESIDUAL_CP))
                samples.append((board.fen(), target, game_id))
            if rng.random() < config['eps']:
                move = rng.choice(list(board.legal_moves))
            board.push(move)
    finally:
        if sf is not None:
            sf.quit()
    return samples


def build_dataset(args, round_no, deadline):
    cache = Path(args.dataset) if args.dataset else None
    if cache and cache.exists() and not args.rebuild_dataset:
        data = torch.load(cache, map_location='cpu', weights_only=True)
        if data.get('format_version') != 1:
            raise ValueError('Unsupported dataset format; use --rebuild-dataset.')
        if (data['X'].ndim != 2 or data['X'].shape[1] != INPUT_SIZE
                or data['y'].shape != (len(data['X']), 1)
                or len(data['groups']) != len(data['X']) or len(data['X']) % 2
                or not torch.equal(data['groups'][::2], data['groups'][1::2])
                or data['groups'].unique().numel() < 2):
            raise ValueError('Invalid compact dataset; use --rebuild-dataset.')
        print(f"Reusing {cache}: {len(data['y'])} positions (generation skipped).", flush=True)
        return data
    config = {name: getattr(args, name) for name in (
        'teacher', 'teacher_depth', 'teacher_ms', 'teacher_nodes', 'max_plies',
        'sample_every', 'eps', 'sf_path', 'sf_depth', 'sf_ms')}
    config['seed'] = args.seed + (round_no - 1) * 1000
    jobs = [(game, config, deadline) for game in range(args.games)]
    samples = []
    started = time.monotonic()
    print(f"Generating {args.games} games: {args.teacher} teacher, "
          f"depth {args.teacher_depth}, {args.teacher_ms:g}ms/{args.teacher_nodes} nodes per move.", flush=True)
    workers = min(args.workers, args.games)
    if workers == 1:
        for job in jobs:
            samples.extend(_generate_game(job))
            if time.monotonic() >= deadline:
                break
    else:
        # Spawn is safe after Torch/MPS/CUDA initialization, unlike fork.
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn')) as pool:
            futures = [pool.submit(_generate_game, job) for job in jobs]
            try:
                for done, future in enumerate(as_completed(futures), 1):
                    samples.extend(future.result())
                    if done % max(1, args.games // 10) == 0:
                        print(f"  {done}/{args.games} games, {len(samples)} samples, "
                              f"{time.monotonic() - started:.1f}s", flush=True)
                    if time.monotonic() >= deadline:
                        for pending in futures:
                            pending.cancel()
                        break
            except KeyboardInterrupt:
                for pending in futures:
                    pending.cancel()
                # Running workers see the deadline; explicitly stop on Ctrl+C.
                pool.terminate_workers()
                raise
    if len({group for _, _, group in samples}) < 2:
        if time.monotonic() >= deadline:
            raise BudgetExpired
        raise ValueError('Need samples from at least two games; increase the teacher budget or max plies.')
    samples.sort(key=lambda sample: (sample[2], sample[0]))
    xs, ys, groups = [], [], []
    for fen, target, group in samples:
        if time.monotonic() >= deadline:
            raise BudgetExpired
        board = chess.Board(fen)
        xs.extend([board_to_tensor(board), board_to_tensor(board.mirror())])
        ys.extend([target, -target])
        groups.extend([group, group])
    data = {'format_version': 1, 'X': torch.stack(xs),
            'y': torch.tensor(ys, dtype=torch.float32).unsqueeze(1),
            'groups': torch.tensor(groups), 'config': config}
    print(f"Dataset: {len(ys)} positions including color mirrors in "
          f"{time.monotonic() - started:.1f}s.", flush=True)
    if cache:
        atomic_save(data, cache)
        print(f"Cached dataset at {cache}.", flush=True)
    return data


def load_weights_into(net, path):
    if not path or not Path(path).exists():
        return False
    try:
        state = torch.load(path, map_location='cpu', weights_only=True)
        if state.get('architecture') != 'compact-residual-64x32':
            print(f"{path} uses a legacy architecture; preserved, starting compact model from scratch.")
            return False
        net.load_state_dict(state['state_dict'])
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f"Cannot resume {path}: {exc}; starting compact model from scratch.")
        return False
    print(f"Resumed compact model from {path}.")
    return True


def _state(net):
    return {key: value.detach().cpu().clone() for key, value in net.state_dict().items()}


def train_round(net, args, device, round_no, deadline):
    data = build_dataset(args, round_no, deadline)
    ids = data['groups'].unique()
    ids = ids[torch.randperm(len(ids), generator=torch.Generator().manual_seed(args.seed))]
    val_ids = ids[:max(1, len(ids) // 5)]
    val_mask = torch.isin(data['groups'], val_ids)
    # Split by whole games, keeping mirrored pairs and adjacent positions together.
    train_x, train_y = data['X'][~val_mask].to(device), data['y'][~val_mask].to(device)
    val_x, val_y = data['X'][val_mask].to(device), data['y'][val_mask].to(device)
    baseline = val_y.square().mean().item()
    print(f"Split: {len(train_y)} train/{len(val_y)} held-out positions by game; "
          f"classical baseline mse={baseline:.5f}.", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4,
                            fused=(device.type == 'cuda'))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.MSELoss()
    use_amp = args.amp and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    def autocast():
        return torch.autocast('cuda') if use_amp else contextlib.nullcontext()

    def validate():
        net.eval()
        total = 0.0
        with torch.inference_mode():
            for start in range(0, len(val_y), args.batch):
                with autocast():
                    indices = torch.arange(start, min(start + args.batch, len(val_y)), device=device)
                    both = net(torch.cat([val_x[indices], val_x[indices ^ 1]]))
                    left, right = both.chunk(2)
                    out = (left - right) * .5
                total += (out - val_y[start:start + args.batch]).square().sum().item()
        return total / len(val_y)

    best_loss, best_state, stale = validate(), _state(net), 0
    initial = best_loss
    trained_steps = 0
    started = time.monotonic()
    try:
        for epoch in range(1, args.epochs + 1):
            if time.monotonic() >= deadline:
                break
            net.train()
            order = torch.randperm(len(train_y), device=device)
            for start in range(0, len(train_y), args.batch):
                if time.monotonic() >= deadline:
                    break
                batch = order[start:start + args.batch]
                opt.zero_grad(set_to_none=True)
                with autocast():
                    both = net(torch.cat([train_x[batch], train_x[batch ^ 1]]))
                    left, right = both.chunk(2)
                    loss = loss_fn((left - right) * .5, train_y[batch])
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(net.parameters(), 1)
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), 1)
                    opt.step()
                trained_steps += 1
            sched.step()
            val_loss = validate()
            improved = val_loss < best_loss
            if improved:
                best_loss, best_state, stale = val_loss, _state(net), 0
                if best_loss < baseline:
                    atomic_save(checkpoint(net, val_mse=best_loss, baseline_mse=baseline,
                                           round=round_no, teacher=args.teacher,
                                           seed=args.seed, trained_steps=trained_steps), args.out)
            else:
                stale += 1
            print(f"Epoch {epoch:2d}/{args.epochs}: val_mse={val_loss:.5f}, "
                  f"residual rmse={RESIDUAL_CP * val_loss ** .5:.1f}cp"
                  f"{'  best saved' if improved and best_loss < baseline else ''}", flush=True)
            if stale >= args.patience:
                print('Early stop: validation stopped improving.', flush=True)
                break
    finally:
        # On budget expiry or Ctrl+C keep the best validated weights, not the last minibatch.
        net.load_state_dict(best_state)
    print(f"Training {time.monotonic() - started:.2f}s; best mse={best_loss:.5f}, "
          f"initial={initial:.5f}, baseline={baseline:.5f}.", flush=True)
    if best_loss >= baseline:
        print('Model did not beat classical baseline on held-out labels; checkpoint unchanged.', flush=True)
    return best_loss


def train(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    started = time.monotonic()
    deadline = started + args.minutes * 60 if args.minutes > 0 else float('inf')
    if args.teacher == 'stockfish':
        args.sf_path = args.sf_path or shutil.which('stockfish')
        if not args.sf_path:
            raise ValueError('Stockfish not found. Use --teacher search or provide --sf-path.')
    net = EvalNet().to(device)
    if not args.fresh:
        load_weights_into(net, args.resume)
    if args.compile:
        if device.type in ('cpu', 'cuda'):
            net.forward = torch.compile(net.forward)
            print('Compiling forward pass; initial compilation is included in the time budget.')
        else:
            print('Compile skipped on MPS; using eager execution.')
    print(f"device={device}, params={sum(p.numel() for p in net.parameters()):,}, "
          f"threads={args.threads}, output={args.out}", flush=True)
    print('Each improving validated epoch is saved atomically. Ctrl+C keeps the best checkpoint.', flush=True)
    round_no = 0
    try:
        while (args.rounds == 0 or round_no < args.rounds) and time.monotonic() < deadline:
            round_no += 1
            train_round(net, args, device, round_no, deadline)
    except BudgetExpired:
        print('Time budget reached during generation; existing weights untouched.')
    except KeyboardInterrupt:
        print('\nInterrupted; last validated checkpoint preserved.')
    print(f"Done: {round_no} round(s), {time.monotonic() - started:.1f}s.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--games', type=int, default=80)
    p.add_argument('--epochs', type=int, default=16)
    p.add_argument('--rounds', type=int, default=1, help='0 = continue until budget/Ctrl+C')
    p.add_argument('--minutes', type=float, default=0)
    p.add_argument('--eps', type=float, default=.12)
    p.add_argument('--max-plies', type=int, default=100)
    p.add_argument('--sample-every', type=int, default=4)
    p.add_argument('--teacher', choices=['search', 'classical', 'stockfish'], default='search')
    p.add_argument('--teacher-depth', type=int, default=2)
    p.add_argument('--teacher-ms', type=float, default=8)
    p.add_argument('--teacher-nodes', type=int, default=256)
    p.add_argument('--sf-path', default='')
    p.add_argument('--sf-depth', type=int, default=12)
    p.add_argument('--sf-ms', type=float, default=20, help='hard time bound for optional Stockfish labels')
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--batch', type=int, default=128)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='auto', choices=['auto', 'cpu', 'mps', 'cuda'])
    p.add_argument('--resume', default=str(WEIGHTS_PATH))
    p.add_argument('--fresh', action='store_true', help='ignore resume weights; output only changes if validation improves')
    p.add_argument('--workers', type=int, default=min(4, os.cpu_count() or 1))
    p.add_argument('--threads', type=int, default=2, help='Torch CPU threads; small models do not need all cores')
    p.add_argument('--patience', type=int, default=4)
    p.add_argument('--dataset', default='', help='cache/reuse encoded dataset at this path')
    p.add_argument('--rebuild-dataset', action='store_true')
    p.add_argument('--amp', action='store_true', help='mixed precision on CUDA')
    p.add_argument('--compile', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--out', default=str(WEIGHTS_PATH))
    args = p.parse_args()
    for name in ('games', 'epochs', 'max_plies', 'sample_every', 'teacher_depth', 'teacher_nodes',
                 'batch', 'workers', 'threads', 'patience', 'sf_depth'):
        if getattr(args, name) <= 0:
            p.error(f'--{name.replace("_", "-")} must be positive')
    if args.games < 2 or args.max_plies < 6:
        p.error('Use at least 2 games and 6 plies for a game-separated validation split.')
    if not 0 <= args.eps <= 1 or args.teacher_ms <= 0 or args.sf_ms <= 0 or args.lr <= 0:
        p.error('eps must be in [0,1]; time limits and learning rate must be positive')
    if args.minutes < 0 or args.rounds < 0:
        p.error('minutes and rounds must be nonnegative')
    # Never let a compact run accidentally replace the original CNN checkpoint.
    if Path(args.out).resolve() == Path(__file__).with_name('weights.pt').resolve():
        p.error('Use a separate compact output (default: fast_weights.pt); weights.pt is the legacy checkpoint.')
    train(args)


if __name__ == '__main__':
    main()
