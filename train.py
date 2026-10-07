"""Train a small residual evaluator from time/node-bounded self-play search.

No Stockfish required. Reuse --dataset to skip game generation on later runs.
Existing large weights.pt is never overwritten by the default output.
"""

import argparse
import atexit
import contextlib
import multiprocessing as mp
import os
import random
import re
import shutil
import time
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from threading import Thread
from pathlib import Path
from multiprocessing.util import Finalize
import traceback

import chess
import chess.engine
import torch
import torch.nn as nn

from evaluation import evaluate_white_cp
from neural import EvalNet, WEIGHTS_PATH, RESIDUAL_CP, INPUT_SIZE, board_to_tensor, checkpoint
from search import Searcher
from runlog import EventLog, console_log, emit, new_run_directory, save_game


QUALITY_PRESETS = {
    'fast': dict(teacher_depth=2, teacher_ms=8, teacher_nodes=256, sf_depth=12, sf_ms=20),
    'balanced': dict(teacher_depth=3, teacher_ms=20, teacher_nodes=1024, sf_depth=14, sf_ms=60),
    'deep': dict(teacher_depth=5, teacher_ms=100, teacher_nodes=8192, sf_depth=18, sf_ms=150),
}
_SF_ENGINE = None
_SF_PATH = None
_SF_SETTINGS = None


def _close_stockfish():
    global _SF_ENGINE
    if _SF_ENGINE is not None:
        try:
            _SF_ENGINE.quit()
        except (chess.engine.EngineError, TimeoutError):
            pass
        finally:
            _SF_ENGINE.close()
            _SF_ENGINE = None


def _stockfish(path, threads=1, hash_mb=32):
    global _SF_ENGINE, _SF_PATH, _SF_SETTINGS
    if _SF_ENGINE is None or _SF_PATH != path or _SF_SETTINGS != (threads, hash_mb):
        _close_stockfish()
        # python-chess inherits daemon status for its UCI event-loop thread.
        # A non-daemon thread would block worker exit before Finalize can quit it.
        ready = Future()
        def launch():
            try:
                ready.set_result(chess.engine.SimpleEngine.popen_uci(path))
            except BaseException as exc:
                ready.set_exception(exc)
        Thread(target=launch, daemon=True, name='stockfish-launch').start()
        _SF_ENGINE = ready.result()
        _SF_ENGINE.configure({'Threads': threads, 'Hash': hash_mb})
        _SF_SETTINGS = (threads, hash_mb)
        _SF_PATH = path
    return _SF_ENGINE


atexit.register(_close_stockfish)


def _init_worker():
    torch.set_num_threads(1)
    # ProcessPool workers do not reliably run ordinary atexit callbacks.
    Finalize(None, _close_stockfish, exitpriority=10)


class BudgetExpired(Exception):
    pass


def parse_duration(value):
    """Read positive durations such as 30s, 90m, 2h or 1.5d."""
    match = re.fullmatch(r'([0-9]+(?:\.[0-9]+)?)\s*([smhd])', value.strip().lower())
    if not match:
        raise argparse.ArgumentTypeError('Use a duration such as 30s, 90m, 2h or 1d.')
    seconds = float(match[1]) * {'s': 1, 'm': 60, 'h': 3600, 'd': 86400}[match[2]]
    if not 0 < seconds < float('inf'):
        raise argparse.ArgumentTypeError('Training duration must be positive and finite.')
    return seconds


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
    available = {'cuda': torch.cuda.is_available(), 'mps': torch.backends.mps.is_available(), 'cpu': True}
    if name == 'auto':
        name = next(device for device in ('cuda', 'mps', 'cpu') if available[device])
    if not available[name]:
        raise ValueError(f'{name} is unavailable in this PyTorch installation; use --device auto or cpu.')
    return torch.device(name)


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
    started = time.monotonic()
    directory = Path(config['game_log_dir']) if config.get('game_log_dir') else None
    log = EventLog(directory / f'game-{game_id:06d}.jsonl') if directory else None
    full = config.get('log_detail', 'full') == 'full'
    reason = 'max_plies'
    def event(name, **fields):
        if log:
            log.event(name, game=game_id, **fields)
    event('game_start', config=config)
    try:
        if time.monotonic() >= deadline:
            reason = 'deadline'
            return []
        if teacher == 'stockfish':
            sf = _stockfish(config['sf_path'], config.get('sf_threads', 1), config.get('sf_hash', 32))
            event('stockfish_ready', engine=sf.id, pid=sf.transport.get_pid(), threads=config.get('sf_threads', 1), hash_mb=config.get('sf_hash', 32))
        searcher = Searcher(evaluate_white_cp, config['teacher_ms'] / 1000,
                            node_limit=config['teacher_nodes'])
        for ply in range(config['max_plies']):
            if time.monotonic() >= deadline:
                reason = 'deadline'
                break
            if board.is_game_over(claim_draw=True):
                reason = 'game_over'
                break
            move_started = time.monotonic()
            fen = board.fen()
            info = {}
            white_cp = None
            target = None
            sampled = ply >= 4 and (ply - 4) % config['sample_every'] == 0
            searched = False
            if ply < 4:
                move = rng.choice(list(board.legal_moves))
                skip = 'random_opening'
            else:
                if sf is not None and sampled:
                    limit = chess.engine.Limit(depth=config['sf_depth'],
                                               time=min(config['sf_ms'] / 1000,
                                                        max(.001, deadline - time.monotonic())))
                    info = sf.analyse(board, limit)
                    white_cp = info['score'].white().score(mate_score=100_000)
                    move = info['pv'][0] if info.get('pv') else greedy_move(board)
                    searched = not board.is_check() and not info['score'].is_mate()
                elif sf is not None:
                    result = sf.play(board, chess.engine.Limit(time=min(config.get('sf_play_ms', 5) / 1000,
                                               max(.001, deadline - time.monotonic()))), info=chess.engine.INFO_ALL if log and full else chess.engine.INFO_NONE)
                    move, info = result.move, result.info
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
                    reason = 'no_move'
                    break
                base = evaluate_white_cp(board)
                if not sampled:
                    skip = 'not_sampled'
                elif not searched:
                    skip = 'check_mate_or_unsearched'
                elif board.is_capture(move):
                    skip = 'capture'
                elif board.gives_check(move):
                    skip = 'checking_move'
                elif abs(white_cp - base) > RESIDUAL_CP * 2:
                    skip = 'residual_out_of_range'
                else:
                    skip = None
                    target = max(-1.0, min(1.0, (white_cp - base) / RESIDUAL_CP))
                    samples.append((fen, target, game_id))
                teacher_move = move
                if rng.random() < config['eps']:
                    move = rng.choice(list(board.legal_moves))
            if full:
                event('move', ply=ply, fen=fen, move=move.uci(), san=board.san(move),
                      teacher_move=teacher_move.uci() if ply >= 4 else None,
                      sampled=sampled, target=target, skip_reason=skip, white_cp=white_cp,
                      search=info, elapsed_seconds=time.monotonic() - move_started)
            elif target is not None:
                event('label', ply=ply, fen=fen, target=target, white_cp=white_cp)
            board.push(move)
    except BaseException as exc:
        reason = 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'error'
        event('error', message=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        outcome = board.outcome(claim_draw=True)
        if outcome:
            reason = outcome.termination.name.lower()
        event('game_end', reason=reason, result=outcome.result() if outcome else '*',
              plies=len(board.move_stack), samples=len(samples), final_fen=board.fen(),
              elapsed_seconds=time.monotonic() - started)
        if directory:
            try:
                save_game(board, directory / f'game-{game_id:06d}.pgn', Event='Scratchfish training',
                          Round=f"{config.get('round', 1)}.{game_id}", Termination=reason,
                          White=teacher, Black=teacher)
            finally:
                log.close()
    return samples


def _generate_chunk(jobs):
    """Engines persist across jobs and rounds until worker finalization."""
    return [sample for job in jobs for sample in _generate_game(job)]


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
        emit(args, 'dataset_reused', path=str(cache), positions=len(data['y']), config=data['config'])
        return data
    config = {name: getattr(args, name) for name in (
        'teacher', 'teacher_depth', 'teacher_ms', 'teacher_nodes', 'max_plies',
        'sample_every', 'eps', 'sf_path', 'sf_depth', 'sf_ms', 'sf_threads', 'sf_hash', 'sf_play_ms', 'log_detail')}
    config['seed'] = args.seed + (round_no - 1) * 1000
    config['round'] = round_no
    if getattr(args, '_run_dir', None):
        config['game_log_dir'] = str(args._run_dir / f'round-{round_no:06d}' / 'games')
    emit(args, 'generation_start', round=round_no, games=args.games, config=config)
    jobs = [(game, config, deadline) for game in range(args.games)]
    samples = []
    started = time.monotonic()
    limits = (f'depth {args.sf_depth}, {args.sf_ms:g}ms per label' if args.teacher == 'stockfish'
              else f'depth {args.teacher_depth}, {args.teacher_ms:g}ms/{args.teacher_nodes} nodes per move')
    print(f'Generating {args.games} games: {args.teacher} teacher, {limits}.', flush=True)
    workers = min(args.workers, args.games)
    if workers == 1:
        for job in jobs:
            samples.extend(_generate_game(job))
            if time.monotonic() >= deadline:
                break
    else:
        # Spawn is safe after Torch/MPS/CUDA initialization, unlike fork.
        with (contextlib.nullcontext(args._pool) if getattr(args, '_pool', None) is not None
              else ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'), initializer=_init_worker)) as pool:
            chunks = [[job] for job in jobs]  # dynamic scheduling keeps every worker busy
            futures = {pool.submit(_generate_chunk, chunk): len(chunk) for chunk in chunks}
            done = 0
            try:
                for future in as_completed(futures):
                    samples.extend(future.result())
                    done += futures[future]
                    emit(args, 'generation_progress', round=round_no, games_done=done, samples=len(samples), elapsed_seconds=time.monotonic() - started)
                    if done % max(1, args.games // 10) == 0 or done == args.games:
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
    emit(args, 'dataset_generated', round=round_no, positions=len(ys), elapsed_seconds=time.monotonic() - started)
    if cache:
        atomic_save(data, cache)
        emit(args, 'artifact_saved', kind='dataset', path=str(cache), positions=len(ys))
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
        net.training_round = int(state.get('metadata', {}).get('round', 0))
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
    train_cpu, train_targets = data['X'][~val_mask], data['y'][~val_mask]
    val_x, val_y = data['X'][val_mask].to(device), data['y'][val_mask].to(device)
    anchor = getattr(args, '_anchor', None)
    if anchor is None:
        path = Path(args.validation_set)
        if path.exists() and not args.reset_validation:
            anchor = torch.load(path, map_location='cpu', weights_only=True)
            if (anchor.get('format_version') != 1 or anchor['X'].ndim != 2
                    or anchor['X'].shape[1] != INPUT_SIZE or len(anchor['X']) % 2
                    or anchor['y'].shape != (len(anchor['X']), 1)):
                raise ValueError('Invalid fixed validation set; use --reset-validation.')
        else:
            anchor = {'format_version': 1, 'X': val_x.cpu(), 'y': val_y.cpu(),
                      'teacher': args.teacher, 'quality': args.quality}
            atomic_save(anchor, path)
            emit(args, 'artifact_saved', kind='fixed_validation', path=str(path), positions=len(anchor['y']))
        args._anchor = anchor
    replay = getattr(args, '_replay', None)
    if replay is not None:
        train_cpu = torch.cat([train_cpu, replay[0]])
        train_targets = torch.cat([train_targets, replay[1]])
    # Keep fixed validation positions out of the training/replay stream.
    forbidden = {row.numpy().tobytes() for row in anchor['X']}
    forbidden.update(row.numpy().tobytes() for row in val_x.cpu())
    keep, seen = [], set()
    for index in range(0, len(train_cpu), 2):
        left = train_cpu[index].numpy().tobytes()
        right = train_cpu[index + 1].numpy().tobytes()
        key = min(left, right)
        if left not in forbidden and right not in forbidden and key not in seen:
            keep.append(index)
            seen.add(key)  # newest labels precede replay labels for the same position
    keep = torch.tensor([row for index in keep for row in (index, index + 1)], dtype=torch.long)
    train_cpu, train_targets = train_cpu[keep], train_targets[keep]
    if not len(train_cpu):
        raise ValueError('No training positions remain after excluding validation; generate fresh data.')
    if args.replay_positions:
        # Preserve mirrored pairs. Sample uniformly so replay does not contain just the oldest games.
        pairs = torch.randperm(len(train_cpu) // 2)[:args.replay_positions // 2]
        rows = torch.stack([pairs * 2, pairs * 2 + 1], dim=1).flatten()
        args._replay = (train_cpu[rows].clone(), train_targets[rows].clone())
    train_x, train_y = train_cpu.to(device), train_targets.to(device)
    anchor_x, anchor_y = anchor['X'].to(device), anchor['y'].to(device)
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

    def validate(x=val_x, y=val_y):
        net.eval()
        total = 0.0
        with torch.inference_mode():
            for start in range(0, len(y), args.batch):
                with autocast():
                    indices = torch.arange(start, min(start + args.batch, len(y)), device=device)
                    both = net(torch.cat([x[indices], x[indices ^ 1]]))
                    left, right = both.chunk(2)
                    out = (left - right) * .5
                total = total + (out - y[start:start + args.batch]).square().sum()
        return (total / len(y)).item()

    emit(args, 'training_split', round=round_no, train_positions=len(train_y), validation_positions=len(val_y), anchor_positions=len(anchor_y), baseline_mse=baseline)
    best_loss, best_state, stale = validate(), _state(net), 0
    initial = best_loss
    best_anchor = validate(anchor_x, anchor_y)
    print(f"Fixed validation: {len(anchor_y)} positions, initial mse={best_anchor:.5f}.", flush=True)
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
                if getattr(args, '_log', None) and args.log_detail == 'full':
                    emit(args, 'minibatch', round=round_no, epoch=epoch, step=trained_steps, positions=len(batch), loss=loss.detach().item(), learning_rate=opt.param_groups[0]['lr'])
            sched.step()
            val_loss = validate()
            anchor_loss = validate(anchor_x, anchor_y)
            improved = val_loss < best_loss and anchor_loss <= best_anchor + 1e-6
            if improved:
                best_anchor = anchor_loss
                best_loss, best_state, stale = val_loss, _state(net), 0
                if best_loss < baseline:
                    atomic_save(checkpoint(net, val_mse=best_loss, baseline_mse=baseline,
                                           round=round_no, teacher=args.teacher,
                                           seed=args.seed, trained_steps=trained_steps,
                                           anchor_val_mse=best_anchor, teacher_config=data['config'],
                                           search_mode=args.search_mode), args.out)
                    emit(args, 'artifact_saved', kind='checkpoint', path=str(args.out), round=round_no, epoch=epoch, val_mse=best_loss, anchor_mse=best_anchor)
            else:
                stale += 1
            emit(args, 'epoch', round=round_no, epoch=epoch, val_mse=val_loss, anchor_mse=anchor_loss, improved=improved, saved=improved and best_loss < baseline, steps=trained_steps)
            print(f"Epoch {epoch:2d}/{args.epochs}: val_mse={val_loss:.5f}, "
                  f"residual rmse={RESIDUAL_CP * val_loss ** .5:.1f}cp, anchor_mse={anchor_loss:.5f}"
                  f"{'  best saved' if improved and best_loss < baseline else ''}", flush=True)
            if stale >= args.patience:
                emit(args, 'early_stop', round=round_no, epoch=epoch, reason='validation_patience')
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


def _train(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    started = time.monotonic()
    deadline = started + args.duration_seconds if args.duration_seconds else float('inf')
    if args.teacher == 'stockfish':
        args.sf_path = args.sf_path or shutil.which('stockfish')
        if not args.sf_path:
            raise ValueError('Stockfish not found. Use --teacher search or provide --sf-path.')
    emit(args, 'device_selected', device=str(device), cuda_available=torch.cuda.is_available(), mps_available=torch.backends.mps.is_available())
    net = EvalNet().to(device)
    if not args.fresh:
        resumed = load_weights_into(net, args.resume)
        emit(args, 'resume', path=args.resume, loaded=resumed)
    if args.compile:
        if device.type in ('cpu', 'cuda'):
            net.forward = torch.compile(net.forward)
            print('Compiling forward pass; initial compilation is included in the time budget.')
        else:
            print('Compile skipped on MPS; using eager execution.')
    print(f"device={device}, params={sum(p.numel() for p in net.parameters()):,}, "
          f"threads={args.threads}, output={args.out}", flush=True)
    print('Each improving validated epoch is saved atomically. Ctrl+C keeps the best checkpoint.', flush=True)
    replay_path = Path(args.out + f'.{args.teacher}.replay.pt')
    if args.replay_positions and replay_path.exists() and not args.fresh:
        replay = torch.load(replay_path, map_location='cpu', weights_only=True)
        if (replay.get('format_version') != 1 or replay['X'].ndim != 2
                or replay['X'].shape[1] != INPUT_SIZE or len(replay['X']) % 2
                or replay['y'].shape != (len(replay['X']), 1)):
            raise ValueError('Invalid replay data; remove the replay file or use --fresh.')
        args._replay = (replay['X'][-args.replay_positions:], replay['y'][-args.replay_positions:])
        print(f'Resumed {len(args._replay[0])} replay positions.', flush=True)
    workers = min(args.workers, args.games)
    if args.teacher == 'stockfish':
        print(f'Stockfish: {workers} workers × {args.sf_threads} threads, {args.sf_hash} MiB hash each; {args.sf_play_ms:g}ms self-play moves.', flush=True)
        if workers * args.sf_threads > (os.cpu_count() or 1):
            print('Stockfish worker × thread count exceeds CPU count; reduce --workers or --sf-threads for better throughput.', flush=True)
    args._pool = (ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'), initializer=_init_worker)
                  if workers > 1 else None)
    round_no = getattr(net, 'training_round', 0)
    completed = 0
    mode = f"{args.duration_seconds:g} seconds" if args.duration_seconds else 'until Ctrl+C' if args.rounds == 0 else f'{args.rounds} round(s)'
    print(f'Training for {mode}; fresh self-play each round unless --dataset is set.', flush=True)
    try:
        while (args.rounds == 0 or completed < args.rounds) and time.monotonic() < deadline:
            round_no += 1
            remaining = f", {max(0, deadline - time.monotonic()):.1f}s remaining" if args.duration_seconds else ''
            print(f'Round {round_no}{remaining}', flush=True)
            emit(args, 'round_start', round=round_no)
            loss = train_round(net, args, device, round_no, deadline)
            emit(args, 'round_end', round=round_no, val_mse=loss)
            completed += 1
            if args.replay_positions and getattr(args, '_replay', None) is not None:
                atomic_save({'format_version': 1, 'X': args._replay[0], 'y': args._replay[1]}, replay_path)
                emit(args, 'artifact_saved', kind='replay', path=str(replay_path), positions=len(args._replay[0]))
    except BudgetExpired:
        emit(args, 'budget_expired', phase='generation')
        print('Time budget reached during generation; existing weights untouched.')
    except KeyboardInterrupt:
        emit(args, 'interrupted')
        if args._pool is not None:
            args._pool.terminate_workers()
        print('\nInterrupted; last validated checkpoint preserved.')
    finally:
        if args._pool is not None:
            args._pool.shutdown(wait=True, cancel_futures=True)
        _close_stockfish()
    emit(args, 'training_end', completed_rounds=completed, elapsed_seconds=time.monotonic() - started)
    print(f"Done: {completed} completed round(s), {time.monotonic() - started:.1f}s.")


def train(args):
    if not args.log_dir:
        args._run_dir = None
        args._log = None
        return _train(args)
    args._run_dir = new_run_directory(args.log_dir, 'train')
    args._log = EventLog(args._run_dir / 'events.jsonl')
    try:
        with console_log(args._run_dir / 'console.log'):
            print(f'Logs: {args._run_dir.resolve()}', flush=True)
            emit(args, 'run_start', config={k: v for k, v in vars(args).items() if not k.startswith('_')}, torch_version=torch.__version__)
            try:
                return _train(args)
            except BaseException as exc:
                emit(args, 'run_error', message=str(exc), traceback=traceback.format_exc())
                traceback.print_exc()
                raise
            finally:
                emit(args, 'run_end')
    finally:
        args._log.close()
        args._log = None


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--games', type=int, default=80)
    p.add_argument('--epochs', type=int, default=16)
    p.add_argument('--rounds', type=int, default=None,
                   help='round limit; defaults to unlimited with a duration, otherwise 1; 0 = unlimited')
    duration = p.add_mutually_exclusive_group()
    duration.add_argument('--for', dest='duration_seconds', type=parse_duration,
                          help='train for a duration, e.g. 30m, 2h, 1d (automatically repeats rounds)')
    duration.add_argument('--minutes', type=float, default=0,
                          help='train for this many minutes (automatically repeats rounds)')
    duration.add_argument('--forever', action='store_true', help='keep training until Ctrl+C')
    p.add_argument('--eps', type=float, default=.12)
    p.add_argument('--max-plies', type=int, default=100)
    p.add_argument('--sample-every', type=int, default=4)
    p.add_argument('--teacher', choices=['search', 'classical', 'stockfish'], default='search')
    p.add_argument('--quality', choices=QUALITY_PRESETS, default='balanced',
                   help='teacher strength/time preset; individual teacher limits override it')
    p.add_argument('--teacher-depth', type=int, default=None)
    p.add_argument('--teacher-ms', type=float, default=None)
    p.add_argument('--teacher-nodes', type=int, default=None)
    p.add_argument('--sf-path', default='')
    p.add_argument('--sf-threads', type=int, default=1, help='CPU threads per Stockfish worker')
    p.add_argument('--sf-hash', type=int, default=64, help='Stockfish hash MiB per worker')
    p.add_argument('--sf-play-ms', type=float, default=5, help='time per unsampled Stockfish self-play move')
    p.add_argument('--log-dir', default='.training/logs', help='run logs and PGNs; empty string disables logging')
    p.add_argument('--log-detail', choices=['summary', 'full'], default='full', help='full includes every move, search result, label decision and minibatch')
    p.add_argument('--sf-depth', type=int, default=None)
    p.add_argument('--sf-ms', type=float, default=None, help='hard time bound for optional Stockfish labels')
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--batch', type=int, default=128)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='auto', choices=['auto', 'cpu', 'mps', 'cuda'])
    p.add_argument('--resume', default=None, help='weights to resume; defaults to --out; empty string starts new')
    p.add_argument('--fresh', action='store_true', help='ignore resume weights; output only changes if validation improves')
    p.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) // 2))
    p.add_argument('--threads', type=int, default=2, help='Torch CPU threads; small models do not need all cores')
    p.add_argument('--patience', type=int, default=4)
    p.add_argument('--reset-validation', action='store_true', help='replace fixed validation with new held-out games')
    p.add_argument('--replay-positions', type=int, default=8192, help='training positions retained between rounds; 0 disables')
    p.add_argument('--search-mode', choices=['root', 'compiled', 'projected'], default='root', help='how gameplay uses this checkpoint')
    p.add_argument('--validation-set', default=None, help='fixed held-out set; defaults to <out>.validation.pt')
    p.add_argument('--dataset', default='', help='cache/reuse encoded dataset at this path')
    p.add_argument('--rebuild-dataset', action='store_true')
    p.add_argument('--amp', action='store_true', help='mixed precision on CUDA')
    p.add_argument('--compile', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--out', default=str(WEIGHTS_PATH))
    args = p.parse_args(argv)
    for name, value in QUALITY_PRESETS[args.quality].items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    if args.duration_seconds is None:
        args.duration_seconds = args.minutes * 60
    if args.rounds is None:
        args.rounds = 0 if args.duration_seconds > 0 or args.forever else 1
    if args.forever and args.rounds != 0:
        p.error('--forever cannot have a finite --rounds limit')
    if args.resume is None:
        args.resume = args.out
    if args.validation_set is None:
        args.validation_set = args.out + f".{args.teacher}.validation.pt"
    for name in ('games', 'epochs', 'max_plies', 'sample_every', 'teacher_depth', 'teacher_nodes',
                 'batch', 'workers', 'threads', 'patience', 'sf_depth', 'sf_threads', 'sf_hash'):
        if getattr(args, name) <= 0:
            p.error(f'--{name.replace("_", "-")} must be positive')
    if args.replay_positions < 0 or args.replay_positions % 2:
        p.error('--replay-positions must be a nonnegative even number')
    if args.games < 2 or args.max_plies < 6:
        p.error('Use at least 2 games and 6 plies for a game-separated validation split.')
    if not 0 <= args.eps <= 1 or args.teacher_ms <= 0 or args.sf_ms <= 0 or args.sf_play_ms <= 0 or args.lr <= 0:
        p.error('eps must be in [0,1]; time limits and learning rate must be positive')
    if not 0 <= args.minutes < float('inf') or args.rounds < 0:
        p.error('minutes must be finite and nonnegative; rounds must be nonnegative')
    for name in ('teacher_ms', 'sf_ms', 'sf_play_ms', 'lr'):
        if not getattr(args, name) < float('inf'):
            p.error(f'--{name.replace("_", "-")} must be finite')
    # Never let a compact run accidentally replace the original CNN checkpoint.
    if Path(args.out).resolve() == Path(__file__).with_name('weights.pt').resolve():
        p.error('Use a separate compact output (default: fast_weights.pt); weights.pt is the legacy checkpoint.')
    return args


def main():
    train(parse_args())


if __name__ == '__main__':
    main()
