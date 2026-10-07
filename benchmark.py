"""Reproducible timing and optional Stockfish move-quality checks (no training).

python benchmark.py --stockfish --positions 24
--baseline-dir can point at a saved copy of the previous Python modules.
"""
import argparse
import hashlib
import importlib.util
import json
import random
import shutil
import statistics
import time
from pathlib import Path

import chess
import chess.engine
import torch

from evaluation import evaluate_white_cp
from neural import EvalNet, FastEvaluator, WEIGHTS_PATH, try_load_net
from search import Searcher


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def positions(count, seed):
    # Diverse reproducible positions from fixed opening lines, with randomized continuation.
    openings = [
        'e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7',
        'd4 d5 c4 e6 Nc3 Nf6 Bg5 Be7 e3 O-O',
        'e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6',
        'd4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Nf3 O-O',
        'e4 e6 d4 d5 Nc3 Bb4 e5 c5 a3 Bxc3+',
        'e4 c6 d4 d5 Nc3 dxe4 Nxe4 Bf5 Ng3 Bg6',
    ]
    fixture = Path(__file__).with_name('tests') / 'positions.json'
    if seed == 42 and count <= 24 and fixture.exists():
        return [chess.Board(fen) for fen in json.loads(fixture.read_text())[:count]]
    rng = random.Random(seed)
    result = []
    for index in range(count):
        board = chess.Board()
        for san in openings[index % len(openings)].split():
            board.push_san(san)
        for _ in range((index // len(openings)) * 4):
            if board.is_game_over():
                break
            if rng.random() < .2:
                move = rng.choice(list(board.legal_moves))
            else:
                move, _ = Searcher(evaluate_white_cp, 1, node_limit=512).best_move(board, 2)
            board.push(move)
        result.append(board)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--positions', type=int, default=12)
    parser.add_argument('--seconds', type=float, default=.35)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--stockfish', action='store_true')
    parser.add_argument('--sf-path', default='')
    parser.add_argument('--baseline-dir', default='')
    parser.add_argument('--out', default='')
    parser.add_argument('--checkpoint', type=Path, default=WEIGHTS_PATH)
    parser.add_argument('--search-mode', choices=['root', 'compiled', 'projected'], default=None)
    parser.add_argument('--compare-search', type=Path, help='compare a saved previous search.py using the same model')
    parser.add_argument('--projected', action='store_true', help='also test learned leaf evaluation')
    args = parser.parse_args()
    if args.positions < 1 or args.seconds <= 0:
        parser.error('positions and seconds must be positive')
    torch.set_num_threads(1)
    boards = positions(args.positions, args.seed)
    net = try_load_net(args.checkpoint)
    search_mode = args.search_mode or getattr(net, 'metadata', {}).get('search_mode', 'root')
    class GameplaySearch:
        def __init__(self, evaluator, seconds):
            self.evaluator, self.seconds = evaluator, seconds
        def best_move(self, board, depth):
            leaf = (self.evaluator.for_search(board, search_mode)
                    if search_mode != 'root' else evaluate_white_cp)
            return Searcher(leaf, self.seconds, root_eval_white_cp=self.evaluator).best_move(board, depth)
    engines = {'classical': (Searcher, evaluate_white_cp, 8),
               'hybrid': (GameplaySearch, net or FastEvaluator(EvalNet()), 8)}
    if args.projected and net:
        class ProjectedSearch:
            def __init__(self, evaluator, seconds):
                self.evaluator, self.seconds = evaluator, seconds
            def best_move(self, board, depth):
                return Searcher(self.evaluator.for_search(board, mode='projected'), self.seconds,
                                root_eval_white_cp=self.evaluator).best_move(board, depth)
        engines['projected'] = (ProjectedSearch, net, 8)
        class CompiledSearch(ProjectedSearch):
            def best_move(self, board, depth):
                return Searcher(self.evaluator.for_search(board), self.seconds,
                                root_eval_white_cp=self.evaluator).best_move(board, depth)
        engines['compiled'] = (CompiledSearch, net, 8)
    if args.compare_search:
        previous = load_module('previous_search', args.compare_search)
        engines['previous_hybrid'] = (lambda evaluator, seconds: previous.Searcher(
            evaluate_white_cp, seconds, root_eval_white_cp=evaluator), net or FastEvaluator(EvalNet()), 8)
    if args.baseline_dir:
        directory = Path(args.baseline_dir)
        old_search = load_module('baseline_search', directory / 'search.py')
        old_nn = load_module('baseline_neural', directory / 'neural.py')
        old_eval = load_module('baseline_evaluation', directory / 'evaluation.py')
        old_net = old_nn.try_load_net(Path(__file__).with_name('weights.pt'))
        engines['old_classical'] = (old_search.Searcher, old_eval.evaluate_white_cp, 3)
        if old_net is not None:
            engines['old_neural'] = (old_search.Searcher, lambda b: old_nn.evaluate_white_nn(old_net, b), 3)
    sf_path = args.sf_path or shutil.which('stockfish')
    sf = chess.engine.SimpleEngine.popen_uci(sf_path) if args.stockfish and sf_path else None
    if args.stockfish and sf is None:
        parser.error('Stockfish unavailable; provide --sf-path or omit --stockfish')
    report = {'positions': args.positions, 'seconds': args.seconds, 'seed': args.seed,
              'checkpoint': str(args.checkpoint) if net else None, 'search_mode': search_mode,
              'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() if net else None,
              'fens': [board.fen() for board in boards], 'engines': {}}
    try:
        if sf:
            sf.configure({'Threads': 1, 'Hash': 64})
        chosen_moves = {}
        for name, (search_class, evaluator, depth) in engines.items():
            elapsed, depths, corruptions, moves = [], [], 0, []
            for index, original in enumerate(boards):
                board = original.copy()
                before, stack = board.fen(), board.move_stack.copy()
                started = time.perf_counter()
                move, info = search_class(evaluator, args.seconds).best_move(board, depth)
                elapsed.append(time.perf_counter() - started)
                depths.append(info['depth'])
                corruptions += int(board.fen() != before or board.move_stack != stack)
                moves.append(move)
                if (index + 1) % 6 == 0:
                    print(f'{name}: {index + 1}/{len(boards)} positions', flush=True)
            result = {'mean_move_ms': round(statistics.mean(elapsed) * 1000, 2),
                      'max_move_ms': round(max(elapsed) * 1000, 2),
                      'mean_depth': round(statistics.mean(depths), 2),
                      'board_corruptions': corruptions}
            chosen_moves[name] = moves
            report['engines'][name] = result
            print(name, json.dumps(result), flush=True)
        if sf:
            losses = {name: [] for name in engines}
            referee_depths = []
            for index, board in enumerate(boards):
                best = sf.play(board, chess.engine.Limit(depth=16, time=.15)).move
                candidates = {best}
                candidates.update(moves[index] for moves in chosen_moves.values()
                                  if moves[index] is not None and moves[index] in board.legal_moves)
                # Score all candidates together at the same MultiPV horizon, avoiding
                # different hash histories and depth budgets for each forced move.
                infos = sf.analyse(board, chess.engine.Limit(depth=16, time=.3),
                                   root_moves=sorted(candidates, key=lambda move: move.uci()),
                                   multipv=len(candidates))
                scores = {info['pv'][0]: info['score'].pov(board.turn).score(mate_score=100_000)
                          for info in infos if info.get('pv')}
                reference = max(scores.values())
                referee_depths.append(min(info.get('depth', 0) for info in infos))
                for name, moves in chosen_moves.items():
                    losses[name].append(max(0, reference - scores[moves[index]])
                                        if moves[index] in scores else 100_000)
            for name, values in losses.items():
                report['engines'][name].update(mean_loss_cp=round(statistics.mean(values), 1),
                                              median_loss_cp=round(statistics.median(values), 1),
                                              blunders_200cp=sum(value >= 200 for value in values))
            report['referee'] = {'method': 'shared-candidate-multipv',
                                 'min_depth': min(referee_depths),
                                 'mean_depth': round(statistics.mean(referee_depths), 1)}
    finally:
        if sf:
            sf.quit()
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
