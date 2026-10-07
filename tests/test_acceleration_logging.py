import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import chess
import chess.pgn
import torch

from engine import ChessEngine
from train import _close_stockfish, _generate_game, _stockfish, parse_args, pick_device, train


class AccelerationTests(unittest.TestCase):
    def test_auto_prefers_cuda_then_metal_then_cpu(self):
        for cuda, metal, expected in [(True, True, 'cuda'), (False, True, 'mps'), (False, False, 'cpu')]:
            with patch('torch.cuda.is_available', return_value=cuda), \
                 patch('torch.backends.mps.is_available', return_value=metal):
                self.assertEqual(pick_device('auto').type, expected)
        with patch('torch.backends.mps.is_available', return_value=False):
            with self.assertRaisesRegex(ValueError, 'unavailable'):
                pick_device('mps')

    def test_stockfish_reused_until_settings_change_then_closed(self):
        _close_stockfish()
        first, second = MagicMock(), MagicMock()
        with patch('chess.engine.SimpleEngine.popen_uci', side_effect=[first, second]) as launch:
            try:
                self.assertIs(_stockfish('/fake', 2, 64), first)
                self.assertIs(_stockfish('/fake', 2, 64), first)
                self.assertEqual(launch.call_count, 1)
                self.assertIs(_stockfish('/fake', 1, 32), second)
                first.quit.assert_called_once()
                second.configure.assert_called_once_with({'Threads': 1, 'Hash': 32})
            finally:
                _close_stockfish()
            second.quit.assert_called_once()

    def test_invalid_resource_flags(self):
        for flag, value in [('--sf-threads', '0'), ('--sf-hash', '-1'), ('--sf-play-ms', 'nan'), ('--sf-ms', 'inf')]:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args([flag, value])

    @unittest.skipUnless(shutil.which('stockfish'), 'Stockfish not installed')
    def test_worker_engine_reuse_across_rounds_and_clean_shutdown(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            command = [sys.executable, 'train.py', '--teacher', 'stockfish', '--quality', 'fast',
                       '--games', '6', '--max-plies', '24', '--sample-every', '2',
                       '--epochs', '1', '--rounds', '2', '--workers', '2', '--device', 'cpu',
                       '--fresh', '--out', str(root / 'model.pt'), '--log-dir', str(root / 'logs')]
            result = subprocess.run(command, cwd=Path(__file__).resolve().parents[1],
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('Done: 2 completed round(s)', result.stdout)
            pids_by_round = []
            for round_dir in sorted((next((root / 'logs').iterdir())).glob('round-*')):
                pids = set()
                for file in round_dir.glob('games/*.jsonl'):
                    records = [json.loads(line) for line in file.read_text().splitlines()]
                    self.assertEqual(records[-1]['event'], 'game_end')
                    pids.update(record['pid'] for record in records if record['event'] == 'stockfish_ready')
                pids_by_round.append(pids)
            self.assertEqual(len(pids_by_round), 2)
            self.assertTrue(pids_by_round[0] & pids_by_round[1])
            self.assertLessEqual(len(set.union(*pids_by_round)), 2)
            for pid in set.union(*pids_by_round):
                with self.assertRaises(ProcessLookupError):
                    os.kill(pid, 0)


class LoggingTests(unittest.TestCase):
    def test_generated_trace_replays_pgn_and_preserves_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            config = dict(seed=0, teacher='classical', teacher_ms=1, teacher_nodes=64,
                          teacher_depth=1, max_plies=12, sample_every=1, eps=.1,
                          game_log_dir=directory, log_detail='full')
            samples = _generate_game((0, config, float('inf')))
            records = [json.loads(line) for line in (Path(directory) / 'game-000000.jsonl').read_text().splitlines()]
            moves = [r for r in records if r['event'] == 'move']
            board = chess.Board()
            for record in moves:
                self.assertEqual(record['fen'], board.fen())
                board.push_uci(record['move'])
            with (Path(directory) / 'game-000000.pgn').open() as stream:
                game = chess.pgn.read_game(stream)
            self.assertFalse(game.errors)
            self.assertEqual(game.end().board().fen(), board.fen())
            self.assertEqual(game.headers['Result'], '*')
            self.assertEqual(records[-1]['samples'], len(samples))
            self.assertEqual(sum(r['target'] is not None for r in moves), len(samples))

    def test_error_trace_retains_partial_game(self):
        with tempfile.TemporaryDirectory() as directory:
            config = dict(seed=0, teacher='search', teacher_ms=1, teacher_nodes=64,
                          teacher_depth=1, max_plies=12, sample_every=1, eps=.1, game_log_dir=directory)
            with patch('train.Searcher.best_move', side_effect=RuntimeError('broken search')):
                with self.assertRaises(RuntimeError):
                    _generate_game((0, config, float('inf')))
            records = [json.loads(line) for line in (Path(directory) / 'game-000000.jsonl').read_text().splitlines()]
            self.assertEqual(records[-1]['reason'], 'error')
            self.assertEqual(records[-1]['plies'], 4)
            self.assertIn('broken search', records[-2]['traceback'])
            self.assertTrue((Path(directory) / 'game-000000.pgn').exists())

    def test_live_moves_reset_switch_and_close(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = ChessEngine(use_nn=False, time_limit=.001, log_dir=directory)
            engine.try_move(chess.E2, chess.E4)
            engine.bot_move()
            engine.switch_side()
            engine.close_log()
            logs = sorted(Path(directory).glob('play-*/game-*.jsonl'))
            self.assertEqual(len(logs), 2)
            first = [json.loads(line) for line in logs[0].read_text().splitlines()]
            self.assertEqual([r['actor'] for r in first if r['event'] == 'move'], ['player', 'bot'])
            self.assertEqual(first[-1]['reason'], 'switch_side')
            with logs[0].with_suffix('.pgn').open() as stream:
                game = chess.pgn.read_game(stream)
            self.assertEqual(game.headers['Black'], 'Scratchfish')
            self.assertEqual(len(list(game.mainline_moves())), 2)

    def test_run_error_is_logged_and_console_saved(self):
        with tempfile.TemporaryDirectory() as directory:
            args = parse_args(['--log-dir', directory, '--device', 'cpu'])
            with patch('train._train', side_effect=RuntimeError('training failed')), \
                 contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(RuntimeError):
                    train(args)
            records = [json.loads(line) for line in (args._run_dir / 'events.jsonl').read_text().splitlines()]
            self.assertEqual(records[-2]['event'], 'run_error')
            self.assertIn('training failed', (args._run_dir / 'console.log').read_text())
            self.assertEqual(records[-1]['event'], 'run_end')


if __name__ == '__main__':
    unittest.main()
