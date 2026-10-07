import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import chess
import numpy as np
import torch

from evaluation import evaluate_white_cp
from neural import EvalNet, FastEvaluator, board_to_tensor
from search import Searcher, _tt_key
from train import atomic_save, parse_args, train, train_round


class LearnedEvaluationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.net = EvalNet()
        with torch.no_grad():
            self.net.net[4].weight.normal_(0, .1)
        self.evaluator = FastEvaluator(self.net)

    def test_analytic_derivative_matches_autograd(self):
        board = chess.Board()
        x = board_to_tensor(board).requires_grad_()
        output = self.net(x)
        output.backward()
        value, gradient = self.evaluator._gradient(board)
        self.assertAlmostEqual(value, output.item(), places=6)
        np.testing.assert_allclose(gradient, x.grad.numpy(), atol=1e-6)

    def test_projected_root_and_mirror_match_full_network(self):
        board = chess.Board()
        for san in ('e4', 'd5', 'exd5'):
            board.push_san(san)
        projected = self.evaluator.for_search(board, mode='projected')
        self.assertLessEqual(abs(projected(board) - self.evaluator(board)), 1)
        mirrored = self.evaluator.for_search(board.mirror(), mode='projected')
        self.assertLessEqual(abs(projected(board) + mirrored(board.mirror())), 1)

    def test_compiled_leaf_evaluation_never_calls_network(self):
        board = chess.Board()
        compiled = self.evaluator.for_search(board)
        with patch.object(self.evaluator, '_gradient', side_effect=AssertionError('neural leaf call')), \
             patch.object(self.evaluator, 'residual', side_effect=AssertionError('neural leaf call')):
            move, info = Searcher(compiled, .02).best_move(board, 6)
        self.assertIn(move, board.legal_moves)
        self.assertGreater(info['nodes'], 20)
        self.assertEqual(board.fen(), chess.STARTING_FEN)

    def test_corrections_preserve_material_and_remain_small(self):
        root = chess.Board()
        for mode in ('compiled', 'projected'):
            ev = self.evaluator.for_search(root, mode)
            for fen in (chess.STARTING_FEN, '6k1/5ppp/8/8/8/8/5PPP/3Q2K1 w - - 0 1'):
                board = chess.Board(fen)
                self.assertLessEqual(abs(ev(board) - evaluate_white_cp(board)), 200)
            self.assertGreater(ev(chess.Board(fen)), 650)

    def test_repetition_history_stops_at_pawn_move(self):
        board = chess.Board()
        for san in ('Nf3', 'Nf6', 'Ng1', 'Ng8', 'e4'):
            board.push_san(san)
        search = Searcher(evaluate_white_cp, 0)
        search.best_move(board)
        self.assertEqual(search._positions, {_tt_key(board): 1})


class TrainingQualityTests(unittest.TestCase):
    def test_quality_presets_allow_explicit_overrides(self):
        fast = parse_args(['--quality', 'fast'])
        deep = parse_args(['--quality', 'deep', '--sf-ms', '200'])
        self.assertEqual(fast.teacher_nodes, 256)
        self.assertEqual(deep.teacher_depth, 5)
        self.assertEqual(deep.sf_depth, 18)
        self.assertEqual(deep.sf_ms, 200)
        self.assertGreater(deep.teacher_nodes, fast.teacher_nodes)

    def test_fixed_validation_is_reused_and_kept_out_of_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            args = parse_args(['--games', '4', '--epochs', '2', '--rounds', '2',
                               '--workers', '1', '--fresh', '--out', str(Path(directory) / 'net.pt')])
            boards = []
            board = chess.Board()
            for san in ('e4', 'e5', 'Nf3', 'Nc6', 'Bc4', 'Nf6', 'd3', 'Be7'):
                board.push_san(san)
                boards.extend([board.copy(), board.mirror()])
            x = torch.stack([board_to_tensor(b) for b in boards])
            data = {'format_version': 1, 'X': x, 'y': torch.zeros(len(x), 1),
                    'groups': torch.arange(len(x)) // 4, 'config': {}}
            with patch('train.build_dataset', return_value=data), contextlib.redirect_stdout(io.StringIO()):
                train(args)
            saved = torch.load(args.validation_set, weights_only=True)
            self.assertTrue(torch.equal(saved['X'], args._anchor['X']))
            forbidden = {row.numpy().tobytes() for row in saved['X']}
            self.assertTrue(all(row.numpy().tobytes() not in forbidden for row in args._replay[0]))
            self.assertEqual(len(args._replay[0]) % 2, 0)
            self.assertLessEqual(len(args._replay[0]), len(x))

    def test_improved_fresh_validation_cannot_overwrite_worsened_anchor(self):
        class SideOnlyNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.value = torch.nn.Parameter(torch.tensor(0.0))
            def forward(self, x):
                return x[:, 768:769] * self.value
        with tempfile.TemporaryDirectory() as directory:
            args = parse_args(['--games', '4', '--epochs', '3', '--workers', '1',
                               '--out', str(Path(directory) / 'net.pt')])
            path = Path(args.out)
            path.write_bytes(b'original validated checkpoint')
            x = torch.zeros(8, 776)
            x[::2, 768] = 1
            for group in range(4):
                x[group * 2:group * 2 + 2, 0] = group + 1
            y = torch.tensor([.2, -.2] * 4).unsqueeze(1)
            data = {'format_version': 1, 'X': x, 'y': y,
                    'groups': torch.arange(8) // 2, 'config': {}}
            anchor_x = x[:2].clone()
            anchor_x[:, 0] = 99  # distinct positions, same side-only feature
            atomic_save({'format_version': 1, 'X': anchor_x, 'y': -y[:2]}, args.validation_set)
            net = SideOnlyNet()
            with patch('train.build_dataset', return_value=data), contextlib.redirect_stdout(io.StringIO()):
                train_round(net, args, torch.device('cpu'), 1, float('inf'))
            self.assertEqual(net.value.item(), 0)
            self.assertEqual(path.read_bytes(), b'original validated checkpoint')


if __name__ == '__main__':
    unittest.main()
