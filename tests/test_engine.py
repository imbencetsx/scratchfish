import os
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')

import tempfile
import threading
import time
import unittest
from collections import Counter
from pathlib import Path

import chess
import numpy as np
import torch

from engine import ChessEngine
from evaluation import evaluate_white_cp
from neural import EvalNet, FastEvaluator, RESIDUAL_CP, checkpoint, evaluate_white_nn, try_load_net
from search import MATE, Searcher, _pack_mate, _unpack_mate, _tt_key
from train import _generate_game, atomic_save


class SearchTests(unittest.TestCase):
    def test_timeout_restores_full_board_and_returns_legal_move(self):
        board = chess.Board()
        for san in ('e4', 'e5', 'Nf3', 'Nc6'):
            board.push_san(san)
        before, stack = board.fen(), board.move_stack.copy()
        move, info = Searcher(evaluate_white_cp, .01).best_move(board, 12)
        self.assertIn(move, board.legal_moves)
        self.assertEqual(board.fen(), before)
        self.assertEqual(board.move_stack, stack)
        self.assertLess(info['elapsed'], .15)

    def test_exception_from_evaluator_also_restores_board(self):
        board = chess.Board()
        def broken(_):
            raise RuntimeError('evaluation failed')
        with self.assertRaises(RuntimeError):
            Searcher(broken, 1).best_move(board, 2)
        self.assertEqual(board.fen(), chess.STARTING_FEN)
        self.assertEqual(len(board.move_stack), 0)

    def test_zero_time_and_node_limit_have_legal_fallback(self):
        for search in (Searcher(evaluate_white_cp, 0), Searcher(evaluate_white_cp, 1, node_limit=32)):
            board = chess.Board()
            move, info = search.best_move(board, 8)
            self.assertIn(move, board.legal_moves)
            self.assertEqual(board.fen(), chess.STARTING_FEN)
            self.assertLessEqual(info['nodes'], 32)

    def test_check_quiescence_searches_quiet_evasions(self):
        board = chess.Board('4r1k1/8/8/8/8/8/8/4K3 w - - 0 1')
        self.assertTrue(board.is_check())
        self.assertFalse(any(board.is_capture(m) for m in board.legal_moves))
        # A bogus +9000 static eval in check must not allow stand-pat beta cutoff.
        search = Searcher(lambda b: 9000 if b.is_check() else -100, 1)
        search._deadline = time.monotonic() + 1
        search._positions = Counter([_tt_key(board)])
        score = search._quiesce(board, -MATE, 500, 0, 0)
        self.assertEqual(score, -100)
        self.assertEqual(board.fen(), '4r1k1/8/8/8/8/8/8/4K3 w - - 0 1')

    def test_mate_in_one_both_colors(self):
        positions = [
            'rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2',
            '6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1',
        ]
        for fen in positions:
            for board in (chess.Board(fen), chess.Board(fen).mirror()):
                self.assertTrue(board.is_valid())
                move, info = Searcher(evaluate_white_cp, .1).best_move(board, 4)
                board.push(move)
                self.assertTrue(board.is_checkmate(), (fen, move, info))
                self.assertGreater(info['score_cp'], MATE - 10)

    def test_free_queen_capture(self):
        board = chess.Board('6k1/8/8/8/3q4/8/8/3R2K1 w - - 0 1')
        move, _ = Searcher(evaluate_white_cp, .05).best_move(board, 5)
        self.assertEqual(move.uci(), 'd1d4')

    def test_promotion_and_stalemate(self):
        board = chess.Board('8/P7/7k/8/8/8/8/6K1 w - - 0 1')
        move, _ = Searcher(evaluate_white_cp, .05).best_move(board, 4)
        self.assertEqual(move.promotion, chess.QUEEN)
        board = chess.Board('7k/5Q2/6K1/8/8/8/8/8 b - - 0 1')
        move, info = Searcher(evaluate_white_cp, .05).best_move(board, 4)
        self.assertIsNone(move)
        self.assertEqual(info['score_cp'], 0)

    def test_exchange_value_for_free_defended_and_pinned_captures(self):
        board = chess.Board('6k1/8/8/8/3q4/8/8/3R2K1 w - - 0 1')
        self.assertEqual(Searcher._see(board, chess.Move.from_uci('d1d4')), 900)
        board = chess.Board('6k1/8/8/3p4/4p3/8/8/4Q1K1 w - - 0 1')
        self.assertLess(Searcher._see(board, chess.Move.from_uci('e1e4')), 0)
        board = chess.Board('4k3/8/8/4np2/6Q1/8/8/4R1K1 w - - 0 1')
        self.assertEqual(Searcher._see(board, chess.Move.from_uci('g4f5')), 100)

    def test_root_neural_ordering_does_not_run_at_every_leaf(self):
        calls = []
        def prior(board):
            calls.append(board.fen())
            return evaluate_white_cp(board)
        board = chess.Board()
        move, info = Searcher(evaluate_white_cp, .05, root_eval_white_cp=prior).best_move(board, 6)
        self.assertEqual(len(calls), 20)
        self.assertGreater(info['nodes'], len(calls))
        self.assertIn(move, board.legal_moves)
        self.assertEqual(board.fen(), chess.STARTING_FEN)

    def test_pawn_only_zugzwang_does_not_use_null_move(self):
        board = chess.Board('8/8/8/8/3k4/3p4/3K4/8 w - - 0 1')
        move, info = Searcher(evaluate_white_cp, .05).best_move(board, 6)
        self.assertIn(move, board.legal_moves)
        self.assertLess(info['score_cp'], 0)

    def test_mate_tt_scores_adjust_for_ply(self):
        for score in (MATE - 5, -MATE + 5, 120):
            self.assertEqual(_unpack_mate(_pack_mate(score, 3), 3), score)
        self.assertEqual(_unpack_mate(_pack_mate(MATE - 5, 3), 7), MATE - 9)

    def test_repetition_uses_game_history(self):
        board = chess.Board()
        for san in ('Nf3', 'Nf6', 'Ng1', 'Ng8', 'Nf3', 'Nf6', 'Ng1', 'Ng8'):
            board.push_san(san)
        search = Searcher(evaluate_white_cp, 0)
        search.best_move(board, 2)
        self.assertTrue(search._draw(board))
        self.assertEqual(search._positions[_tt_key(board)], 3)


class EvaluationTests(unittest.TestCase):
    def test_color_symmetry_and_material_floor(self):
        net = EvalNet()
        with torch.no_grad():
            for param in net.parameters():
                param.uniform_(-.1, .1)
        ev = FastEvaluator(net)
        for fen in (chess.STARTING_FEN, '6k1/5ppp/8/8/8/8/5PPP/3Q2K1 w - - 0 1'):
            board = chess.Board(fen)
            base = evaluate_white_cp(board)
            self.assertEqual(base, -evaluate_white_cp(board.mirror()))
            self.assertEqual(ev(board), -ev(board.mirror()))
            self.assertLessEqual(abs(ev(board) - base), RESIDUAL_CP)
        self.assertGreater(ev(chess.Board(fen)), 650)

    def test_sparse_inference_matches_torch(self):
        torch.manual_seed(42)
        net = EvalNet()
        with torch.no_grad():
            net.net[4].weight.normal_(0, .1)
        board = chess.Board()
        for san in ('e4', 'd5', 'exd5'):
            board.push_san(san)
        self.assertLessEqual(abs(FastEvaluator(net)(board) - evaluate_white_nn(net, board)), 1)

    def test_atomic_checkpoint_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fast.pt'
            atomic_save(checkpoint(EvalNet(), val_mse=.1), path)
            ev = try_load_net(path)
            self.assertIsInstance(ev, FastEvaluator)
            self.assertEqual(ev(chess.Board()), evaluate_white_cp(chess.Board()))
            self.assertFalse(path.with_name('fast.pt.tmp').exists())

    def test_generation_caps_do_not_create_draw_labels(self):
        config = dict(seed=0, teacher='search', teacher_ms=20, teacher_nodes=1024,
                      teacher_depth=2, max_plies=12, sample_every=1, eps=.1)
        samples = _generate_game((0, config, float('inf')))
        self.assertGreater(len(samples), 0)
        self.assertTrue(any(target != 0 for _, target, _ in samples))
        self.assertTrue(all(-1 <= target <= 1 for _, target, _ in samples))


class GuiTests(unittest.TestCase):
    def test_background_search_and_new_game_reject_stale_move(self):
        from gui import ChessGUI
        import pygame
        engine = ChessEngine(bot_is_white=True, time_limit=.05, use_nn=False)
        gui = ChessGUI(engine)
        try:
            gui.update_bot()
            self.assertIsNotNone(gui._pending)
            self.assertEqual(len(engine.board.move_stack), 0)
            gui.handle_click(gui.new_game_rect.center)
            new_board = engine.board
            gui._pending.result(timeout=1)
            gui.update_bot()
            self.assertIs(engine.board, new_board)
            self.assertEqual(len(engine.board.move_stack), 0)
            self.assertIsNotNone(gui._pending)
        finally:
            gui._stop_search.set()
            gui._executor.shutdown(wait=True)
            pygame.quit()

    def test_engine_bot_move_applies_one_legal_move(self):
        engine = ChessEngine(time_limit=.01, use_nn=False)
        self.assertTrue(engine.bot_move())
        self.assertEqual(len(engine.board.move_stack), 1)
        self.assertGreater(engine.last_search_info['nodes'], 0)


if __name__ == '__main__':
    unittest.main()
