import contextlib
import io
import unittest
from unittest.mock import patch

from train import parse_args, train


class DurationTests(unittest.TestCase):
    def test_duration_units_and_unlimited_rounds(self):
        for text, seconds in [('30s', 30), ('90m', 5400), ('2h', 7200), ('1.5d', 129600)]:
            args = parse_args(['--for', text])
            self.assertEqual(args.duration_seconds, seconds)
            self.assertEqual(args.rounds, 0)
        self.assertEqual(parse_args(['--minutes', '60']).rounds, 0)
        self.assertEqual(parse_args(['--forever']).rounds, 0)
        self.assertEqual(parse_args([]).rounds, 1)

    def test_explicit_round_cap_and_custom_resume(self):
        args = parse_args(['--for', '2h', '--rounds', '3', '--out', '/tmp/other.pt'])
        self.assertEqual(args.rounds, 3)
        self.assertEqual(args.resume, args.out)
        self.assertEqual(parse_args(['--resume', '']).resume, '')

    def test_invalid_duration_and_conflicts(self):
        for argv in [['--for', '0s'], ['--for', '-2h'], ['--for', 'abc'],
                     ['--for', '1h', '--minutes', '2'], ['--forever', '--rounds', '2'],
                     ['--minutes', 'nan'], ['--minutes', 'inf']]:
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args(argv)

    def test_duration_actually_runs_multiple_rounds_and_stops(self):
        args = parse_args(['--for', '3s', '--fresh'])
        clock = [0.0]
        rounds = []
        def fake_round(net, args, device, number, deadline):
            rounds.append(number)
            clock[0] += 1
        with patch('train.time.monotonic', side_effect=lambda: clock[0]), \
             patch('train.train_round', side_effect=fake_round), \
             contextlib.redirect_stdout(io.StringIO()):
            train(args)
        self.assertEqual(rounds, [1, 2, 3])

    def test_forever_stops_cleanly_on_interrupt(self):
        args = parse_args(['--forever', '--fresh'])
        with patch('train.train_round', side_effect=KeyboardInterrupt), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            train(args)
        self.assertIn('last validated checkpoint preserved', output.getvalue())


if __name__ == '__main__':
    unittest.main()
