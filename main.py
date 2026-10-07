"""Entry point: wire the engine to the GUI."""

import argparse
from pathlib import Path

from neural import WEIGHTS_PATH
from engine import ChessEngine
from gui import ChessGUI


def main() -> None:
    parser = argparse.ArgumentParser(description="Play Scratchfish.")
    parser.add_argument('--log-dir', default='.training/logs', help='game trace and PGN directory; empty string disables')
    parser.add_argument('--weights', type=Path, default=WEIGHTS_PATH)
    parser.add_argument('--search-mode', choices=['root', 'compiled', 'projected'], default=None)
    parser.add_argument('--seconds', type=float, default=.35)
    args = parser.parse_args()
    if not 0 < args.seconds < float('inf'):
        parser.error('--seconds must be positive and finite')
    engine = ChessEngine(bot_is_white=False, time_limit=args.seconds,
                         weights=args.weights, search_mode=args.search_mode, log_dir=args.log_dir)
    gui = ChessGUI(engine)
    gui.run()


if __name__ == "__main__":
    main()
