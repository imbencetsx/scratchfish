"""Entry point: wire the engine to the GUI."""

from engine import ChessEngine
from gui import ChessGUI


def main() -> None:
    engine = ChessEngine(bot_is_white=False)
    gui = ChessGUI(engine)
    gui.run()


if __name__ == "__main__":
    main()
