"""Durable, process-local JSONL traces and replayable chess games."""

import contextlib
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import threading
import uuid

import chess.pgn


def new_run_directory(root, kind):
    path = Path(root) / f'{kind}-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:8]}'
    path.mkdir(parents=True)
    return path


class EventLog:
    def __init__(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = path.open('a', encoding='utf-8', buffering=1)
        self.lock = threading.Lock()

    def event(self, event, **fields):
        record = {'time': datetime.now(timezone.utc).isoformat(), 'event': event, **fields}
        with self.lock:
            self.file.write(json.dumps(record, default=str, allow_nan=False) + '\n')

    def close(self):
        self.file.close()


def emit(args, event, **fields):
    log = getattr(args, '_log', None)
    if log is not None:
        log.event(event, **fields)


def save_game(board, path, **headers):
    game = chess.pgn.Game.from_board(board)
    # Game.from_board supplies the true result; capped/interrupted games stay '*'.
    game.headers.update({key: str(value) for key, value in headers.items()})
    path = Path(path)
    temporary = path.with_suffix('.pgn.tmp')
    temporary.write_text(str(game) + '\n\n', encoding='utf-8')
    temporary.replace(path)


class _Tee:
    def __init__(self, stream, file):
        self.stream, self.file = stream, file
    def write(self, text):
        self.file.write(text)
        return self.stream.write(text)
    def flush(self):
        self.file.flush()
        self.stream.flush()
    def __getattr__(self, name):
        return getattr(self.stream, name)


@contextlib.contextmanager
def console_log(path):
    with Path(path).open('a', encoding='utf-8', buffering=1) as file:
        with contextlib.redirect_stdout(_Tee(sys.stdout, file)), contextlib.redirect_stderr(_Tee(sys.stderr, file)):
            yield
