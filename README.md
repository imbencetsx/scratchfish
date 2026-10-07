# Scratchfish

A time-bounded chess bot with a Pygame board. Tactical search uses a fast
handcrafted evaluator; a small trained network guides root move ordering.
Gameplay and training do not require Stockfish.

## Play

```sh
uv sync
uv run python main.py
```

The default move budget is **0.35 seconds**, with iterative deepening up to
8 plies. Search runs in a worker so the window keeps drawing and New Game,
Switch, and Quit remain responsive. `ChessEngine(depth=…, time_limit=…)`
controls the tradeoff. `last_search_info` reports depth, nodes, score and timing.

The included `fast_weights.pt` is a trained compact model. Without it, the bot
still searches using material, piece-square tables, pawn structure, bishop
pairs, rook files and phase-dependent king placement. The original large
`weights.pt` is preserved and is not loaded automatically.

## Train

```sh
# About 10–12 seconds on the development Mac, no Stockfish required.
uv run python train.py --dataset .training/positions.pt

# Reuse the dataset and fine-tune the current compact checkpoint.
uv run python train.py --dataset .training/positions.pt --epochs 24

# Fresh self-play each round, with a five-minute budget.
uv run python train.py --minutes 5 --rounds 0
```

Defaults: 80 games, up to 100 plies per game, 4 workers, 16 epochs, and a
51,841-parameter model. CPU is the default on Macs for this small model;
`--device mps` and `--device cuda` are available. Generation uses the bot's
own search at depth 2, capped at 256 nodes or 8ms per move. Increase
`--teacher-depth`, `--teacher-nodes` and `--teacher-ms` for stronger, slower labels.

Only sampled, actually searched positions become labels. Unfinished games
are not labeled draws. Color mirrors stay with their source game, and entire
games are held out for validation. Training matches the color-symmetric
inference function and saves an epoch only when it improves the current model
and beats the classical baseline on held-out labels. This is a label-quality
gate, not an Elo estimate. Early stopping limits overfitting. The model adds
at most 200cp to static evaluation and guides root ordering in gameplay;
search determines tactics.

Checkpoints are atomic and saved after each improving validated epoch.
Ctrl+C preserves the last good checkpoint. `--fresh` ignores resume weights;
`--out` selects a separate output. Default output is `fast_weights.pt`, and
training refuses to overwrite the legacy `weights.pt`. Resuming restores
model weights; the optimizer starts fresh. `--dataset` reuses its stored
positions and teacher settings even if generation flags change; use
`--rebuild-dataset` to replace it. Without `--dataset`, every round generates
fresh games. Time budgets are checked during generation and minibatches;
worker startup and an in-flight operation may slightly exceed the budget.

Stockfish is optional:

```sh
uv run python train.py --teacher stockfish --sf-depth 12 --sf-ms 20
```

`--sf-ms` limits labeling time per sampled position. Only the sampled
positions receive Stockfish labels; self-play between them uses our search.

## Verification

```sh
uv run python -m unittest discover -s tests -v
uv run python benchmark.py --positions 24
uv run python benchmark.py --positions 24 --stockfish
```

The 18 regression tests cover mate, quiet check evasions, free captures,
promotion, stalemate, exchange pruning, repetition, pawn endings, board
restoration after timeouts/exceptions, checkpoint round trips, and restarting
the GUI during a search. The default 24 benchmark positions are frozen in
`tests/positions.json`; Stockfish is only a referee, not a runtime dependency.

On the development Mac, the original 10.7-million-parameter network took
about 4ms per leaf. Its search averaged 9.77 seconds despite a 0.15-second
budget and changed the input board on 12 of 24 searches. The replacement
respects its budget and leaves the board intact. An 80-game generation and
training run took 11.5 seconds; cached training took about 0.5 seconds including
startup. These are local measurements, not hardware-independent guarantees.
See `benchmarks/local.json` for the delivered model's move timing and referee
scores. A small opening/middlegame suite does not establish an Elo rating.
