# Scratchfish

A time-bounded chess bot with a Pygame board. Tactical search uses a fast
handcrafted evaluator; a small trained network guides root move ordering.
Gameplay and default training do not require Stockfish. For stronger training
targets, use the optional Stockfish teacher.

## Play

```sh
uv sync
uv run python main.py
# Optional: experiment with learned square values at deeper search nodes.
uv run python main.py --search-mode compiled
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
# Fast teacher for a quick experiment; no Stockfish required.
uv run python train.py --quality fast --dataset .training/positions.pt

# Reuse the dataset and fine-tune the current compact checkpoint.
uv run python train.py --dataset .training/positions.pt --epochs 24

# Train for any duration; fresh self-play continues across rounds.
uv run python train.py --for 30m
uv run python train.py --for 2h

# Keep training until you press Ctrl+C.
uv run python train.py --forever
```

`--for` accepts seconds (`30s`), minutes (`90m`), hours (`2h`), or days
(`1d`), including decimals (`1.5h`). `--minutes 60` also works. A duration
repeats rounds automatically; no extra `--rounds 0` flag is needed.
`--forever` runs until Ctrl+C. Without either option, the default remains one
round. An explicit `--rounds N` can cap a timed run. Early stopping ends only
the current round; the next round uses fresh games and training continues
until the requested duration. Longer training does not guarantee stronger play.

Re-run the command to continue from the saved model. It also continues the
self-play seed sequence from the saved round rather than replaying the same
initial games. With `--out my-model.pt`, it resumes that same file by default;
`--resume` overrides this. For long runs, omit `--dataset` to keep generating
fresh positions; caching is useful for short experiments on a fixed dataset.

Defaults: 80 games, up to 100 plies per game, half the logical CPU count in
workers (5 on this Mac), 16 epochs, and a 51,841-parameter model. The default
teacher preset is `balanced`. `--device auto` selects CUDA, then the Mac GPU
through Metal (`mps`), then CPU. An explicitly requested unavailable GPU fails
with a clear error. `--device cpu` remains available. The model and encoded
training/validation tensors stay on the selected device throughout optimization.
GPU training is supported, but this small model can still train faster on CPU;
GPU dispatch overhead differs from the large matrix workloads in LLM training.
Larger `--batch` sizes can reduce dispatch overhead; CUDA also supports `--amp`. Generation uses the bot's
own search at depth 3, capped at 1024 nodes or 20ms per move. Use
`--quality fast` for the old depth-2/256-node/8ms limits. Individual teacher
limits override the preset; stronger teachers take longer to generate data.

| Preset | Own-search depth / nodes / time | Stockfish depth / time per label |
| --- | --- | --- |
| `fast` | 2 / 256 / 8ms | 12 / 20ms |
| `balanced` | 3 / 1024 / 20ms | 14 / 60ms |
| `deep` | 5 / 8192 / 100ms | 18 / 150ms |

These are depth ceilings with time limits, not guaranteed achieved depths.
The gameplay budget stays at 0.35 seconds regardless of the teacher preset.

Only sampled, actually searched positions become labels. Forced mates,
positions in check (for Stockfish labels), best-move captures/checks, and
corrections outside the small model's useful range are skipped. Unfinished
games are not labeled draws. Color mirrors stay with their source game, and entire
games are held out for validation. Training matches the color-symmetric
inference function and saves an epoch only when it improves the current model
and beats the classical baseline on held-out labels. This is a label-quality
gate, not an Elo estimate. A fixed held-out set also has to avoid regression
before a checkpoint is saved; it persists in `<out>.<teacher>.validation.pt`.
Exact validation positions are excluded from the training stream. Up to
8192 training positions are replayed between rounds to reduce forgetting,
and replay data persists in `<out>.<teacher>.replay.pt`. `--replay-positions`
controls its size. Use `--reset-validation` to deliberately replace the fixed
set when changing a training experiment. Early stopping limits overfitting. The model adds
at most 200cp to static evaluation and guides root ordering in gameplay;
search determines tactics. Default checkpoints use root guidance. Experimental
`--search-mode compiled` fuses conservative learned adjustments into the normal
piece-square tables, so deeper nodes use training without running Torch or the
network per leaf. `--search-mode projected` uses a less conservative local
linear approximation. Neither is exact neural inference or guaranteed stronger;
both remain opt-in because the tested root-guided search scored better. A
training checkpoint records `--search-mode`; gameplay follows it unless you
override the mode. `--weights` selects another checkpoint when playing.

Checkpoints are atomic and saved after each improving validated epoch.
Ctrl+C preserves the last good checkpoint. `--fresh` ignores resume weights;
`--out` selects a separate output. Default output is `fast_weights.pt`, and
training refuses to overwrite the legacy `weights.pt`. Resuming restores
model weights; the optimizer starts fresh. `--dataset` reuses its stored
positions and teacher settings even if generation flags change; use
`--rebuild-dataset` to replace it. Without `--dataset`, every round generates
fresh games. Time budgets are checked during generation and minibatches;
worker startup and an in-flight operation may slightly exceed the budget.

For serious longer training, I recommend Stockfish: it supplies knowledge beyond
our handcrafted evaluator. The own-search teacher is useful for fast experiments
and has no external dependency, but it has a lower learning ceiling.

```sh
# Recommended stronger teacher and fresh positions for a two-hour run.
uv run python train.py --teacher stockfish --quality deep --for 2h --games 240

# Fast smoke test with the optional teacher.
uv run python train.py --teacher stockfish --quality fast --games 8
```

`--sf-ms` limits labeling time per sampled position. Sampled positions use the
Stockfish analysis PV; intermediate self-play moves use inexpensive 5ms Stockfish
searches, improving the quality of generated positions. Each worker keeps its engine alive across games and rounds. Games are scheduled
individually so a worker that finishes early can pick up another game. Worker
processes use one Torch CPU thread to avoid oversubscription. Engines are closed
at worker shutdown. Training takes longer with the deeper
teacher, while move-time performance is unchanged. Avoid `--dataset` during
long improvement runs so training keeps seeing fresh games. The generation
filtering is inspired by [Stockfish's configurable training-data filtering](https://github.com/official-stockfish/nnue-pytorch/blob/master/data_loader/config.py).

## GPU, Stockfish throughput, and logs

```sh
# Uses your Mac GPU automatically, with detailed logs enabled by default.
uv run python train.py --teacher stockfish --quality deep --for 2h

# Tune CPU resources: five concurrent games, one Stockfish thread per game.
uv run python train.py --teacher stockfish --workers 5 --sf-threads 1 --sf-hash 64 --for 30m

# Reduce log volume and per-minibatch GPU synchronization for longer runs.
uv run python train.py --teacher stockfish --log-detail summary --for 2h
```

Stockfish itself runs on CPU, including its NNUE evaluation; GPU acceleration
applies to our PyTorch model training, not Stockfish searches. See the
[Stockfish FAQ](https://official-stockfish.github.io/docs/stockfish-wiki/Stockfish-FAQ.html)
and [PyTorch Metal backend documentation](https://docs.pytorch.org/docs/stable/notes/mps.html).
Data generation is currently the largest part of a full training run. More
Stockfish threads can increase depth within the same time budget, but do not
necessarily produce more games per second. Keep `--workers × --sf-threads`
within your CPU count. `--sf-hash` is MiB **per worker** (64 by default).
`--sf-play-ms` sets the unsampled self-play move budget (5ms by default);
`--sf-ms` sets the sampled label budget. Reducing these limits trades teacher
quality for throughput.

A local two-round, 80-game generation check took 6.48s with the updated defaults
versus 9.24s with the previous worker/chunk settings, about 30% less elapsed time.
Logging was disabled for this comparison; time-bounded searches produced different
label counts, so this does not measure strength or promise the same speedup on
other hardware. Details are in `benchmarks/acceleration.json`. A separate full-log
smoke test completed two training rounds on Metal and saved validated checkpoints.

Every training invocation prints a unique directory under `.training/logs/`:

- `console.log`: all main-process console output and error tracebacks.
- `events.jsonl`: settings, selected GPU/CPU, resume status, round progress,
  dataset sizes, validation split, minibatch loss/rate, epoch metrics, early
  stopping, checkpoint decisions, and paths of saved datasets/weights/replay.
- `round-NNNNNN/games/game-NNNNNN.jsonl`: every move's FEN, SAN/UCI, teacher
  move, search statistics/PV, score, accepted target or skip reason, timing,
  and final result. Labels describe the original position; encoding also adds
  its color mirror with the opposite target.
- `round-NNNNNN/games/game-NNNNNN.pgn`: replayable games, including capped games
  marked `*`, with termination reasons. Cached dataset reuse generates no new
  games and is recorded explicitly.

Playing with `main.py` also creates a `play-...` directory with per-game JSONL
and PGN files, player/bot moves and bot search statistics. New Game, Switch and
Quit finalize the current log; live PGNs are updated after each move.

`--log-dir PATH` changes the destination; `--log-dir ''` disables logging.
`--log-detail summary` keeps game starts/ends, accepted labels, PGNs and epoch
metrics but omits per-move search traces and minibatch losses. Full logging is
line-flushed for inspection during training and includes errors. A force-stopped
worker may leave a partial JSONL without a finalized PGN; its recorded moves
still allow replay. Logs report search summaries, not each internal search node
or tensor element. Full logs consume disk space and minibatch loss logging
synchronizes the GPU; use summary mode when throughput matters most.

## Verification

```sh
uv run python -m unittest discover -s tests -v
uv run python benchmark.py --positions 24
uv run python benchmark.py --positions 24 --stockfish
```

The regression tests cover mate, quiet check evasions, free captures,
promotion, stalemate, exchange pruning, repetition, pawn endings, board
restoration after timeouts/exceptions, checkpoint round trips, and restarting
the GUI during a search. They also verify duration parsing, repeated rounds,
custom checkpoint resume selection, clean interruption, GPU selection, Stockfish
engine reuse, replayable training/GUI logs, error logging, analytic neural
derivatives, learned evaluation cost bounds, replay isolation, and rejecting
a checkpoint that worsens fixed validation despite improving fresh validation. The default 24 benchmark positions are frozen in
`tests/positions.json`; Stockfish is only a referee, not a runtime dependency.

On the development Mac, the original 10.7-million-parameter network took
about 4ms per leaf. Its search averaged 9.77 seconds despite a 0.15-second
budget and changed the input board on 12 of 24 searches. The replacement
respects its budget and leaves the board intact. An 80-game generation and
training run with the `fast` own-search teacher took 11.5 seconds; cached training took about 0.5 seconds including
startup. These are local measurements, not hardware-independent guarantees.
See `benchmarks/local.json` for the delivered model's move timing and referee
scores. A small opening/middlegame suite does not establish an Elo rating.

The new search uses principal-variation windows at the root and truncates
repetition history at irreversible moves. `benchmarks/improvements.json` records
a paired 24-position comparison using the same model snapshot: root-guided search
reached 4.50 plies on average versus 4.25 previously, with mean move loss of
21.2cp versus 26.6cp and mean time of 351.63ms versus 351.53ms. Stockfish scored
all candidate moves together in one MultiPV search per position; these numbers
are a small-suite check, not an Elo claim. Older benchmark loss figures used
separate forced-move searches and should not be compared directly with this
shared-referee method.

A local 120-game `deep` Stockfish run generated 2688 mirrored quiet-position
samples and completed in about 107 seconds; neural optimization itself took
0.12 seconds. Teacher data generation dominates training time.
