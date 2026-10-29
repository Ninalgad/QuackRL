# scrabble-rl: Quackle from Python

A small bridge that lets Python drive the Quackle Scrabble engine: generate
every legal move, play moves, read the board and racks as numpy arrays, build
afterstates for candidate moves, and use Quackle's **Speedy Player** as a
baseline opponent.

You don't need to know C++ to use it. You write Python; one compiled file
(`qenv`) does the engine work.

```
scrabble-rl/
├── cpp/qenv.cpp            the bridge (the only C++ file)
├── CMakeLists.txt          build recipe: Quackle's engine + the bridge + tools
├── env.py                  Python game environment (rules, rewards, observations)
├── example.py              random agent and Speedy vs Speedy matches
├── example_afterstates.py  stage-2 inputs for a batch of candidate moves
├── test_qenv.py            correctness checks and a speed test
└── tools/
    ├── make_gaddag.cpp     builds a .gaddag from a .dawg, no Qt needed
    └── dump_moves.py       checks GADDAG and DAWG move generation agree
```

## How the bridge works

Python can import modules written in C++ if they are compiled into a shared
library (`qenv.cpython-3xx-....so` on Linux/macOS, `.pyd` on Windows).
**pybind11** is a header-only library that writes the glue: in `qenv.cpp`, the
block at the bottom (`PYBIND11_MODULE(qenv, m) { ... }`) lists which C++
functions Python can call and under which names. Everything above it is
ordinary C++ that calls Quackle and copies results into numpy arrays.

```
your Python  ──►  qenv (compiled)  ──►  libquackle (Quackle's engine)
  env.py           cpp/qenv.cpp           board, move generator, evaluator
```

**CMake** is the build tool. It compiles Quackle's engine (`libquackle`, which
does not need Qt) and the bridge, then links them into one module.

## Build (Linux or macOS)

Prerequisites: a C++ compiler (`g++` or `clang`; on macOS run
`xcode-select --install`), CMake 3.16+, Python 3.9+.

```bash
# 1. get the code side by side
git clone https://github.com/quackle/quackle.git
#    (put this scrabble-rl folder next to it)

# 2. Python dependencies
pip install pybind11 numpy

# 3. build (takes ~1 minute the first time)
cd scrabble-rl
cmake -S . -B build
cmake --build build -j

# 4. recommended: build a GADDAG for ~1.6x faster move generation (see below)
bin/make_gaddag ../quackle/data/lexica/nwl23.dawg ../quackle/data/lexica/nwl23.gaddag

# 5. check it works
python test_qenv.py ../quackle/data
python example.py ../quackle/data
```

The build places `qenv.*.so` in this folder, so `import qenv` works from here,
and `make_gaddag` in `bin/`. If your Quackle checkout is elsewhere:
`cmake -S . -B build -DQUACKLE_DIR=/path/to/quackle`.

On Windows, the easiest route is WSL (Ubuntu) and the Linux steps above.

**After editing `cpp/qenv.cpp`,** rerun only `cmake --build build -j`.
Changes to `env.py` need no rebuild.

## Using it

### The environment (`env.py`)

```python
from env import ScrabbleEnv

env = ScrabbleEnv("../quackle/data")       # loads the lexicon once
obs = env.reset(seed=0)
while not env.done:
    i = env.speedy_action(obs)             # or your network's choice
    obs, reward, done, info = env.step(i)
env.render()                               # board, scores, rack, bag
```

`reward` is from the point of view of the player who just moved: +1 win,
−1 loss, 0 draw, and 0 on every non-final move. `info["scores"]` has the
scores (final scores, with rack penalties, once the game is over).

Other methods:

- `env.render(top=5)` prints the board, scores, rack, bag size and the top
  candidate moves by equity. On the board, `=` is triple word, `-` double word,
  `"` triple letter, `'` double letter; a lowercase letter is a blank.
- `env.to_play()` returns the player to move (0 or 1).
- `env.speedy_action(obs)` returns the index of Speedy's move.
- `env.afterstates(obs, indices)` returns stage-2 inputs (below).
- `env.game.move_strings()` gives human-readable moves (e.g. `8D ONESIES`)
  aligned with `obs["moves"]`. With pass restricted, it has one extra
  entry at the end (`-`, pass) that you can ignore.

### Observations

Each observation is a dict, seen from the player to move:

| key | shape / type | meaning |
|---|---|---|
| `to_move` | int | 0 or 1 |
| `board_letters` | (15,15) uint8 | 0 empty, 1–26 = A–Z |
| `board_blanks` | (15,15) bool | square holds a blank |
| `board_owner` | (15,15) int8 | 0 empty, 1 placed by the player to move, 2 by the opponent |
| `board_age` | (15,15) int16 | moves since the tile was placed (0 = opponent's last move, 1 = my last move), −1 empty |
| `premium` | (15,15) int8 | uncovered premium squares: 0 none, 1 DL, 2 TL, 3 DW, 4 TW |
| `cross_h`, `cross_v` | (15,15) uint32 | cross-check masks (below) |
| `rack` | (27,) uint8 | counts; index 0 = blank, 1–26 = A–Z |
| `unseen` | (27,) uint8 | bag + opponent's rack, from the mover's view |
| `bag_size`, `score_me`, `score_opp`, `scoreless_turns` | int | |
| `moves` | dict of arrays | every candidate move, one row each (below) |

**Cross-checks:** bit k set means letter A+k may go on that square. `cross_h`
is the constraint from the word formed *across* through the square, so it
limits vertical plays; `cross_v` comes from the word formed *down* and limits
horizontal plays. Empty squares with no neighbours in that direction allow
all 26 letters; filled squares are 0. `unpack_cross(mask)` (from `env`) turns
a mask array into booleans with a trailing axis of 26.

**Ownership and age** are tracked by `env.py`, because Quackle's board stores
only letters. Tiles that a later word plays through keep their original owner.

`obs["moves"]`: N rows, one per legal move, sorted by Quackle's equity:

| key | shape | meaning |
|---|---|---|
| `kind` | (N,) | 0 place, 1 exchange, 2 pass |
| `row`, `col` | (N,) | start square of a placement (−1 otherwise) |
| `horizontal` | (N,) bool | direction of a placement |
| `tiles` | (N,7,4) | newly placed tiles: row, col, letter (1–26), is_blank; padded with −1 |
| `ntiles` | (N,) | tiles placed or exchanged |
| `score` | (N,) | points the move scores |
| `equity` | (N,) float | Quackle's static equity: score + leave value + small terms |
| `leave` | (N,27) | rack counts after the move (before drawing) |
| `bingo` | (N,) bool | uses all 7 tiles |

`equity` is the Speedy Player's evaluation, so `argmax(equity)` *is* the
Speedy baseline, and it doubles as a score-plus-leave prior. `tiles` and
`leave` are what stage-1 candidate queries need; the board arrays, `rack` and
`unseen` feed the encoder.

### Afterstates for stage 2

```python
top = stage1_topk(obs)                 # indices into obs["moves"]
a = env.afterstates(obs, top)          # call on the current obs, before env.step()
```

Each candidate's position after its move is placed, **before** replacement
tiles are drawn, seen by the player who moved. Everything has a leading axis K:

| key | shape | meaning |
|---|---|---|
| `letters`, `blanks` | (K,15,15) | board with the move placed |
| `placed` | (K,15,15) bool | squares this move just filled |
| `board_owner`, `board_age` | (K,15,15) | as in the observation; the new tiles are owner 1, age 0 |
| `cross_h`, `cross_v` | (K,15,15) uint32 | cross-checks recomputed after the move |
| `premium` | (K,15,15) int8 | premiums still uncovered after the move |
| `leave` | (K,27) | rack kept |
| `kind`, `move_score`, `equity`, `ntiles`, `bingo` | (K,) | from the move list |
| `score_me`, `score_opp`, `scoreless_turns` | (K,) | scoreboard after the move |
| `unseen`, `bag_size` | (27,), int | unchanged until you draw, shared by all K |

Exchanges and pass leave the board unchanged. The real game is untouched:
afterstates are built on a lightweight copy of the board, so no tiles are
drawn and no random draws are used up. Cost is about 22 µs per candidate
(under 1 ms for K = 32). `example_afterstates.py` prints a small batch and all
the shapes.

### Rule decisions made in `env.py` (change them freely)

- **Two players**, as in tournament Scrabble.
- **Six scoreless turns end the game**, and each player loses the value of
  their rack. Quackle counts scoreless turns but doesn't end the game, so
  `env.py` enforces it.
- **`restrict_pass=True`** offers pass only when the bag is empty or nothing
  else is legal. Set it to `False` for unrestricted passing.
- Phonies and challenges aren't modeled: every generated move is valid.
- Ownership and age are only correct for games played through `env.step()`.

### Word lists

Pass the lexicon name to `ScrabbleEnv(data_dir, lexicon=...)` or
`qenv.init(...)`. English lists shipped in `quackle/data/lexica`: `nwl23`
(North American, default), `csw24` (Collins, international), `nswl23`
(NASPA school list) and `sowpods` (older Collins). Build a GADDAG for each
one you use. The other lists there (Greek, Korean and so on) need a different
alphabet and tile set, which the bridge doesn't support yet.

For a word list of your own, compile a text file (one word per line) into a
`.dawg` with Quackle's `makeminidawg` tool (needs Qt; see
`quackle/makeminidawg/README`), put it in `data/lexica/`, and pass its name.
Leave values are tuned to a lexicon, so with a custom list Quackle falls back
to its generic English leave tables and Speedy gets somewhat weaker.

### The raw bridge (`qenv`)

```python
import qenv
qenv.init("../quackle/data", "nwl23", seed=0)   # once per process
g = qenv.Game()
m = g.legal_moves()            # dict of arrays as above; pass is always last
print(g.move_strings()[:5])    # human-readable, e.g. '8D ONESIES'
g.play(0)                      # index into the last legal_moves()
g.play_speedy()                # let Speedy move instead; returns its move string
print(g.board_string(), g.scores(), g.rack_string(), g.bag_size)
h = g.copy()                   # independent copy, e.g. for lookahead
g.cross_checks()               # (cross_h, cross_v) for the current board
g.afterstates([0, 3, 7])       # boards after those moves, before drawing
qenv.premiums()                # (15,15) premium-square layout
qenv.has_gaddag()              # True if a GADDAG was loaded
qenv.seed(123)                 # reseed tile drawing
```

`g.turn` is Quackle's round number: both players' moves in a round share it.
Ownership and age live in `env.py`, not in `qenv`.

## Speed and parallelism

Measured here on one CPU core, without / with a GADDAG file:

- Move generation: about **2.8 / 1.8 ms per turn on average**.
- Speedy vs Speedy: about **13 / 19 games per second** per core.
- Afterstates: about **22 µs per candidate**.
- Moves per turn: about 1,000 on average, but **up to ~50,000** with
  blank-heavy racks. Cap or sample candidates before stage 1, or a few turns
  will dominate your batch cost.

Your machine may be slower or faster; `test_qenv.py` prints these numbers.

Quackle keeps its lexicon and random-number generator in one global object,
so use **one process per core** (`multiprocessing`), each calling
`qenv.init(...)`, rather than threads.

### Building a GADDAG

Quackle generates moves faster with a GADDAG file, which the repo doesn't ship.
Quackle's own builder needs Qt, so the build compiles a Qt-free one,
`bin/make_gaddag`, that reads the word list out of the `.dawg` Quackle does
ship (takes a few seconds, writes an ~19 MB file):

```bash
bin/make_gaddag ../quackle/data/lexica/nwl23.dawg ../quackle/data/lexica/nwl23.gaddag
```

Over 1,385 turns of seeded games, every turn's full move list was identical
with and without the GADDAG (`tools/dump_moves.py` reproduces this check).
Once the file exists, the "couldn't open gaddag" message at startup goes away.

## Checking the bridge

`test_qenv.py` verifies that:

- every move's leave plus its used tiles equals the rack,
- placed tiles appear on the board after the move,
- `argmax(equity)` picks exactly the move Quackle's own Speedy Player picks,
- `copy()` is independent,
- every afterstate (board, blanks and cross-checks) matches the board you get
  by actually playing that move,
- full games terminate (Speedy averages ~440 points per game, in line with
  published static-bot averages).

## Licenses

Quackle is **GPL-3.0**. This bridge links Quackle, so if you *distribute*
software that includes it, that software must be GPL-compatible. Private
research use is unaffected. The word lists in `quackle/data/lexica` are
covered by the notices in `data/lexica/copyrights.txt`.