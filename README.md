# FSTSP

Code accompanying Rafael Hamelink's BSc thesis *Heuristic approach for the FSTSP*
(Tilburg School of Economics and Management). Implements two solution methods
for the Flying Sidekick TSP introduced by Murray & Chu (2015):

- the route-and-reassign heuristic of Murray & Chu (2015), and
- (planned) the arc-based MILP of Boccia et al. (2023).

## Layout

```
src/fstsp/
  instance.py    Instance dataclass (truck/drone matrices, endurance, service times)
  solution.py    Solution / Sortie / Subroute; truck-timing simulator
  heuristic.py   Murray & Chu (2015) Algorithms 1-5
  validate.py    Boccia-style feasibility checks
  examples.py    The §4.1.2 instance fixture
tests/           pytest regression tests pinned to thesis numbers
legacy/          Rafael's first-pass code (kept for reference, not run by tests)
```

## Run

```sh
uv sync --extra dev          # install package + pytest + ruff
uv run pytest                # 15 tests, all green
uv run ruff check .
```

The exact solver lives behind an optional extra:

```sh
uv sync --extra exact        # adds gurobipy
```
