"""Run the three solution methods over a bank of random instances and collect
comparable records.

Design: small instances (where the MILP is tractable) get all three methods, so
the exact optimum anchors an optimality gap for the heuristic and SA. Large
instances get only the heuristic and SA. The MILP's subtour elimination is
enumerated over customer subsets, so it grows ~3x per customer and is only run up
to ``milp_max_n`` (default 12; n=13 ~ 10s, n=14 ~ 25s on a laptop).

The module holds the logic (so it can be tested); ``scripts/run_experiments.py``
is the CLI that writes a CSV, prints tables, and draws figures.
"""

from __future__ import annotations

import csv
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from fstsp.heuristic import murray_chu
from fstsp.instance import Instance
from fstsp.instances import initial_truck_solution, nearest_neighbour_route, two_opt
from fstsp.sa import simulated_annealing
from fstsp.solution import Solution
from fstsp.validate import is_feasible

# fstsp.milp / gurobipy are imported lazily inside run_milp so the heuristic and SA
# runners work on machines without Gurobi installed (e.g. the remote worker).

CSV_FIELDS = (
    "method",
    "n",
    "seed",
    "objective",
    "runtime_s",
    "status",
    "proven_optimal",
    "mip_gap",
    "n_drone_customers",
    "n_sorties",
    "feasible",
)


@dataclass
class RunRecord:
    """One (method, instance) outcome, flattened for CSV."""

    method: str  # "milp" | "heuristic" | "sa"
    n: int  # number of customers
    seed: int  # instance seed
    objective: float | None
    runtime_s: float
    status: str
    proven_optimal: bool
    mip_gap: float | None
    n_drone_customers: int
    n_sorties: int
    feasible: bool


def _solution_stats(sol: Solution) -> tuple[int, int, bool]:
    return len(sol.drone_customers), len(sol.sorties), is_feasible(sol)


def run_milp(inst: Instance, n: int, seed: int, *, time_limit: float, env=None) -> RunRecord:
    from fstsp.milp import solve_milp  # lazy: keeps gurobipy off the heuristic/SA path

    res = solve_milp(inst, time_limit=time_limit, env=env)
    if res.solution is None:
        return RunRecord(
            "milp", n, seed, None, res.runtime, res.status, False, res.gap, 0, 0, False
        )
    dc, ns, feas = _solution_stats(res.solution)
    return RunRecord(
        "milp", n, seed, res.objective, res.runtime, res.status,
        res.is_optimal, res.gap, dc, ns, feas,
    )


def run_heuristic(inst: Instance, n: int, seed: int) -> RunRecord:
    route = two_opt(nearest_neighbour_route(inst), inst)
    t0 = time.perf_counter()
    sol = murray_chu(inst, route)
    runtime = time.perf_counter() - t0
    dc, ns, feas = _solution_stats(sol)
    return RunRecord(
        "heuristic", n, seed, sol.total_completion_time(), runtime,
        "heuristic", False, None, dc, ns, feas,
    )


def run_sa(inst: Instance, n: int, seed: int, *, iterations: int, sa_seed: int) -> RunRecord:
    start = initial_truck_solution(inst)
    t0 = time.perf_counter()
    res = simulated_annealing(start, iterations=iterations, seed=sa_seed)
    runtime = time.perf_counter() - t0
    dc, ns, feas = _solution_stats(res.best)
    return RunRecord(
        "sa", n, seed, res.best_objective, runtime,
        "sa", False, None, dc, ns, feas,
    )


def run_suite(
    *,
    small_sizes: tuple[int, ...],
    large_sizes: tuple[int, ...],
    seeds: int,
    instance_kwargs: dict | None = None,
    sa_iterations: int = 50_000,
    milp_time_limit: float = 60.0,
    milp_max_n: int = 12,
    env=None,
    progress=None,
    skip: set[tuple[str, int, int]] | None = None,
) -> list[RunRecord]:
    """Run the full experiment bank.

    Small sizes get MILP + heuristic + SA; large sizes get heuristic + SA. The
    MILP is also skipped for any size above ``milp_max_n``. ``progress`` is an
    optional callback ``(record) -> None`` for live logging / incremental writes.
    ``skip`` is a set of ``(method, n, seed)`` keys to omit (for resuming a partial
    run); the instance is only generated when at least one of its methods is due.
    """
    from fstsp.instances import random_euclidean

    ikwargs = instance_kwargs or {}
    done = skip or set()
    records: list[RunRecord] = []

    def emit(rec: RunRecord) -> None:
        records.append(rec)
        if progress is not None:
            progress(rec)

    all_sizes = sorted({*small_sizes, *large_sizes})
    run_milp_for = set(small_sizes)
    for n in all_sizes:
        for seed in range(seeds):
            need_milp = (
                n in run_milp_for and n <= milp_max_n and ("milp", n, seed) not in done
            )
            need_heur = ("heuristic", n, seed) not in done
            need_sa = ("sa", n, seed) not in done
            if not (need_milp or need_heur or need_sa):
                continue
            inst = random_euclidean(n_customers=n, seed=seed, **ikwargs)
            if need_milp:
                emit(run_milp(inst, n, seed, time_limit=milp_time_limit, env=env))
            if need_heur:
                emit(run_heuristic(inst, n, seed))
            if need_sa:
                emit(run_sa(inst, n, seed, iterations=sa_iterations, sa_seed=seed))
    return records


def optimality_gaps(records: list[RunRecord]) -> list[dict]:
    """Per (n, seed) relative gap of each heuristic vs the proven MILP optimum.

    gap = (method_obj - milp_obj) / milp_obj. Only instances where the MILP proved
    optimality contribute; others are skipped (no trustworthy reference).
    """
    milp_opt: dict[tuple[int, int], float] = {
        (r.n, r.seed): r.objective
        for r in records
        if r.method == "milp" and r.proven_optimal and r.objective is not None
    }
    rows: list[dict] = []
    for r in records:
        if r.method == "milp" or r.objective is None:
            continue
        ref = milp_opt.get((r.n, r.seed))
        if ref is None or ref <= 0:
            continue
        rows.append(
            {"method": r.method, "n": r.n, "seed": r.seed, "gap": (r.objective - ref) / ref}
        )
    return rows


def record_to_dict(rec: RunRecord) -> dict:
    return asdict(rec)


def _record_from_row(row: dict) -> RunRecord:
    def opt_float(v: str) -> float | None:
        return float(v) if v not in ("", "None") else None

    return RunRecord(
        method=row["method"],
        n=int(row["n"]),
        seed=int(row["seed"]),
        objective=opt_float(row["objective"]),
        runtime_s=float(row["runtime_s"]),
        status=row["status"],
        proven_optimal=row["proven_optimal"] == "True",
        mip_gap=opt_float(row["mip_gap"]),
        n_drone_customers=int(row["n_drone_customers"]),
        n_sorties=int(row["n_sorties"]),
        feasible=row["feasible"] == "True",
    )


def read_csv(path: str | Path) -> list[RunRecord]:
    """Load previously written records (for resuming / summarising a finished run)."""
    with Path(path).open(newline="") as f:
        return [_record_from_row(row) for row in csv.DictReader(f)]


def done_keys(records: list[RunRecord]) -> set[tuple[str, int, int]]:
    """(method, n, seed) keys already present, for ``run_suite(skip=...)``."""
    return {(r.method, r.n, r.seed) for r in records}
