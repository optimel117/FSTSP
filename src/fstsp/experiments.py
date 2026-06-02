"""Run the three solution methods over Rafael's experiment grid and collect
comparable, fully-reproducible records.

The design factors (Rafael's thesis spec) are crossed completely:

* instance size ``n``         -- default {10, 20, 40}
* replication ``r``           -- default 1..10  (10 customer sets per size)
* drone endurance ``D_tl``    -- default {1170, 1950, 2700} seconds
* hub position                -- {center, corner}

That is 3 sizes x 10 reps x 3 endurances x 2 hubs = 180 runs, each solved by the
MILP (exact, lazy subtour cuts), the Murray-Chu heuristic, and SA.

Reproducibility: the customer cloud is fixed by the seed alone, and the seed is
``2026 + 1000*n + r`` -- it depends on size and replication but NOT on endurance
or hub. So all six (endurance x hub) variants of a given (n, r) share the same
customers; the hub only moves the depot and endurance never touches the RNG.
Every record stores its seed, so any instance and any SA run can be regenerated.

The module holds the logic (so it can be tested); ``scripts/run_experiments.py``
is the CLI that writes the CSVs, prints tables, and draws figures.
"""

from __future__ import annotations

import csv
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, cast

from fstsp.heuristic import murray_chu
from fstsp.instance import Instance
from fstsp.instances import initial_truck_solution, nearest_neighbour_route, two_opt
from fstsp.sa import simulated_annealing
from fstsp.solution import Solution
from fstsp.validate import is_feasible

# fstsp.milp / gurobipy are imported lazily inside run_milp so the heuristic and SA
# runners work on machines without Gurobi installed (e.g. the remote worker).

# Defaults matching the thesis grid; the CLI can override every one.
SIZES = (10, 20, 40)
REPLICATIONS = tuple(range(1, 11))  # r = 1..10
ENDURANCES = (1170.0, 1950.0, 2700.0)
HUBS = ("center", "corner")

CSV_FIELDS = (
    "method",
    "n",
    "replication",
    "seed",
    "hub",
    "endurance",
    "sa_seed",
    "objective",
    "runtime_s",
    "status",
    "proven_optimal",
    "mip_gap",
    "n_sorties",
    "truck_route",
    "drone_sorties",
    "feasible",
)


def make_seed(n: int, replication: int) -> int:
    """The instance seed for size ``n`` and replication ``r`` (Rafael's formula).

    ``2026 + 1000*n + r``: distinct sizes get well-separated seeds (so a bigger
    instance is not a superset of a smaller one), while the seed is independent of
    endurance and hub so those factors can be compared on identical customers.
    """
    return 2026 + 1000 * n + replication


@dataclass
class RunRecord:
    """One (method, instance-config, run) outcome, flattened for CSV.

    The instance config is (n, replication, hub, endurance); ``seed`` is derived
    from (n, replication) via :func:`make_seed`. ``sa_seed`` is SA's own
    stochastic seed (0 for the deterministic methods). The unique key is
    (method, n, replication, endurance, hub, sa_seed). ``truck_route`` and
    ``drone_sorties`` are human-readable strings ("D" = end depot).
    """

    method: str  # "milp" | "heuristic" | "sa"
    n: int
    replication: int
    seed: int
    hub: str
    endurance: float
    sa_seed: int
    objective: float | None
    runtime_s: float
    status: str
    proven_optimal: bool
    mip_gap: float | None
    n_sorties: int
    truck_route: str
    drone_sorties: str
    feasible: bool


def _label(node: int, inst: Instance) -> str:
    return "D" if node == inst.end_depot else str(node)


def _fmt_route(sol: Solution) -> str:
    """Truck route as "0>3>6>...>D" (start depot 0, end depot D)."""
    return ">".join(_label(n, sol.instance) for n in sol.truck_route)


def _fmt_sorties(sol: Solution) -> str:
    """Drone sorties as "launch>customer>rendezvous" segments joined by " | "."""
    inst = sol.instance
    return " | ".join(
        f"{_label(s.launch, inst)}>{s.customer}>{_label(s.rendezvous, inst)}"
        for s in sol.sorties
    )


def _stats(sol: Solution) -> tuple[int, str, str, bool]:
    return len(sol.sorties), _fmt_route(sol), _fmt_sorties(sol), is_feasible(sol)


def run_milp(
    inst: Instance, n: int, r: int, hub: str, endurance: float, *, time_limit: float, env=None
) -> RunRecord:
    from fstsp.milp import solve_milp  # lazy: keeps gurobipy off the heuristic/SA path

    res = solve_milp(inst, time_limit=time_limit, env=env)
    seed = make_seed(n, r)
    if res.solution is None:
        return RunRecord(
            "milp", n, r, seed, hub, endurance, 0, None, res.runtime, res.status,
            False, res.gap, 0, "", "", False,
        )
    ns, route, sorties, feas = _stats(res.solution)
    return RunRecord(
        "milp", n, r, seed, hub, endurance, 0, res.objective, res.runtime, res.status,
        res.is_optimal, res.gap, ns, route, sorties, feas,
    )


def run_heuristic(inst: Instance, n: int, r: int, hub: str, endurance: float) -> RunRecord:
    route = two_opt(nearest_neighbour_route(inst), inst)
    t0 = time.perf_counter()
    sol = murray_chu(inst, route)
    runtime = time.perf_counter() - t0
    ns, route_str, sorties, feas = _stats(sol)
    return RunRecord(
        "heuristic", n, r, make_seed(n, r), hub, endurance, 0,
        sol.total_completion_time(), runtime, "heuristic", False, None,
        ns, route_str, sorties, feas,
    )


def run_sa(
    inst: Instance, n: int, r: int, hub: str, endurance: float, *, iterations: int, sa_seed: int
) -> RunRecord:
    start = initial_truck_solution(inst)
    t0 = time.perf_counter()
    res = simulated_annealing(start, iterations=iterations, seed=sa_seed)
    runtime = time.perf_counter() - t0
    ns, route_str, sorties, feas = _stats(res.best)
    return RunRecord(
        "sa", n, r, make_seed(n, r), hub, endurance, sa_seed,
        res.best_objective, runtime, "sa", False, None,
        ns, route_str, sorties, feas,
    )


def run_suite(
    *,
    sizes: tuple[int, ...] = SIZES,
    replications: tuple[int, ...] = REPLICATIONS,
    endurances: tuple[float, ...] = ENDURANCES,
    hubs: tuple[str, ...] = HUBS,
    instance_kwargs: dict | None = None,
    methods: tuple[str, ...] = ("milp", "heuristic", "sa"),
    sa_iterations: int = 100_000,
    sa_repetitions: int = 1,
    milp_time_limit: float = 600.0,
    milp_max_n: int | None = None,
    env=None,
    progress=None,
    skip: set[tuple] | None = None,
) -> list[RunRecord]:
    """Run the full crossed grid (size x replication x endurance x hub).

    Each instance config gets the methods in ``methods`` (MILP skipped when no
    ``env``, when "milp" is not requested, or when ``n > milp_max_n``). SA is
    stochastic, so it is repeated ``sa_repetitions`` times with seeds 0..reps-1.
    ``progress`` is an optional ``(record) -> None`` callback for live logging /
    incremental writes; ``skip`` is a set of done keys for resuming.
    """
    from fstsp.instances import random_euclidean

    ikwargs = instance_kwargs or {}
    done = skip or set()
    records: list[RunRecord] = []
    want_milp = "milp" in methods and env is not None
    want_heur = "heuristic" in methods
    want_sa = "sa" in methods

    def emit(rec: RunRecord) -> None:
        records.append(rec)
        if progress is not None:
            progress(rec)

    for n in sorted(sizes):
        for r in sorted(replications):
            seed = make_seed(n, r)
            for hub in hubs:
                for endurance in endurances:
                    need_milp = (
                        want_milp
                        and (milp_max_n is None or n <= milp_max_n)
                        and ("milp", n, r, endurance, hub, 0) not in done
                    )
                    need_heur = want_heur and ("heuristic", n, r, endurance, hub, 0) not in done
                    sa_todo = (
                        [k for k in range(sa_repetitions)
                         if ("sa", n, r, endurance, hub, k) not in done]
                        if want_sa else []
                    )
                    if not (need_milp or need_heur or sa_todo):
                        continue
                    inst = random_euclidean(
                        n_customers=n, seed=seed,
                        depot_position=cast(Literal["center", "corner"], hub),
                        drone_endurance=endurance, **ikwargs,
                    )
                    if need_milp:
                        emit(run_milp(inst, n, r, hub, endurance,
                                      time_limit=milp_time_limit, env=env))
                    if need_heur:
                        emit(run_heuristic(inst, n, r, hub, endurance))
                    for sa_seed in sa_todo:
                        emit(run_sa(inst, n, r, hub, endurance,
                                    iterations=sa_iterations, sa_seed=sa_seed))
    return records


def optimality_gaps(records: list[RunRecord]) -> list[dict]:
    """Per-config relative gap of each heuristic vs the proven MILP optimum.

    gap = (method_obj - milp_obj) / milp_obj, matched on
    (n, replication, endurance, hub). Only configs where the MILP proved
    optimality contribute (others have no trustworthy reference).
    """
    key = lambda r: (r.n, r.replication, r.endurance, r.hub)  # noqa: E731
    milp_opt: dict[tuple, float] = {
        key(r): r.objective
        for r in records
        if r.method == "milp" and r.proven_optimal and r.objective is not None
    }
    rows: list[dict] = []
    for r in records:
        if r.method == "milp" or r.objective is None:
            continue
        ref = milp_opt.get(key(r))
        if ref is None or ref <= 0:
            continue
        rows.append(
            {
                "method": r.method,
                "n": r.n,
                "replication": r.replication,
                "endurance": r.endurance,
                "hub": r.hub,
                "sa_seed": r.sa_seed,
                "gap": (r.objective - ref) / ref,
            }
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
        replication=int(row["replication"]),
        seed=int(row["seed"]),
        hub=row["hub"],
        endurance=float(row["endurance"]),
        sa_seed=int(row["sa_seed"]),
        objective=opt_float(row["objective"]),
        runtime_s=float(row["runtime_s"]),
        status=row["status"],
        proven_optimal=row["proven_optimal"] == "True",
        mip_gap=opt_float(row["mip_gap"]),
        n_sorties=int(row["n_sorties"]),
        truck_route=row["truck_route"],
        drone_sorties=row["drone_sorties"],
        feasible=row["feasible"] == "True",
    )


def read_csv(path: str | Path) -> list[RunRecord]:
    """Load previously written records (for resuming / summarising a finished run)."""
    with Path(path).open(newline="") as f:
        return [_record_from_row(row) for row in csv.DictReader(f)]


def done_keys(records: list[RunRecord]) -> set[tuple]:
    """(method, n, replication, endurance, hub, sa_seed) keys already present."""
    return {(r.method, r.n, r.replication, r.endurance, r.hub, r.sa_seed) for r in records}


# --- instance coordinate dump -------------------------------------------------

INSTANCE_FIELDS = ("n", "replication", "seed", "hub", "node", "role", "x", "y")


def write_instances_csv(
    path: str | Path,
    *,
    sizes: tuple[int, ...] = SIZES,
    replications: tuple[int, ...] = REPLICATIONS,
    hubs: tuple[str, ...] = HUBS,
) -> int:
    """Write the customer/depot coordinates for every (n, replication, hub).

    Coordinates are independent of endurance, so this is one block per
    (n, replication, hub). Lets Rafael see the actual instance behind each run.
    Returns the number of coordinate rows written.
    """
    from fstsp.instances import random_euclidean

    rows = 0
    with Path(path).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=INSTANCE_FIELDS)
        w.writeheader()
        for n in sorted(sizes):
            for r in sorted(replications):
                seed = make_seed(n, r)
                for hub in hubs:
                    inst = random_euclidean(
                        n_customers=n, seed=seed,
                        depot_position=cast(Literal["center", "corner"], hub),
                    )
                    assert inst.coords is not None
                    for node in range(n + 1):
                        x, y = inst.coords[node]
                        w.writerow({
                            "n": n, "replication": r, "seed": seed, "hub": hub,
                            "node": node, "role": "depot" if node == 0 else "customer",
                            "x": f"{x:.4f}", "y": f"{y:.4f}",
                        })
                        rows += 1
    return rows
