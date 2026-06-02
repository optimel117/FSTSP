"""Experiment harness tests. The MILP path needs Gurobi, so these exercise the
heuristic/SA runners, the grid, the seed formula, and the gap arithmetic."""

from __future__ import annotations

from fstsp.experiments import (
    RunRecord,
    make_seed,
    optimality_gaps,
    run_heuristic,
    run_sa,
    run_suite,
    write_instances_csv,
)
from fstsp.instances import random_euclidean


def _rec(method, n, r, endurance, hub, obj, *, opt=False, sa_seed=0):
    return RunRecord(
        method, n, r, make_seed(n, r), hub, endurance, sa_seed, obj, 0.1,
        method, opt, 0.0 if opt else None, 1, "0>1>D", "", True,
    )


def test_seed_formula_decorrelates_sizes_but_not_hub_or_endurance() -> None:
    # n is baked into the seed, so n=10 and n=20 at the same replication differ.
    assert make_seed(10, 1) != make_seed(20, 1)
    assert make_seed(10, 1) == 2026 + 1000 * 10 + 1
    # Same seed/hub => identical customers regardless of endurance.
    a = random_euclidean(10, seed=make_seed(10, 1), depot_position="center", drone_endurance=1170)
    b = random_euclidean(10, seed=make_seed(10, 1), depot_position="center", drone_endurance=2700)
    assert (a.coords == b.coords).all()
    # Center vs corner share the customer cloud; only the depot (node 0) moves.
    c = random_euclidean(10, seed=make_seed(10, 1), depot_position="corner")
    assert (a.coords[1:] == c.coords[1:]).all()
    assert (a.coords[0] != c.coords[0]).any()


def test_heuristic_and_sa_runners_produce_feasible_records() -> None:
    inst = random_euclidean(n_customers=8, seed=make_seed(8, 1), drone_endurance=1950)
    h = run_heuristic(inst, 8, 1, "center", 1950.0)
    s = run_sa(inst, 8, 1, "center", 1950.0, iterations=2000, sa_seed=0)
    for rec in (h, s):
        assert rec.feasible
        assert rec.objective is not None and rec.objective > 0
        assert rec.runtime_s >= 0
        assert rec.truck_route.startswith("0>") and rec.truck_route.endswith(">D")


def test_run_suite_without_milp_runs_heuristic_and_sa_only() -> None:
    records = run_suite(
        sizes=(6, 9),
        replications=(1, 2),
        endurances=(1950.0,),
        hubs=("center",),
        methods=("heuristic", "sa"),  # no MILP => no Gurobi dependency
        sa_iterations=1000,
        sa_repetitions=1,
    )
    methods = {r.method for r in records}
    assert methods == {"heuristic", "sa"}
    # 2 sizes * 2 reps * 1 endurance * 1 hub * (1 heuristic + 1 sa)
    assert len(records) == 2 * 2 * 1 * 1 * 2
    assert all(r.feasible for r in records)


def test_hybrid_sa_runs_and_repeats_with_its_own_seeds() -> None:
    records = run_suite(
        sizes=(7,),
        replications=(1,),
        endurances=(1950.0,),
        hubs=("center",),
        methods=("hybrid_sa",),
        sa_iterations=1000,
        sa_repetitions=4,  # hybrid_repetitions defaults to this
    )
    assert {r.method for r in records} == {"hybrid_sa"}
    assert len(records) == 4
    assert {r.sa_seed for r in records} == {0, 1, 2, 3}
    assert all(r.feasible and r.objective and r.objective > 0 for r in records)


def test_run_suite_crosses_endurance_and_hub() -> None:
    records = run_suite(
        sizes=(6,),
        replications=(1,),
        endurances=(1170.0, 2700.0),
        hubs=("center", "corner"),
        methods=("heuristic",),
        sa_repetitions=0,
    )
    # 1 size * 1 rep * 2 endurances * 2 hubs * 1 heuristic
    assert len(records) == 4
    assert {(r.endurance, r.hub) for r in records} == {
        (1170.0, "center"), (1170.0, "corner"), (2700.0, "center"), (2700.0, "corner")
    }


def test_run_suite_repeats_sa_with_distinct_seeds() -> None:
    records = run_suite(
        sizes=(7,),
        replications=(1,),
        endurances=(1950.0,),
        hubs=("center",),
        methods=("sa",),
        sa_iterations=1000,
        sa_repetitions=3,
    )
    sa = [r for r in records if r.method == "sa"]
    assert len(sa) == 3
    assert {r.sa_seed for r in sa} == {0, 1, 2}


def test_optimality_gaps_uses_only_proven_optima() -> None:
    records = [
        _rec("milp", 10, 1, 1950.0, "center", 100.0, opt=True),
        _rec("heuristic", 10, 1, 1950.0, "center", 110.0),
        _rec("sa", 10, 1, 1950.0, "center", 105.0),
        # a config where the MILP only hit the time limit -> no reference
        _rec("milp", 20, 1, 1950.0, "center", 200.0, opt=False),
        _rec("heuristic", 20, 1, 1950.0, "center", 210.0),
    ]
    gaps = {(g["method"], g["n"]): g["gap"] for g in optimality_gaps(records)}
    assert abs(gaps[("heuristic", 10)] - 0.10) < 1e-9
    assert abs(gaps[("sa", 10)] - 0.05) < 1e-9
    assert ("heuristic", 20) not in gaps  # no proven optimum to compare against


def test_write_instances_csv(tmp_path) -> None:
    path = tmp_path / "instances.csv"
    rows = write_instances_csv(path, sizes=(5,), replications=(1, 2), hubs=("center", "corner"))
    # (5 customers + 1 depot) * 2 reps * 2 hubs
    assert rows == 6 * 2 * 2
    assert path.read_text().splitlines()[0].startswith("n,replication,seed,hub,node,role,x,y")
