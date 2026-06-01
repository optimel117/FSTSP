"""Experiment harness tests. The MILP path needs Gurobi, so these exercise the
heuristic/SA runners and the gap arithmetic, which don't."""

from __future__ import annotations

from fstsp.experiments import (
    RunRecord,
    optimality_gaps,
    run_heuristic,
    run_sa,
    run_suite,
)
from fstsp.instances import random_euclidean


def test_heuristic_and_sa_runners_produce_feasible_records() -> None:
    inst = random_euclidean(n_customers=8, seed=3)
    h = run_heuristic(inst, n=8, seed=3)
    s = run_sa(inst, n=8, seed=3, iterations=2000, sa_seed=3)
    for rec in (h, s):
        assert rec.feasible
        assert rec.objective is not None and rec.objective > 0
        assert rec.runtime_s >= 0
        # one sortie serves exactly one drone customer in this model
        assert rec.n_sorties == rec.n_drone_customers


def test_run_suite_without_milp_runs_heuristic_and_sa_only() -> None:
    # No small sizes => no MILP => no Gurobi dependency.
    records = run_suite(
        small_sizes=(),
        large_sizes=(6, 9),
        seeds=2,
        sa_iterations=2000,
    )
    methods = {r.method for r in records}
    assert methods == {"heuristic", "sa"}
    assert len(records) == 2 * 2 * 2  # 2 sizes * 2 seeds * 2 methods
    assert all(r.feasible for r in records)


def test_optimality_gaps_uses_only_proven_optima() -> None:
    records = [
        RunRecord("milp", 5, 0, 100.0, 0.1, "optimal", True, 0.0, 1, 1, True),
        RunRecord("heuristic", 5, 0, 110.0, 0.0, "heuristic", False, None, 1, 1, True),
        RunRecord("sa", 5, 0, 105.0, 0.0, "sa", False, None, 1, 1, True),
        # n=6 MILP hit the time limit -> not proven optimal -> no gap reference.
        RunRecord("milp", 6, 0, 200.0, 60.0, "time_limit", False, 0.05, 1, 1, True),
        RunRecord("heuristic", 6, 0, 210.0, 0.0, "heuristic", False, None, 1, 1, True),
    ]
    gaps = {(g["method"], g["n"]): g["gap"] for g in optimality_gaps(records)}
    assert gaps[("heuristic", 5)] == 0.10
    assert gaps[("sa", 5)] == 0.05
    assert ("heuristic", 6) not in gaps  # no proven optimum to compare against
