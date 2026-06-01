"""Exact MILP tests. Skipped when gurobipy or a Gurobi licence is unavailable."""

from __future__ import annotations

import math

import pytest

from fstsp.examples import thesis_4_1_2
from fstsp.validate import validate

gp = pytest.importorskip("gurobipy")

from fstsp.gurobi_env import make_env  # noqa: E402
from fstsp.milp import solve_milp  # noqa: E402


@pytest.fixture(scope="module")
def env():
    """A shared Gurobi env; skip the whole module if no licence is available."""
    try:
        e = make_env()
    except gp.GurobiError as exc:  # no licence / WLS unreachable
        pytest.skip(f"no Gurobi licence available: {exc}")
    yield e
    e.dispose()


def test_thesis_instance_solves_to_optimality(env) -> None:
    instance, _ = thesis_4_1_2()
    res = solve_milp(instance, env=env)
    assert res.status == "optimal"
    assert res.solution is not None
    validate(res.solution)


def test_objective_equals_simulated_makespan(env) -> None:
    """The MILP objective must equal the Solution's simulated completion time.

    This guards the depot-launch SL accounting: the objective discounts SL for a
    sortie launched at the start depot (thesis eq 3.7) and Solution._simulate must
    do the same, or the two definitions of completion time drift apart.
    """
    instance, _ = thesis_4_1_2()
    res = solve_milp(instance, env=env)
    assert res.solution is not None
    assert res.objective is not None
    assert math.isclose(
        res.objective, res.solution.total_completion_time(), abs_tol=1e-6
    )


def test_exact_optimum_beats_greedy_heuristic(env) -> None:
    """The exact optimum (13.9, via a depot-launched sortie) is below the greedy
    Murray-Chu result (14.5) on the thesis instance."""
    instance, _ = thesis_4_1_2()
    res = solve_milp(instance, env=env)
    assert res.objective is not None
    assert math.isclose(res.objective, 13.9, abs_tol=1e-6)
