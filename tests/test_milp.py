"""Exact MILP tests. Skipped when gurobipy or a Gurobi licence is unavailable."""

from __future__ import annotations

import math

import pytest

from fstsp.examples import thesis_4_1_2
from fstsp.heuristic import murray_chu
from fstsp.instances import (
    initial_truck_solution,
    nearest_neighbour_route,
    random_euclidean,
)
from fstsp.sa import simulated_annealing
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


def test_exact_optimum_on_thesis_instance(env) -> None:
    """The exact optimum on the thesis instance is 12.45, well below the greedy
    Murray-Chu result (14.5). It chains three sorties (0->2->1, 1->5->3, 3->6->7):
    nodes 1 and 3 each act as a rendezvous and then the next launch -- the pattern
    the removed node-exclusivity constraint used to forbid."""
    instance, route = thesis_4_1_2()
    res = solve_milp(instance, env=env)
    assert res.solution is not None
    assert res.objective is not None
    assert math.isclose(res.objective, 12.45, abs_tol=1e-6)
    assert res.objective < murray_chu(instance, route).total_completion_time()
    launches = {s.launch for s in res.solution.sorties}
    rendezvous = {s.rendezvous for s in res.solution.sorties}
    assert launches & rendezvous, "expected chained sorties (shared launch/rendezvous node)"


def test_exact_optimum_never_beaten_by_heuristics(env) -> None:
    """An exact optimum can never lose to a heuristic. This pins the chaining bug:
    on n=10 seed 5 the MILP used to return 15558 while the heuristic found 15500,
    because it could not represent chained sorties."""
    for seed in (1, 5):
        inst = random_euclidean(n_customers=10, seed=seed)
        res = solve_milp(inst, env=env, time_limit=120)
        assert res.status == "optimal"
        assert res.solution is not None
        assert res.objective is not None
        heuristic = murray_chu(inst, nearest_neighbour_route(inst)).total_completion_time()
        sa = simulated_annealing(initial_truck_solution(inst), iterations=30000, seed=0)
        assert res.objective <= heuristic + 1e-6
        assert res.objective <= sa.best_objective + 1e-6
        validate(res.solution)
        assert math.isclose(res.objective, res.solution.total_completion_time(), abs_tol=1e-6)
