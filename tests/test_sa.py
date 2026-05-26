"""Tests for the simulated-annealing metaheuristic."""

from __future__ import annotations

import pytest

from fstsp import (
    FeasibilityError,
    initial_truck_solution,
    is_feasible,
    random_euclidean,
    simulated_annealing,
    validate,
)


def _small_instance():
    # A tight area so the drone genuinely pays off and sorties survive the
    # endurance bound; keeps runs short enough for the test suite.
    return random_euclidean(n_customers=8, area_side_km=12.0, seed=4)


def test_sa_best_is_feasible():
    sol = initial_truck_solution(_small_instance())
    result = simulated_annealing(sol, iterations=2000, seed=0)
    validate(result.best)  # raises if infeasible


def test_sa_never_worsens_initial():
    sol = initial_truck_solution(_small_instance())
    z0 = sol.total_completion_time()
    result = simulated_annealing(sol, iterations=2000, seed=0)
    assert result.best_objective <= z0 + 1e-9


def test_sa_is_seed_deterministic():
    sol = initial_truck_solution(_small_instance())
    a = simulated_annealing(sol, iterations=1500, seed=7)
    b = simulated_annealing(sol, iterations=1500, seed=7)
    assert a.best.truck_route == b.best.truck_route
    assert a.best.sorties == b.best.sorties
    assert a.best_objective == b.best_objective


def test_sa_different_seeds_stay_feasible():
    sol = initial_truck_solution(_small_instance())
    for seed in range(3):
        result = simulated_annealing(sol, iterations=1000, seed=seed)
        assert is_feasible(result.best)


def test_sa_does_not_mutate_input_solution():
    sol = initial_truck_solution(_small_instance())
    route_before = list(sol.truck_route)
    sorties_before = list(sol.sorties)
    simulated_annealing(sol, iterations=500, seed=1)
    assert sol.truck_route == route_before
    assert sol.sorties == sorties_before


def test_sa_finds_sorties_on_favourable_instance():
    # With a short endurance-friendly area, SA should discover at least one
    # worthwhile sortie within a modest budget.
    sol = initial_truck_solution(_small_instance())
    result = simulated_annealing(sol, iterations=5000, seed=0)
    assert len(result.best.sorties) >= 1
    assert result.best_objective < sol.total_completion_time()


def test_sa_result_counters_are_consistent():
    sol = initial_truck_solution(_small_instance())
    result = simulated_annealing(sol, iterations=1000, seed=2)
    assert result.iterations == 1000
    assert result.infeasible_moves + result.accepted_moves <= result.iterations
    assert result.improved_moves <= result.accepted_moves
    assert result.runtime_seconds >= 0.0


def test_sa_rejects_bad_arguments():
    sol = initial_truck_solution(_small_instance())
    with pytest.raises(ValueError):
        simulated_annealing(sol, iterations=0)
    with pytest.raises(ValueError):
        simulated_annealing(sol, tau_start=0.001, tau_final=0.01)  # not cooling
    with pytest.raises(ValueError):
        simulated_annealing(sol, tau_final=-0.001)


def test_sa_rejects_infeasible_start():
    inst = _small_instance()
    sol = initial_truck_solution(inst)
    sol.truck_route = sol.truck_route[:-1]  # drop the closing depot -> infeasible
    with pytest.raises(FeasibilityError):
        simulated_annealing(sol, iterations=100)
