"""Tests for the random Euclidean generator and the truck-only TSP helpers."""

from __future__ import annotations

import numpy as np
import pytest

from fstsp import (
    Instance,
    initial_truck_solution,
    nearest_neighbour_route,
    random_euclidean,
    two_opt,
    validate,
)


def test_random_euclidean_shape_and_coords():
    inst = random_euclidean(n_customers=10, seed=42)
    assert inst.n_nodes == 11
    assert inst.depot == 0
    assert inst.customers == tuple(range(1, 11))
    assert inst.coords is not None
    assert inst.coords.shape == (11, 2)
    assert inst.t.shape == (11, 11)
    assert inst.d.shape == (11, 11)


def test_random_euclidean_is_seed_deterministic():
    a = random_euclidean(n_customers=5, seed=7)
    b = random_euclidean(n_customers=5, seed=7)
    np.testing.assert_array_equal(a.coords, b.coords)
    np.testing.assert_array_equal(a.t, b.t)


def test_random_euclidean_distance_consistency():
    inst = random_euclidean(n_customers=8, seed=1)
    diffs = inst.coords[:, None, :] - inst.coords[None, :, :]
    euclid = np.linalg.norm(diffs, axis=-1)
    np.testing.assert_allclose(inst.t, 156.0 * euclid)
    np.testing.assert_allclose(inst.d, 78.0 * euclid)


def test_random_euclidean_depot_center_vs_corner():
    centered = random_euclidean(n_customers=20, seed=3, depot_position="center")
    cornered = random_euclidean(n_customers=20, seed=3, depot_position="corner")
    np.testing.assert_array_equal(centered.coords[0], [0.0, 0.0])
    np.testing.assert_array_equal(cornered.coords[0], [0.0, 0.0])
    assert centered.coords[1:].min() < 0  # centered → some negatives
    assert cornered.coords[1:].min() >= 0  # cornered → all in [0, side]


def test_random_euclidean_rejects_bad_args():
    with pytest.raises(ValueError):
        random_euclidean(n_customers=0)
    with pytest.raises(ValueError):
        random_euclidean(depot_position="middle")  # type: ignore[arg-type]


def test_nearest_neighbour_route_is_valid_tour():
    inst = random_euclidean(n_customers=12, seed=5)
    route = nearest_neighbour_route(inst)
    assert route[0] == inst.depot
    assert route[-1] == inst.end_depot
    assert sorted(route[1:-1]) == list(inst.customers)


def test_two_opt_does_not_worsen_route():
    inst = random_euclidean(n_customers=15, seed=11)
    nn = nearest_neighbour_route(inst)

    def cost(r):
        return sum(inst.truck_time(r[k], r[k + 1]) for k in range(len(r) - 1))

    improved = two_opt(nn, inst)
    assert sorted(improved[1:-1]) == list(inst.customers)
    assert improved[0] == inst.depot and improved[-1] == inst.end_depot
    assert cost(improved) <= cost(nn) + 1e-9


def test_initial_truck_solution_is_feasible():
    inst = random_euclidean(n_customers=10, seed=2)
    sol = initial_truck_solution(inst)
    validate(sol)  # raises FeasibilityError if not feasible
    assert sol.sorties == []
    assert sol.truck_route[0] == inst.depot
    assert sol.truck_route[-1] == inst.end_depot


def test_instance_rejects_wrong_coords_shape():
    inst = random_euclidean(n_customers=3, seed=0)
    with pytest.raises(ValueError, match="coords shape"):
        Instance(
            depot=inst.depot,
            customers=inst.customers,
            t=inst.t,
            d=inst.d,
            drone_endurance=inst.drone_endurance,
            sl=inst.sl,
            sr=inst.sr,
            coords=np.zeros((2, 2)),
        )
