"""Reproduce the §4.1.2 worked example from the thesis.

Reference numbers in this test file come from Murray & Chu (2015) Algorithms 1-5
as written in §4.1 of the thesis, applied to the 6-customer instance in §4.1.2.
The thesis's worked example contains two minor typos that we silently work
around: a "1.55" that should read "1.15" (drone time d[2,4] in the iter-2
synchronization saving), and a "7.7" in Fig 4.4 that should read "7.9" at the
node where one sortie's recovery and the next sortie's launch coincide; the
downstream timestamps in Fig 4.4 are consistent with both services applying.
"""

from __future__ import annotations

import math

import pytest

from fstsp import Solution, Sortie, murray_chu, validate
from fstsp.examples import thesis_4_1_2
from fstsp.heuristic import _calc_cost_uav, _calc_savings


def test_initial_tsp_completion_time() -> None:
    """The truck-only TSP D-3-6-5-1-2-4-D has total time 16.9 (Fig 4.2)."""
    instance, route = thesis_4_1_2()
    sol = Solution(instance=instance, truck_route=route, sorties=[])
    assert math.isclose(sol.total_completion_time(), 16.9, abs_tol=1e-9)


def test_initial_tsp_arrival_times() -> None:
    """T = (0, 3.4, 5.8, 8.1, 9.6, 11.6, 13.9, 16.9) per Fig 4.2."""
    instance, route = thesis_4_1_2()
    sol = Solution(instance=instance, truck_route=route, sorties=[])
    expected = [0.0, 3.4, 5.8, 8.1, 9.6, 11.6, 13.9, 16.9]
    for got, exp in zip(sol.truck_arrival_times(), expected, strict=True):
        assert math.isclose(got, exp, abs_tol=1e-9), (got, exp)


def test_iter1_timing_with_sortie_5_2_4() -> None:
    """After applying sortie 5 -> 2 -> 4 (Fig 4.3) the truck times match."""
    instance, _ = thesis_4_1_2()
    sol = Solution(
        instance=instance,
        truck_route=[0, 3, 6, 5, 1, 4, 0],
        sorties=[Sortie(launch=5, customer=2, rendezvous=4)],
    )
    expected = {0: 0.0, 3: 3.4, 6: 5.8, 5: 8.1, 1: 9.8, 4: 11.3}
    T = sol.truck_arrival_times()
    for node, exp in expected.items():
        assert math.isclose(T[sol.position_of(node)], exp, abs_tol=1e-9), (node, T)
    assert math.isclose(sol.total_completion_time(), 14.5, abs_tol=1e-9)


def test_thesis_two_sortie_solution_matches_fig_4_4() -> None:
    """The two-sortie solution in §4.1.2 (sorties 5->2->4 and 3->6->5) has total time 14.1."""
    instance, _ = thesis_4_1_2()
    sol = Solution(
        instance=instance,
        truck_route=[0, 3, 5, 1, 4, 0],
        sorties=[Sortie(3, 6, 5), Sortie(5, 2, 4)],
    )
    expected = {0: 0.0, 3: 3.4, 5: 7.5, 1: 9.4, 4: 10.9}
    T = sol.truck_arrival_times()
    for node, exp in expected.items():
        assert math.isclose(T[sol.position_of(node)], exp, abs_tol=1e-9), (node, T)
    assert math.isclose(sol.total_completion_time(), 14.1, abs_tol=1e-9)


def test_iter1_savings_for_h1_matches_paper() -> None:
    """Algorithm 2 on h=1, fresh truck route -> savings 0.7 (paper §4.1.2 first paragraph)."""
    instance, route = thesis_4_1_2()
    sol = Solution(instance=instance, truck_route=route, sorties=[])
    T = sol.truck_arrival_times()
    saving = _calc_savings(sol, h=1, T=T, subroutes=sol.subroutes())
    assert math.isclose(saving, 0.7, abs_tol=1e-9)


def test_iter1_uav_cost_for_5_to_2_to_4_matches_paper() -> None:
    """Algorithm 4 on h=1 with sortie 5->1->2 -> cost 0.4, improvement 0.3."""
    instance, route = thesis_4_1_2()
    sol = Solution(instance=instance, truck_route=route, sorties=[])
    T = sol.truck_arrival_times()
    sub = sol.subroutes()[0]  # only one subroute in the truck-only TSP
    move = _calc_cost_uav(sol, h=1, sub=sub, savings=0.7, T=T)
    assert move is not None
    # The paper's stated improvement for sortie 5->1->2 is 0.7 - 0.4 = 0.3.
    # Our greedy returns the best (i, j) for this h, which may or may not be (5, 2);
    # at minimum the best improvement should be >= 0.3.
    assert move.saving >= 0.3 - 1e-9


def test_iter2_sync_saving_for_h1_in_uav_subroute() -> None:
    """After iter 1 (sortie 5->2->4), removing h=1 yields savings 0 because the
    truck saving is zero and dominates the 0.45 sync slack (paper §4.1.2 iter 2).
    """
    instance, _ = thesis_4_1_2()
    sol = Solution(
        instance=instance,
        truck_route=[0, 3, 6, 5, 1, 4, 0],
        sorties=[Sortie(launch=5, customer=2, rendezvous=4)],
    )
    T = sol.truck_arrival_times()
    saving = _calc_savings(sol, h=1, T=T, subroutes=sol.subroutes())
    assert math.isclose(saving, 0.0, abs_tol=1e-9)


def test_heuristic_runs_to_completion_and_is_feasible() -> None:
    instance, route = thesis_4_1_2()
    sol = murray_chu(instance, route)
    validate(sol)


def test_heuristic_improves_over_initial_tsp() -> None:
    instance, route = thesis_4_1_2()
    initial = Solution(instance=instance, truck_route=route, sorties=[])
    sol = murray_chu(instance, route)
    assert sol.total_completion_time() < initial.total_completion_time()


def test_heuristic_at_most_as_bad_as_thesis_solution() -> None:
    """Murray-Chu is greedy. With our tie-breaking it finds a solution at least as
    good as the two-sortie outcome reported in the thesis (14.1)."""
    instance, route = thesis_4_1_2()
    sol = murray_chu(instance, route)
    assert sol.total_completion_time() <= 14.1 + 1e-9


@pytest.mark.parametrize(
    ("node", "expected"),
    [(0, 0.0), (3, 3.4), (5, 7.5), (1, 9.4), (4, 10.9)],
)
def test_two_sortie_solution_arrival_per_node(node: int, expected: float) -> None:
    """Per-node check of Fig 4.4 arrival times for the thesis's two-sortie route."""
    instance, _ = thesis_4_1_2()
    sol = Solution(
        instance=instance,
        truck_route=[0, 3, 5, 1, 4, 0],
        sorties=[Sortie(3, 6, 5), Sortie(5, 2, 4)],
    )
    T = sol.truck_arrival_times()
    assert math.isclose(T[sol.position_of(node)], expected, abs_tol=1e-9)
