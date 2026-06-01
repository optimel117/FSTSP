"""The depot is split into a start id (`depot`) and an end id (`end_depot`).

These two ids name the same physical location, which lets a truck route be
``[depot, ..., end_depot]`` and lets a sortie launch at the start depot and/or
rendezvous at the end depot without the two colliding under
:meth:`Solution.position_of`.
"""

from __future__ import annotations

import math

import numpy as np

from fstsp import Instance, Solution, Sortie, validate


def _triangle_instance() -> Instance:
    """Depot + two customers, unit truck times, drone twice as fast."""
    t = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    return Instance.from_truck_matrix(t, drone_endurance=5.0, sl=0.2, sr=0.2)


def test_end_depot_id_and_matrix_mapping() -> None:
    inst = _triangle_instance()
    assert inst.end_depot == inst.n_nodes  # one past the last physical node
    assert inst.end_depot not in inst.customers
    assert inst.matrix_index(inst.end_depot) == inst.depot
    # Travel to/from the end-depot reuses the physical depot's row/column.
    assert inst.truck_time(1, inst.end_depot) == inst.t[1, inst.depot]
    assert inst.drone_time(inst.end_depot, 2) == inst.d[inst.depot, 2]


def test_sortie_rendezvous_at_end_depot_is_valid_and_simulates() -> None:
    inst = _triangle_instance()
    ed = inst.end_depot
    sol = Solution(
        instance=inst,
        truck_route=[inst.depot, 1, ed],
        sorties=[Sortie(launch=1, customer=2, rendezvous=ed)],
    )
    validate(sol)  # would raise "rendezvous before launch" under the old single-depot model
    # truck 0->1 (1.0) + SL at the launch (0.2) + max(truck 1->depot, drone 1->2->depot) + SR
    # = 1.0 + 0.2 + max(1.0, 0.5 + 0.5) + 0.2 = 2.4
    assert math.isclose(sol.total_completion_time(), 2.4, abs_tol=1e-9)


def test_launch_at_start_depot_and_rendezvous_at_end_depot_are_distinct() -> None:
    """A single sortie spanning the whole route: launch at the start depot,
    rendezvous at the end depot. The two depot ids must resolve to positions 0
    and last, not collapse to 0."""
    inst = _triangle_instance()
    ed = inst.end_depot
    sol = Solution(
        instance=inst,
        truck_route=[inst.depot, 1, ed],
        sorties=[Sortie(launch=inst.depot, customer=2, rendezvous=ed)],
    )
    validate(sol)
    assert sol.position_of(inst.depot) == 0
    assert sol.position_of(ed) == len(sol.truck_route) - 1
