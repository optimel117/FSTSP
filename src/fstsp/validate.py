"""Feasibility checks for FSTSP solutions, mirroring the Boccia (2023) constraints."""

from __future__ import annotations

from itertools import pairwise

from fstsp.solution import Solution


class FeasibilityError(ValueError):
    pass


def validate(sol: Solution) -> None:
    """Raise FeasibilityError if `sol` violates any FSTSP constraint.

    Checks:
      * truck route starts and ends at the depot;
      * no customer is visited by both truck and drone;
      * each customer is served exactly once;
      * sortie launch and rendezvous nodes occur in the truck route in the
        correct order, with no two sorties overlapping;
      * each sortie respects the drone endurance limit. The drone is airborne
        for max(drone flight, truck segment) and must land within Dtl - SR, so
        BOTH the drone flight (d[i,h] + d[h,j]) and the truck segment between
        launch and rendezvous are bounded by Dtl - SR (Boccia constraints).
    """
    inst = sol.instance
    route = sol.truck_route

    if not route or route[0] != inst.depot or route[-1] != inst.end_depot:
        raise FeasibilityError("truck route must start at the depot and end at the end-depot")

    truck_customers = [n for n in route[1:-1] if n in inst.customers]
    drone_customers = [s.customer for s in sol.sorties]

    if len(truck_customers) != len(set(truck_customers)):
        raise FeasibilityError("truck visits a customer more than once")
    if len(drone_customers) != len(set(drone_customers)):
        raise FeasibilityError("drone visits the same customer in multiple sorties")

    overlap = set(truck_customers) & set(drone_customers)
    if overlap:
        raise FeasibilityError(f"customer(s) served by both truck and drone: {sorted(overlap)}")

    served = set(truck_customers) | set(drone_customers)
    expected = set(inst.customers)
    if served != expected:
        missing = sorted(expected - served)
        extra = sorted(served - expected)
        raise FeasibilityError(f"customer service mismatch: missing={missing}, extra={extra}")

    intervals: list[tuple[int, int]] = []
    for s in sol.sorties:
        if s.launch not in route:
            raise FeasibilityError(f"sortie launch {s.launch} not in truck route")
        if s.rendezvous not in route:
            raise FeasibilityError(f"sortie rendezvous {s.rendezvous} not in truck route")
        lp = sol.position_of(s.launch)
        rp = sol.position_of(s.rendezvous)
        if rp <= lp:
            raise FeasibilityError(
                f"sortie ({s.launch}->{s.customer}->{s.rendezvous}) has rendezvous before launch"
            )
        intervals.append((lp, rp))

        # Boccia endurance: the drone is airborne for max(drone flight, truck
        # segment) and must land within Dtl - SR, so BOTH legs are bounded.
        limit = inst.drone_endurance - inst.sr
        drone_leg = inst.drone_time(s.launch, s.customer) + inst.drone_time(
            s.customer, s.rendezvous
        )
        if drone_leg > limit + 1e-9:
            raise FeasibilityError(
                f"sortie ({s.launch}->{s.customer}->{s.rendezvous}) violates endurance "
                f"(drone flight): {drone_leg:.4f} > {limit:.4f}"
            )
        truck_leg = sum(inst.truck_time(route[k], route[k + 1]) for k in range(lp, rp))
        if truck_leg > limit + 1e-9:
            raise FeasibilityError(
                f"sortie ({s.launch}->{s.customer}->{s.rendezvous}) violates endurance "
                f"(truck segment): {truck_leg:.4f} > {limit:.4f}"
            )

    intervals.sort()
    for (a1, b1), (a2, b2) in pairwise(intervals):
        if b1 > a2:
            raise FeasibilityError(f"sorties overlap on truck route: {(a1, b1)} and {(a2, b2)}")


def is_feasible(sol: Solution) -> bool:
    """Non-raising counterpart to :func:`validate` (handy in tight search loops)."""
    try:
        validate(sol)
    except FeasibilityError:
        return False
    return True
