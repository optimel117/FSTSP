"""Random instance generation + truck-only TSP construction.

The Euclidean generator follows Rafael's SA setup (in `legacy/`): customers
are dropped uniformly into a square, distances are Euclidean, and travel
times are linear in distance with separate truck- and drone-specific
seconds-per-km. The default parameters are the ones Rafael uses in the
thesis (truck 156 s/km, drone 78 s/km, SL = SR = 60 s, 15 km endurance
budget). Units of time are seconds — the same numbers flow straight into
`Instance.t`, `Instance.d`, `sl`, `sr`, and `drone_endurance`.

Endurance convention: this generator stores `drone_endurance` in the Boccia
sense used elsewhere in this package, i.e. the constraint at validation time
is `max(drone_flight, truck_segment) + SR <= drone_endurance`. Rafael's SA
in `legacy/` uses the slightly tighter `SL + max(...) + SR <= Dtl`; if you
want strict numerical equivalence, pass `drone_endurance = Dtl - SL`.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from fstsp.instance import Instance
from fstsp.solution import Solution


def random_euclidean(
    n_customers: int = 20,
    *,
    area_side_km: float = 30.0,
    depot_position: Literal["center", "corner"] = "center",
    seed: int = 0,
    truck_seconds_per_km: float = 156.0,
    drone_seconds_per_km: float = 78.0,
    sl: float = 60.0,
    sr: float = 60.0,
    drone_endurance: float = 1170.0,
) -> Instance:
    """A random Euclidean FSTSP instance.

    Node 0 is the depot; customers are 1..n_customers. Coordinates are stored
    on the returned Instance (used by `viz` and any downstream visualisation)
    and travel-time matrices are computed in seconds.
    """
    if n_customers < 1:
        raise ValueError("n_customers must be at least 1")
    if depot_position not in {"center", "corner"}:
        raise ValueError("depot_position must be 'center' or 'corner'")

    rng = np.random.default_rng(seed)
    coords = np.empty((n_customers + 1, 2), dtype=float)
    if depot_position == "center":
        coords[0] = (0.0, 0.0)
        coords[1:] = rng.uniform(-area_side_km / 2, area_side_km / 2, size=(n_customers, 2))
    else:
        coords[0] = (0.0, 0.0)
        coords[1:] = rng.uniform(0.0, area_side_km, size=(n_customers, 2))

    diffs = coords[:, None, :] - coords[None, :, :]
    dist = np.linalg.norm(diffs, axis=-1)
    t = truck_seconds_per_km * dist
    d = drone_seconds_per_km * dist

    return Instance(
        depot=0,
        customers=tuple(range(1, n_customers + 1)),
        t=t,
        d=d,
        drone_endurance=drone_endurance,
        sl=sl,
        sr=sr,
        coords=coords,
    )


def nearest_neighbour_route(inst: Instance) -> list[int]:
    """Greedy nearest-neighbour TSP tour, starting and ending at the depot."""
    unvisited = set(inst.customers)
    route = [inst.depot]
    current = inst.depot
    while unvisited:
        nxt = min(unvisited, key=lambda j: inst.t[current, j])
        route.append(nxt)
        unvisited.remove(nxt)
        current = nxt
    route.append(inst.depot)
    return route


def _route_cost(route: list[int], t: np.ndarray) -> float:
    return float(sum(t[route[k], route[k + 1]] for k in range(len(route) - 1)))


def two_opt(route: list[int], inst: Instance) -> list[int]:
    """First-improvement 2-opt on a closed tour. Leaves the depot endpoints fixed."""
    best = route[:]
    best_cost = _route_cost(best, inst.t)
    improved = True
    while improved:
        improved = False
        n = len(best)
        for i in range(1, n - 2):
            for k in range(i + 1, n - 1):
                candidate = best[:i] + best[i : k + 1][::-1] + best[k + 1 :]
                cand_cost = _route_cost(candidate, inst.t)
                if cand_cost + 1e-12 < best_cost:
                    best, best_cost = candidate, cand_cost
                    improved = True
                    break
            if improved:
                break
    return best


def initial_truck_solution(inst: Instance) -> Solution:
    """Truck-only initial Solution: NN tour, refined by 2-opt, no sorties."""
    route = two_opt(nearest_neighbour_route(inst), inst)
    return Solution(instance=inst, truck_route=route, sorties=[])
