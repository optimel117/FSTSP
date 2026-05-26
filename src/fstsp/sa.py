"""Simulated annealing metaheuristic for the FSTSP.

Ported from Rafael's standalone ``legacy/fstsp_simulated_annealing.py`` to run on
the package's typed :class:`~fstsp.solution.Solution` / :class:`~fstsp.solution.Sortie`
and the feasibility convention used throughout the package.

The objective is :meth:`Solution.total_completion_time`, which is algebraically
identical to the legacy ``Z(S)``: every sortie segment costs
``SL + max(truck_leg, drone_leg) + SR`` and pure-truck segments cost their travel
time, so SA minimises exactly the same quantity as the legacy script.

Feasibility, however, follows the package (Boccia) convention
``d[i,h] + d[h,j] <= drone_endurance - SR`` rather than the legacy bound
``SL + max(truck_leg, drone_leg) + SR <= Dtl``. This keeps SA results directly
comparable with :func:`fstsp.heuristic.murray_chu` rather than bit-identical to
the legacy numbers.

A sortie is never rendezvoused on the final depot: :meth:`Solution.position_of`
resolves the depot to its first occurrence (index 0), so only interior nodes are
valid rendezvous points. Launching from the starting depot is fine.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass

from fstsp.instance import Instance
from fstsp.solution import Solution, Sortie
from fstsp.validate import is_feasible, validate

EPS = 1e-9


@dataclass
class SAResult:
    """Outcome of a simulated-annealing run."""

    best: Solution
    best_objective: float
    final_objective: float
    accepted_moves: int
    improved_moves: int
    infeasible_moves: int
    iterations: int
    runtime_seconds: float


def _clone(sol: Solution) -> Solution:
    """Cheap copy for the search loop.

    Copies only the mutable route list and the (immutable) sortie list; the
    Instance, with its travel-time matrices, is shared rather than deep-copied.
    """
    return Solution(
        instance=sol.instance,
        truck_route=list(sol.truck_route),
        sorties=list(sol.sorties),
    )


def _protected_nodes(sol: Solution) -> set[int]:
    """Launch and rendezvous nodes — these may not be relocated on the route."""
    nodes: set[int] = set()
    for s in sol.sorties:
        nodes.add(s.launch)
        nodes.add(s.rendezvous)
    return nodes


def _feasible_pairs(route: list[int], h: int, inst: Instance) -> list[tuple[int, int]]:
    """Launch/rendezvous ``(i, j)`` pairs whose drone leg fits the endurance bound.

    Pre-filters only on the single-sortie endurance limit; overlap with other
    sorties and route ordering are left to :func:`validate`. The rendezvous is
    never the final depot (see the module docstring).
    """
    limit = inst.drone_endurance - inst.sr + EPS
    last = len(route) - 1
    pairs: list[tuple[int, int]] = []
    for a in range(last):  # launch may be the starting depot (a == 0)
        i = route[a]
        for b in range(a + 1, last):  # rendezvous strictly before the final depot
            j = route[b]
            if inst.d[i, h] + inst.d[h, j] <= limit:
                pairs.append((i, j))
    return pairs


def _move_truck_to_truck(sol: Solution, rng: random.Random) -> None:
    """Relocate a truck-served customer to another position on the truck route."""
    candidates = [h for h in sol.truck_route[1:-1] if h not in _protected_nodes(sol)]
    if not candidates:
        return
    h = rng.choice(candidates)
    route = sol.truck_route
    route.remove(h)
    route.insert(rng.randint(1, len(route) - 1), h)


def _move_truck_to_uav(sol: Solution, rng: random.Random) -> None:
    """Convert a truck-served customer into a new drone sortie."""
    candidates = [h for h in sol.truck_route[1:-1] if h not in _protected_nodes(sol)]
    if not candidates:
        return
    h = rng.choice(candidates)
    route = sol.truck_route
    old_idx = route.index(h)
    route.pop(old_idx)
    pairs = _feasible_pairs(route, h, sol.instance)
    if not pairs:
        route.insert(old_idx, h)  # undo: no endurance-feasible sortie exists for h
        return
    i, j = rng.choice(pairs)
    sol.sorties.append(Sortie(launch=i, customer=h, rendezvous=j))


def _move_uav_to_truck(sol: Solution, rng: random.Random) -> None:
    """Pull a drone-served customer back onto the truck route."""
    if not sol.sorties:
        return
    s = rng.choice(sol.sorties)
    sol.sorties.remove(s)
    sol.truck_route.insert(rng.randint(1, len(sol.truck_route) - 1), s.customer)


def _move_change_sortie(sol: Solution, rng: random.Random) -> None:
    """Re-time an existing sortie: keep its customer, draw new launch/rendezvous."""
    if not sol.sorties:
        return
    old = rng.choice(sol.sorties)
    sol.sorties.remove(old)
    pairs = _feasible_pairs(sol.truck_route, old.customer, sol.instance)
    if not pairs:
        sol.sorties.append(old)  # undo
        return
    i, j = rng.choice(pairs)
    sol.sorties.append(Sortie(launch=i, customer=old.customer, rendezvous=j))


_MOVES_WITH_SORTIES = ("truck_to_truck", "truck_to_uav", "uav_to_truck", "change_sortie")
_MOVES_TRUCK_ONLY = ("truck_to_truck", "truck_to_uav")


def _apply_random_move(sol: Solution, rng: random.Random) -> str:
    """Apply one random neighbourhood move in place; return its name.

    With no sorties yet, only the two moves that can create one are sampled;
    once at least one sortie exists, all four are eligible.
    """
    move = rng.choice(_MOVES_WITH_SORTIES if sol.sorties else _MOVES_TRUCK_ONLY)
    if move == "truck_to_truck":
        _move_truck_to_truck(sol, rng)
    elif move == "truck_to_uav":
        _move_truck_to_uav(sol, rng)
    elif move == "uav_to_truck":
        _move_uav_to_truck(sol, rng)
    else:
        _move_change_sortie(sol, rng)
    return move


def simulated_annealing(
    sol: Solution,
    *,
    iterations: int = 100_000,
    tau_start: float = 0.01,
    tau_final: float = 0.001,
    seed: int = 0,
) -> SAResult:
    """Minimise total completion time by simulated annealing.

    Starts from a feasible solution `sol` (typically
    :func:`fstsp.instances.initial_truck_solution`) and explores four moves:
    relocate a truck customer, convert a truck customer into a drone sortie, pull
    a drone customer back to the truck, or re-time an existing sortie. Acceptance
    is the Metropolis rule with geometric cooling from ``T_start = tau_start * Z0``
    to ``T_final = tau_final * Z0``, where ``Z0`` is the initial objective.

    `seed` controls only the SA search; instance generation is seeded separately
    (see :func:`fstsp.random_euclidean`), mirroring the legacy script's split of
    ``--instance-seed`` and ``--sa-seed``.
    """
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if not 0 < tau_final < tau_start:
        raise ValueError("require 0 < tau_final < tau_start for cooling")

    validate(sol)  # the caller must hand us a feasible starting point

    rng = random.Random(seed)
    start = time.perf_counter()

    current = _clone(sol)
    z_current = current.total_completion_time()
    best = _clone(current)
    z_best = z_current

    t_start = tau_start * z_current
    t_final = tau_final * z_current
    alpha = math.exp(math.log(t_final / t_start) / iterations)
    temperature = t_start

    accepted = improved = infeasible = 0

    for _ in range(iterations):
        candidate = _clone(current)
        _apply_random_move(candidate, rng)

        if not is_feasible(candidate):
            infeasible += 1
            temperature *= alpha
            continue

        z_candidate = candidate.total_completion_time()
        delta = z_candidate - z_current
        if delta <= 0 or rng.random() <= math.exp(-delta / temperature):
            current = candidate
            z_current = z_candidate
            accepted += 1
            if delta <= 0:
                improved += 1
            if z_current < z_best:
                best = _clone(current)
                z_best = z_current

        temperature *= alpha

    return SAResult(
        best=best,
        best_objective=z_best,
        final_objective=z_current,
        accepted_moves=accepted,
        improved_moves=improved,
        infeasible_moves=infeasible,
        iterations=iterations,
        runtime_seconds=time.perf_counter() - start,
    )
