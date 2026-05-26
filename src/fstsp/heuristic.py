from __future__ import annotations

from dataclasses import dataclass

from fstsp.instance import Instance
from fstsp.solution import Solution, Sortie, Subroute

EPS = 1e-9


@dataclass(frozen=True)
class Move:
    """A candidate reassignment evaluated by Algorithms 3 & 4."""

    h: int
    i: int
    j: int
    served_by_uav: bool
    saving: float


def murray_chu(instance: Instance, initial_route: list[int]) -> Solution:
    """Murray & Chu (2015) FSTSP heuristic, Algorithms 1-5 of the thesis.

    Starts from `initial_route` (a truck-only TSP, including depot at both ends)
    and iteratively applies the single best reassignment until no positive-saving
    move remains.
    """
    sol = Solution(instance=instance, truck_route=list(initial_route), sorties=[])
    eligible: set[int] = set(instance.customers) & set(initial_route)

    while True:
        best = _find_best_move(sol, eligible)
        if best is None:
            return sol
        _apply_move(sol, best)
        if best.served_by_uav:
            eligible.discard(best.h)
            eligible.discard(best.i)
            eligible.discard(best.j)


def _find_best_move(sol: Solution, eligible: set[int]) -> Move | None:
    best: Move | None = None
    subroutes = sol.subroutes()
    T = sol.truck_arrival_times()

    for h in sorted(eligible):
        savings = _calc_savings(sol, h, T, subroutes)
        for sub in subroutes:
            if h in {sol.truck_route[p] for p in sub.positions} and sub.sortie is None:
                cand = _calc_cost_uav(sol, h, sub, savings, T)
            elif sub.sortie is not None:
                cand = _calc_cost_truck(sol, h, sub, savings, T)
            else:
                cand = _calc_cost_uav(sol, h, sub, savings, T)
            if cand is None:
                continue
            if best is None or cand.saving > best.saving + EPS:
                best = cand
    if best is None or best.saving <= EPS:
        return None
    return best


def _calc_savings(sol: Solution, h: int, T: list[float], subroutes: list[Subroute]) -> float:
    """Algorithm 2: savings from removing h from the truck route at its current position.

    Bounded by the sync slack at the rendezvous when h sits inside a UAV-paired
    subroute (line 9 of Algorithm 2).
    """
    inst = sol.instance
    route = sol.truck_route
    pos_h = sol.position_of(h)
    i_node = route[pos_h - 1]
    j_node = route[pos_h + 1]
    savings = inst.t[i_node, h] + inst.t[h, j_node] - inst.t[i_node, j_node]

    sub = _subroute_containing(subroutes, pos_h)
    if sub is not None and sub.sortie is not None:
        s = sub.sortie
        a = s.launch
        b = s.rendezvous
        h_prime = s.customer
        T_prime_b = _truck_arrival_with_h_removed(sol, h, target_position=sub.positions[-1])
        T_a = T[sol.position_of(a)]
        sync_saving = T_prime_b - (T_a + inst.d[a, h_prime] + inst.d[h_prime, b] + inst.sr)
        savings = min(savings, sync_saving)
    return savings


def _calc_cost_truck(
    sol: Solution, h: int, sub: Subroute, savings: float, T: list[float]
) -> Move | None:
    """Algorithm 3: try to insert h between adjacent (i, j) of a UAV-paired subroute."""
    del T  # unused; Algorithm 3 reasons in terms of edge truck-times only
    inst = sol.instance
    route = sol.truck_route
    nodes = [route[p] for p in sub.positions]
    if h in nodes:
        return None  # not a useful move within h's own subroute via this path
    truck_subroute_time = sum(inst.t[nodes[k], nodes[k + 1]] for k in range(len(nodes) - 1))

    best: Move | None = None
    for k in range(len(nodes) - 1):
        i_node, j_node = nodes[k], nodes[k + 1]
        cost = inst.t[i_node, h] + inst.t[h, j_node] - inst.t[i_node, j_node]
        if cost >= savings:
            continue
        # Endurance: existing UAV must still complete its sortie while the truck takes
        # the longer (with-h) path between the launch and rendezvous nodes.
        if truck_subroute_time + cost > inst.drone_endurance - inst.sr + EPS:
            continue
        improvement = savings - cost
        if best is None or improvement > best.saving + EPS:
            best = Move(h=h, i=i_node, j=j_node, served_by_uav=False, saving=improvement)
    return best


def _calc_cost_uav(
    sol: Solution, h: int, sub: Subroute, savings: float, T: list[float]
) -> Move | None:
    """Algorithm 4: try to assign h as a drone-served customer over (i, j) in a non-UAV subroute."""
    inst = sol.instance
    route = sol.truck_route
    nodes = [route[p] for p in sub.positions]
    # Algorithm 4 allows (i, j) within h's own subroute provided i precedes j and neither is h.
    candidate_nodes = [n for n in nodes if n != h] if h in nodes else nodes

    # Truck segment between launch and rendezvous, measured on the route with h
    # removed (h becomes drone-served). Needed for the truck-segment endurance bound.
    removed = [n for n in route if n != h]

    best: Move | None = None
    for a_idx in range(len(candidate_nodes) - 1):
        for b_idx in range(a_idx + 1, len(candidate_nodes)):
            i_node, j_node = candidate_nodes[a_idx], candidate_nodes[b_idx]
            drone_leg = inst.d[i_node, h] + inst.d[h, j_node]
            if drone_leg > inst.drone_endurance - inst.sr + EPS:
                continue
            # Boccia: the drone is airborne until the rendezvous, so the truck
            # segment launch->rendezvous must also fit within Dtl - SR.
            ip, jp = removed.index(i_node), removed.index(j_node)
            truck_seg = sum(inst.t[removed[k], removed[k + 1]] for k in range(ip, jp))
            if truck_seg > inst.drone_endurance - inst.sr + EPS:
                continue
            j_pos = sol.position_of(j_node)
            T_prime_j = _truck_arrival_with_h_removed(sol, h, target_position=j_pos)
            T_i = T[sol.position_of(i_node)]
            delta = T_prime_j - T_i
            cost = max(
                0.0,
                max(delta + inst.sl + inst.sr, drone_leg + inst.sl + inst.sr) - delta,
            )
            improvement = savings - cost
            if best is None or improvement > best.saving + EPS:
                best = Move(h=h, i=i_node, j=j_node, served_by_uav=True, saving=improvement)
    return best


def _apply_move(sol: Solution, move: Move) -> None:
    """Algorithm 5: apply the best move in-place."""
    if move.served_by_uav:
        # Remove h from the truck route and add the new sortie.
        sol.truck_route.remove(move.h)
        sol.sorties.append(Sortie(launch=move.i, customer=move.h, rendezvous=move.j))
    else:
        # Truck reinsertion: remove h from its current position, insert between i and j (adjacent).
        sol.truck_route.remove(move.h)
        i_pos = sol.truck_route.index(move.i)
        j_pos = sol.truck_route.index(move.j)
        if j_pos != i_pos + 1:
            raise RuntimeError(
                "calcCostTruck must only propose adjacent (i, j); got non-adjacent insertion"
            )
        sol.truck_route.insert(j_pos, move.h)


def _subroute_containing(subroutes: list[Subroute], position: int) -> Subroute | None:
    for sub in subroutes:
        if position in sub.positions and position not in (sub.positions[0], sub.positions[-1]):
            return sub
    # Fallback: position is exactly an endpoint (only if it's the route's start/end).
    for sub in subroutes:
        if position in sub.positions:
            return sub
    return None


def _truck_arrival_with_h_removed(sol: Solution, h: int, *, target_position: int) -> float:
    """Truck arrival time at `target_position` if customer h were removed from the route.

    Uses a temporary clone so the live solution is unchanged.
    """
    if h not in sol.truck_route:
        return sol.truck_arrival_times()[target_position]
    pos_h = sol.position_of(h)
    new_route = sol.truck_route[:pos_h] + sol.truck_route[pos_h + 1 :]
    new_target = target_position if target_position < pos_h else target_position - 1
    if new_target < 0 or new_target >= len(new_route):
        raise IndexError("target position out of range after removing h")
    tmp = Solution(instance=sol.instance, truck_route=new_route, sorties=list(sol.sorties))
    return tmp.truck_arrival_times()[new_target]
