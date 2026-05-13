"""
Simplified Simulated Annealing heuristic for the Flying Sidekick TSP (FSTSP).

This script follows the simplified SA structure described in the thesis:
    S = (R, sorties)
where
    R       = truck route, starting at s=0 and ending at t=0
    sorties = list of tuples (i, h, j), meaning launch at i, UAV serves h,
              and rendezvous at j.

The code generates random customer coordinates, builds truck/UAV travel-time
matrices, computes an initial truck-only TSP route using nearest-neighbour
plus 2-opt, and then applies Simulated Annealing.

It prints exactly 50 progress updates: one at each N/50-th iteration.

Author: generated for Rafael Hamelink's FSTSP thesis experimentation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import argparse
import copy
import math
import random
import time

Node = int
Sortie = Tuple[Node, Node, Node]  # (launch, drone_customer, rendezvous)


@dataclass
class Instance:
    coords: Dict[Node, Tuple[float, float]]
    customers: List[Node]
    truck_time: Dict[Tuple[Node, Node], float]
    drone_time: Dict[Tuple[Node, Node], float]
    SL: float
    SR: float
    Dtl: float


@dataclass
class Solution:
    truck_route: List[Node]
    sorties: List[Sortie]


@dataclass
class SAResult:
    best_solution: Solution
    best_objective: float
    final_current_objective: float
    accepted_moves: int
    improved_moves: int
    infeasible_moves: int
    total_iterations: int
    runtime_seconds: float


def euclidean(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def generate_instance(
    n_customers: int = 20,
    area_side_km: float = 30.0,
    depot_position: str = "center",
    seed: int = 1,
    truck_seconds_per_km: float = 156.0,
    drone_seconds_per_km: float = 78.0,
    SL: float = 60.0,
    SR: float = 60.0,
    drone_endurance_km: float = 15.0,
) -> Instance:
    """Generate a random square instance.

    Nodes:
        0 = depot.
        1..n = customers.

    Distance/time assumptions:
        truck_time = 156 seconds per Euclidean km by default;
        drone_time = 78 seconds per Euclidean km by default;
        Dtl = drone_endurance_km * drone_seconds_per_km.

    depot_position:
        'center': depot at the centre of the square;
        'corner': depot at the lower-left corner of the square.
    """
    rng = random.Random(seed)

    if depot_position not in {"center", "corner"}:
        raise ValueError("depot_position must be 'center' or 'corner'.")

    coords: Dict[Node, Tuple[float, float]] = {}
    if depot_position == "center":
        coords[0] = (0.0, 0.0)
        low, high = -area_side_km / 2, area_side_km / 2
        for i in range(1, n_customers + 1):
            coords[i] = (rng.uniform(low, high), rng.uniform(low, high))
    else:
        coords[0] = (0.0, 0.0)
        for i in range(1, n_customers + 1):
            coords[i] = (rng.uniform(0, area_side_km), rng.uniform(0, area_side_km))

    nodes = list(coords.keys())
    truck_time: Dict[Tuple[Node, Node], float] = {}
    drone_time: Dict[Tuple[Node, Node], float] = {}
    for i in nodes:
        for j in nodes:
            if i == j:
                truck_time[(i, j)] = 0.0
                drone_time[(i, j)] = 0.0
            else:
                dist = euclidean(coords[i], coords[j])
                truck_time[(i, j)] = truck_seconds_per_km * dist
                drone_time[(i, j)] = drone_seconds_per_km * dist

    return Instance(
        coords=coords,
        customers=list(range(1, n_customers + 1)),
        truck_time=truck_time,
        drone_time=drone_time,
        SL=SL,
        SR=SR,
        Dtl=drone_endurance_km * drone_seconds_per_km,
    )


def route_cost(route: List[Node], truck_time: Dict[Tuple[Node, Node], float]) -> float:
    return sum(truck_time[(route[i], route[i + 1])] for i in range(len(route) - 1))


def nearest_neighbour_route(instance: Instance) -> List[Node]:
    unvisited = set(instance.customers)
    route = [0]
    current = 0
    while unvisited:
        nxt = min(unvisited, key=lambda j: instance.truck_time[(current, j)])
        route.append(nxt)
        unvisited.remove(nxt)
        current = nxt
    route.append(0)
    return route


def two_opt(route: List[Node], truck_time: Dict[Tuple[Node, Node], float]) -> List[Node]:
    """Simple 2-opt improvement for the initial truck-only route."""
    best = route[:]
    improved = True
    while improved:
        improved = False
        n = len(best)
        for i in range(1, n - 2):
            for k in range(i + 1, n - 1):
                candidate = best[:i] + best[i:k + 1][::-1] + best[k + 1:]
                if route_cost(candidate, truck_time) + 1e-9 < route_cost(best, truck_time):
                    best = candidate
                    improved = True
                    break
            if improved:
                break
    return best


def initial_tsp_solution(instance: Instance) -> Solution:
    route = nearest_neighbour_route(instance)
    route = two_opt(route, instance.truck_time)
    return Solution(truck_route=route, sorties=[])


def index_map(route: List[Node]) -> Dict[Node, int]:
    return {node: idx for idx, node in enumerate(route)}


def truck_segment_time(route: List[Node], start_idx: int, end_idx: int, instance: Instance) -> float:
    return sum(instance.truck_time[(route[a], route[a + 1])] for a in range(start_idx, end_idx))


def feasibility_check(solution: Solution, instance: Instance) -> bool:
    """Return True if the solution satisfies the simplified FSTSP checks."""
    route = solution.truck_route
    sorties = solution.sorties

    if len(route) < 2 or route[0] != 0 or route[-1] != 0:
        return False

    truck_customers = set(route[1:-1])
    if len(truck_customers) != len(route[1:-1]):
        return False  # no repeated truck customers

    drone_customers_list = [h for _, h, _ in sorties]
    drone_customers = set(drone_customers_list)
    if len(drone_customers) != len(drone_customers_list):
        return False  # no customer can be served by UAV twice

    if truck_customers & drone_customers:
        return False

    if truck_customers | drone_customers != set(instance.customers):
        return False

    pos = index_map(route)
    intervals: List[Tuple[int, int]] = []

    for i, h, j in sorties:
        if h not in instance.customers:
            return False
        if i not in pos or j not in pos:
            return False
        if pos[i] >= pos[j]:
            return False

        truck_part = truck_segment_time(route, pos[i], pos[j], instance)
        drone_part = instance.drone_time[(i, h)] + instance.drone_time[(h, j)]
        active_sortie_duration = instance.SL + max(truck_part, drone_part) + instance.SR
        if active_sortie_duration > instance.Dtl + 1e-9:
            return False

        intervals.append((pos[i], pos[j]))

    # One UAV only: sortie intervals in the truck route may not overlap.
    # Touching intervals are allowed, e.g. one sortie ends at node j and the next starts at j.
    intervals.sort()
    last_end: Optional[int] = None
    for start, end in intervals:
        if last_end is not None and start < last_end:
            return False
        last_end = end

    return True


def evaluate_solution(solution: Solution, instance: Instance) -> float:
    """Return Z(S): total completion time of the truck-UAV route in seconds."""
    if not feasibility_check(solution, instance):
        return float("inf")

    route = solution.truck_route
    pos = index_map(route)
    sorties_by_start: Dict[int, Sortie] = {}
    for sortie in solution.sorties:
        i, _, _ = sortie
        start_idx = pos[i]
        if start_idx in sorties_by_start:
            return float("inf")  # should already be infeasible, but safe
        sorties_by_start[start_idx] = sortie

    Z = 0.0
    ell = 0
    while ell < len(route) - 1:
        if ell in sorties_by_start:
            i, h, j = sorties_by_start[ell]
            k = pos[j]
            truck_part = truck_segment_time(route, ell, k, instance)
            drone_part = instance.drone_time[(i, h)] + instance.drone_time[(h, j)]
            Z += instance.SL + max(truck_part, drone_part) + instance.SR
            ell = k
        else:
            Z += instance.truck_time[(route[ell], route[ell + 1])]
            ell += 1
    return Z


def launch_or_rendezvous_nodes(solution: Solution) -> set[Node]:
    nodes = set()
    for i, _, j in solution.sorties:
        nodes.add(i)
        nodes.add(j)
    return nodes


def random_insertion_index(route: List[Node], rng: random.Random) -> int:
    """Return an index at which a customer can be inserted before the final depot."""
    return rng.randint(1, len(route) - 1)


def truck_to_truck(solution: Solution, rng: random.Random) -> None:
    protected = launch_or_rendezvous_nodes(solution)
    candidates = [h for h in solution.truck_route[1:-1] if h not in protected]
    if not candidates:
        return
    h = rng.choice(candidates)
    route = solution.truck_route
    old_idx = route.index(h)
    route.pop(old_idx)
    new_idx = random_insertion_index(route, rng)
    route.insert(new_idx, h)


def feasible_launch_pairs_after_removal(
    route: List[Node], h: Node, instance: Instance
) -> List[Tuple[Node, Node]]:
    """Return launch/rendezvous pairs that satisfy the single-sortie duration check.

    This does not check overlap with existing sorties; the full FeasibilityCheck
    is still called by the SA main loop. It only avoids generating drone sorties
    that are immediately impossible because of the endurance bound.
    """
    pairs: List[Tuple[Node, Node]] = []
    for a in range(0, len(route) - 1):
        for b in range(a + 1, len(route)):
            i, j = route[a], route[b]
            truck_part = truck_segment_time(route, a, b, instance)
            drone_part = instance.drone_time[(i, h)] + instance.drone_time[(h, j)]
            duration = instance.SL + max(truck_part, drone_part) + instance.SR
            if duration <= instance.Dtl + 1e-9:
                pairs.append((i, j))
    return pairs


def truck_to_uav(solution: Solution, rng: random.Random, instance: Instance) -> None:
    protected = launch_or_rendezvous_nodes(solution)
    candidates = [h for h in solution.truck_route[1:-1] if h not in protected]
    if not candidates:
        return

    h = rng.choice(candidates)
    route = solution.truck_route
    old_idx = route.index(h)
    route.pop(old_idx)

    pairs = feasible_launch_pairs_after_removal(route, h, instance)
    if not pairs:
        # No feasible sortie for this customer in the current route, so undo the move.
        route.insert(old_idx, h)
        return

    i, j = rng.choice(pairs)
    solution.sorties.append((i, h, j))


def uav_to_truck(solution: Solution, rng: random.Random) -> None:
    if not solution.sorties:
        return
    sortie = rng.choice(solution.sorties)
    _, h, _ = sortie
    solution.sorties.remove(sortie)
    idx = random_insertion_index(solution.truck_route, rng)
    solution.truck_route.insert(idx, h)


def change_sortie(solution: Solution, rng: random.Random, instance: Instance) -> None:
    if not solution.sorties:
        return
    old_sortie = rng.choice(solution.sorties)
    _, h, _ = old_sortie
    solution.sorties.remove(old_sortie)

    route = solution.truck_route
    pairs = feasible_launch_pairs_after_removal(route, h, instance)
    if not pairs:
        solution.sorties.append(old_sortie)
        return

    i, j = rng.choice(pairs)
    solution.sorties.append((i, h, j))


def apply_random_move(solution: Solution, rng: random.Random, instance: Instance) -> str:
    # If no UAV sortie exists yet, only the two relevant moves are sampled.
    # Once sorties exist, all four move operators can be selected.
    if solution.sorties:
        move = rng.choice(["truck_to_truck", "truck_to_uav", "uav_to_truck", "change_sortie"])
    else:
        move = rng.choice(["truck_to_truck", "truck_to_uav"])

    if move == "truck_to_truck":
        truck_to_truck(solution, rng)
    elif move == "truck_to_uav":
        truck_to_uav(solution, rng, instance)
    elif move == "uav_to_truck":
        uav_to_truck(solution, rng)
    elif move == "change_sortie":
        change_sortie(solution, rng, instance)
    return move


def format_seconds(seconds: float) -> str:
    return f"{seconds:.2f} sec ({seconds / 60:.2f} min)"


def simulated_annealing(
    instance: Instance,
    initial_solution: Solution,
    N: int = 100_000,
    tau_s: float = 0.01,
    tau_f: float = 0.001,
    seed: int = 99,
    updates: int = 50,
    verbose: bool = True,
) -> SAResult:
    if N <= 0:
        raise ValueError("N must be positive.")
    if tau_s <= 0 or tau_f <= 0:
        raise ValueError("tau_s and tau_f must be positive.")
    if tau_f >= tau_s:
        raise ValueError("tau_f should be smaller than tau_s for cooling.")

    rng = random.Random(seed)
    start_time = time.perf_counter()

    S_cur = copy.deepcopy(initial_solution)
    if not feasibility_check(S_cur, instance):
        raise ValueError("Initial solution is infeasible.")

    Z_cur = evaluate_solution(S_cur, instance)
    S_best = copy.deepcopy(S_cur)
    Z_best = Z_cur

    T_s = tau_s * Z_cur
    T_f = tau_f * Z_cur
    alpha = math.exp(math.log(T_f / T_s) / N)
    T = T_s

    update_iters = {max(1, round(k * N / updates)) for k in range(1, updates + 1)}

    accepted = 0
    improved = 0
    infeasible = 0

    if verbose:
        print("\n--- Simplified FSTSP Simulated Annealing ---")
        print(f"Initial objective: {format_seconds(Z_cur)}")
        print(f"N={N:,}, tau_s={tau_s}, tau_f={tau_f}, T_s={T_s:.4f}, T_f={T_f:.4f}, alpha={alpha:.8f}")
        print(f"Progress updates: {len(update_iters)}\n")

    for q in range(1, N + 1):
        S_new = copy.deepcopy(S_cur)
        move = apply_random_move(S_new, rng, instance)

        if feasibility_check(S_new, instance):
            Z_new = evaluate_solution(S_new, instance)
            delta = Z_new - Z_cur

            accept = False
            if delta <= 0:
                accept = True
                improved += 1
            else:
                u = rng.random()
                if u <= math.exp(-delta / T):
                    accept = True

            if accept:
                S_cur = S_new
                Z_cur = Z_new
                accepted += 1

                if Z_cur < Z_best:
                    S_best = copy.deepcopy(S_cur)
                    Z_best = Z_cur
        else:
            infeasible += 1

        T *= alpha

        if verbose and q in update_iters:
            print(
                f"Update {len([x for x in update_iters if x <= q]):02d}/{len(update_iters)} | "
                f"iter={q:>8,} | T={T:>10.4f} | "
                f"current={Z_cur:>10.2f} | best={Z_best:>10.2f} | "
                f"sorties={len(S_best.sorties):>2} | last_move={move}"
            )

    runtime = time.perf_counter() - start_time
    if verbose:
        print("\n--- Final solution ---")
        print_solution(S_best, instance)
        print(f"Best objective: {format_seconds(Z_best)}")
        print(f"Runtime: {runtime:.2f} seconds")
        print(f"Accepted moves: {accepted:,}")
        print(f"Improving accepted moves: {improved:,}")
        print(f"Rejected infeasible moves: {infeasible:,}")

    return SAResult(
        best_solution=S_best,
        best_objective=Z_best,
        final_current_objective=Z_cur,
        accepted_moves=accepted,
        improved_moves=improved,
        infeasible_moves=infeasible,
        total_iterations=N,
        runtime_seconds=runtime,
    )


def print_instance(instance: Instance) -> None:
    print("\nCoordinates:")
    print("node, x_km, y_km")
    for node in sorted(instance.coords):
        x, y = instance.coords[node]
        label = "depot" if node == 0 else f"customer {node}"
        print(f"{node:>3} ({label:>10}): {x:>8.3f}, {y:>8.3f}")


def print_solution(solution: Solution, instance: Instance) -> None:
    print(f"Truck route: {solution.truck_route}")
    if solution.sorties:
        print("UAV sorties:")
        for i, h, j in solution.sorties:
            flight = instance.drone_time[(i, h)] + instance.drone_time[(h, j)]
            print(f"  launch {i:>2} -> serve {h:>2} -> rendezvous {j:>2} | flight={flight:.2f} sec")
    else:
        print("UAV sorties: none")


def main() -> None:
    parser = argparse.ArgumentParser(description="Simplified SA for the FSTSP.")
    parser.add_argument("--customers", type=int, default=20, help="number of customers")
    parser.add_argument("--area", type=float, default=30.0, help="side length of square area in km")
    parser.add_argument("--depot", choices=["center", "corner"], default="center", help="depot placement")
    parser.add_argument("--instance-seed", type=int, default=1, help="seed for coordinate generation")
    parser.add_argument("--sa-seed", type=int, default=99, help="seed for simulated annealing")
    parser.add_argument("--N", type=int, default=100000, help="number of SA iterations")
    parser.add_argument("--tau-s", type=float, default=0.01, help="relative starting temperature")
    parser.add_argument("--tau-f", type=float, default=0.001, help="relative final temperature")
    parser.add_argument("--SL", type=float, default=60.0, help="launch service time in seconds")
    parser.add_argument("--SR", type=float, default=60.0, help="recovery service time in seconds")
    parser.add_argument("--truck-sec-km", type=float, default=156.0, help="truck seconds per Euclidean km")
    parser.add_argument("--drone-sec-km", type=float, default=78.0, help="drone seconds per Euclidean km")
    parser.add_argument("--endurance-km", type=float, default=15.0, help="drone endurance in km equivalent")
    parser.add_argument("--hide-coordinates", action="store_true", help="do not print generated coordinates")
    args = parser.parse_args()

    instance = generate_instance(
        n_customers=args.customers,
        area_side_km=args.area,
        depot_position=args.depot,
        seed=args.instance_seed,
        truck_seconds_per_km=args.truck_sec_km,
        drone_seconds_per_km=args.drone_sec_km,
        SL=args.SL,
        SR=args.SR,
        drone_endurance_km=args.endurance_km,
    )

    if not args.hide_coordinates:
        print_instance(instance)

    S0 = initial_tsp_solution(instance)
    print("\nInitial truck-only solution:")
    print_solution(S0, instance)
    print(f"Initial TSP objective: {format_seconds(evaluate_solution(S0, instance))}")

    simulated_annealing(
        instance=instance,
        initial_solution=S0,
        N=args.N,
        tau_s=args.tau_s,
        tau_f=args.tau_f,
        seed=args.sa_seed,
        updates=50,
        verbose=True,
    )


if __name__ == "__main__":
    main()
