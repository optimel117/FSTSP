"""
FSTSP heuristic solver based on the route-and-reassign heuristic described in
Rafael Hamelink's thesis draft, Chapter 4.

The code is intentionally written as a thesis-friendly implementation:
- no external packages are required;
- it prints step-by-step feedback similar to an MILP solution log;
- it can solve the initial truck-only TSP exactly for moderate instances;
- it then applies a Murray-and-Chu-style improvement heuristic.

Main assumptions implemented here
---------------------------------
1. One truck and one drone/UAV.
2. A drone sortie has the form: launch node i -> drone customer h -> rendezvous node j.
3. Each drone sortie serves exactly one customer.
4. Launch and recovery service times are included.
5. The drone endurance check is:
       SL + drone_time(i,h) + drone_time(h,j) + SR <= endurance
6. Drone sorties may not overlap in time along the truck route, but two consecutive
   sorties may share an endpoint. For example, 3 -> 6 -> 5 and 5 -> 2 -> 4 is allowed.
7. By default, the depot is not used as a launch or rendezvous node for UAV sorties,
   so the built-in example follows the thesis example more closely. You can allow
   depot endpoints by setting allow_depot_launch=True and/or allow_depot_rendezvous=True.
8. After assigning h to the drone between i and j, the nodes i, h, and j are removed
   from the eligible set C', matching the update rule in the thesis draft.

How to use
----------
Run this file directly to test the six-customer example from the thesis:

    python fstsp_heuristic_solver.py

Or import the solver in another file:

    from fstsp_heuristic_solver import FSTSPHeuristicSolver

    solver = FSTSPHeuristicSolver(truck_times, drone_times, depot=0, endurance=5, SL=0.2, SR=0.2)
    solution = solver.solve(verbose=True)

Input format
------------
truck_times and drone_times may be either:
- a dictionary with keys (i, j), e.g. {(0, 1): 4.4, (1, 0): 4.4, ...}, or
- a list-of-lists matrix, where matrix[i][j] is the travel time from i to j.

Node labels may be integers or strings. In the built-in example, depot D is represented as 0.

You can also build truck_times and drone_times automatically from coordinates by using
build_travel_times_from_coordinates(...), so you only need to define customer locations,
speeds, endurance, launch time, recovery time, and optional customer service times.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from pathlib import Path
from functools import lru_cache
from itertools import combinations, permutations
from math import inf, hypot
from time import perf_counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

Node = Any
ArcTimes = Union[Dict[Tuple[Node, Node], float], List[List[float]], Tuple[Tuple[float, ...], ...]]
Coordinates = Dict[Node, Tuple[float, float]]


@dataclass(frozen=True)
class Sortie:
    """A single drone sortie: launch -> drone_customer -> rendezvous."""

    launch: Node
    drone_customer: Node
    rendezvous: Node

    def as_tuple(self) -> Tuple[Node, Node, Node]:
        return (self.launch, self.drone_customer, self.rendezvous)


@dataclass
class Move:
    """Best move found in one heuristic iteration."""

    move_type: str  # "uav" or "truck"
    h: Node
    launch_or_prev: Node
    rendezvous_or_next: Node
    improvement: float
    old_completion: float
    new_completion: float
    removal_saving: float
    insertion_or_uav_cost: float
    new_route: List[Node]
    new_sorties: List[Sortie]
    explanation: str
    tie_metric: float = 0.0  # smaller is preferred when improvements are equal


@dataclass
class Solution:
    truck_route: List[Node]
    sorties: List[Sortie]
    completion_time: float
    truck_only_completion_time: float
    improvement_abs: float
    improvement_pct: float
    eligible_remaining: List[Node]
    iterations: int
    runtime_seconds: float
    log: List[str]
    benchmark_completion_time: Optional[float] = None
    heuristic_gap_pct: Optional[float] = None


class FSTSPHeuristicSolver:
    def __init__(
        self,
        truck_times: ArcTimes,
        drone_times: Optional[ArcTimes] = None,
        *,
        depot: Node = 0,
        customers: Optional[Iterable[Node]] = None,
        drone_eligible: Optional[Iterable[Node]] = None,
        endurance: float = 5.0,
        SL: float = 0.0,
        SR: float = 0.0,
        tsp_method: str = "auto",
        exact_tsp_limit: int = 18,
        allow_truck_reinsertion_into_uav_subroutes: bool = True,
        allow_depot_launch: bool = False,
        allow_depot_rendezvous: bool = False,
        truck_service_times: Optional[Dict[Node, float]] = None,
        drone_service_times: Optional[Dict[Node, float]] = None,
        tolerance: float = 1e-9,
    ) -> None:
        self.truck_times = truck_times
        self.drone_times = drone_times if drone_times is not None else truck_times
        self.depot = depot
        self.endurance = float(endurance)
        self.SL = float(SL)
        self.SR = float(SR)
        self.tsp_method = tsp_method
        self.exact_tsp_limit = exact_tsp_limit
        self.allow_truck_reinsertion_into_uav_subroutes = allow_truck_reinsertion_into_uav_subroutes
        self.allow_depot_launch = allow_depot_launch
        self.allow_depot_rendezvous = allow_depot_rendezvous
        self.truck_service_times = truck_service_times or {}
        self.drone_service_times = drone_service_times or {}
        self.tol = tolerance

        inferred_nodes = self._infer_nodes(truck_times)
        if customers is None:
            self.customers = [node for node in inferred_nodes if node != depot]
        else:
            self.customers = list(customers)

        if drone_eligible is None:
            self.initial_eligible = set(self.customers)
        else:
            self.initial_eligible = set(drone_eligible)

    # ------------------------------------------------------------------
    # Basic travel-time helpers
    # ------------------------------------------------------------------
    def _infer_nodes(self, times: ArcTimes) -> List[Node]:
        if isinstance(times, dict):
            nodes = set()
            for i, j in times:
                nodes.add(i)
                nodes.add(j)
            return sorted(nodes, key=lambda x: str(x))
        return list(range(len(times)))

    def truck_time(self, i: Node, j: Node) -> float:
        return self._time(self.truck_times, i, j)

    def drone_time(self, i: Node, j: Node) -> float:
        return self._time(self.drone_times, i, j)

    def truck_service_time(self, node: Node) -> float:
        """Service time if the truck serves this node. The depot defaults to zero."""
        if node == self.depot:
            return 0.0
        return float(self.truck_service_times.get(node, 0.0))

    def drone_service_time(self, node: Node) -> float:
        """Service time if the UAV serves this node. The depot defaults to zero."""
        if node == self.depot:
            return 0.0
        return float(self.drone_service_times.get(node, 0.0))

    def truck_leg_time(self, i: Node, j: Node) -> float:
        """Truck time for moving from i to j plus the service time at j, if j is a customer."""
        return self.truck_time(i, j) + self.truck_service_time(j)

    def drone_sortie_time_without_SL_SR(self, i: Node, h: Node, j: Node) -> float:
        """UAV flight plus service at the UAV-served customer, excluding SL and SR."""
        return self.drone_time(i, h) + self.drone_service_time(h) + self.drone_time(h, j)

    @staticmethod
    def _time(times: ArcTimes, i: Node, j: Node) -> float:
        if isinstance(times, dict):
            return float(times[(i, j)])
        return float(times[i][j])

    # ------------------------------------------------------------------
    # Initial TSP construction
    # ------------------------------------------------------------------
    def solve_initial_tsp(self) -> Tuple[List[Node], float, str]:
        n = len(self.customers)
        method = self.tsp_method.lower()
        if method == "auto":
            method = "exact" if n <= self.exact_tsp_limit else "nearest_2opt"

        if method == "given":
            raise ValueError("For a given route, call solve(initial_route=[...]) instead.")
        if method == "exact":
            route, cost = self._held_karp_tsp()
            return route, cost, "exact Held-Karp dynamic programming"
        if method in {"nearest", "nearest_2opt", "heuristic"}:
            route = self._nearest_neighbor_route()
            if method in {"nearest_2opt", "heuristic"}:
                route = self._two_opt(route)
            return route, self.route_travel_time(route), "nearest-neighbor + 2-opt"
        if method == "bruteforce":
            route, cost = self._bruteforce_tsp()
            return route, cost, "brute force enumeration"

        raise ValueError(f"Unknown tsp_method: {self.tsp_method}")

    def _bruteforce_tsp(self) -> Tuple[List[Node], float]:
        best_route: Optional[List[Node]] = None
        best_cost = inf
        for perm in permutations(self.customers):
            route = [self.depot, *perm, self.depot]
            cost = self.route_travel_time(route)
            if cost < best_cost:
                best_cost = cost
                best_route = route
        assert best_route is not None
        return best_route, best_cost

    def _held_karp_tsp(self) -> Tuple[List[Node], float]:
        """Exact TSP by dynamic programming. Suitable for moderate n."""
        customers = tuple(self.customers)
        n = len(customers)
        idx_to_node = {i: customers[i] for i in range(n)}
        node_to_idx = {customers[i]: i for i in range(n)}

        # dp[(mask, last)] = (cost, previous_last)
        dp: Dict[Tuple[int, int], Tuple[float, Optional[int]]] = {}
        for k, node in idx_to_node.items():
            mask = 1 << k
            dp[(mask, k)] = (self.truck_leg_time(self.depot, node), None)

        for size in range(2, n + 1):
            for subset in combinations(range(n), size):
                mask = sum(1 << k for k in subset)
                for last in subset:
                    prev_mask = mask ^ (1 << last)
                    last_node = idx_to_node[last]
                    best_cost = inf
                    best_prev = None
                    for prev in subset:
                        if prev == last:
                            continue
                        prev_cost, _ = dp[(prev_mask, prev)]
                        cost = prev_cost + self.truck_leg_time(idx_to_node[prev], last_node)
                        if cost < best_cost:
                            best_cost = cost
                            best_prev = prev
                    dp[(mask, last)] = (best_cost, best_prev)

        full_mask = (1 << n) - 1
        best_total = inf
        best_last = None
        for last in range(n):
            path_cost, _ = dp[(full_mask, last)]
            total = path_cost + self.truck_leg_time(idx_to_node[last], self.depot)
            if total < best_total:
                best_total = total
                best_last = last

        # reconstruct route
        assert best_last is not None
        mask = full_mask
        last = best_last
        reverse_nodes = []
        while last is not None:
            reverse_nodes.append(idx_to_node[last])
            _, prev = dp[(mask, last)]
            mask ^= 1 << last
            last = prev
        route = [self.depot, *reversed(reverse_nodes), self.depot]
        return route, best_total

    def _nearest_neighbor_route(self) -> List[Node]:
        unvisited = set(self.customers)
        route = [self.depot]
        current = self.depot
        while unvisited:
            nxt = min(unvisited, key=lambda j: self.truck_leg_time(current, j))
            route.append(nxt)
            unvisited.remove(nxt)
            current = nxt
        route.append(self.depot)
        return route

    def _two_opt(self, route: List[Node]) -> List[Node]:
        best = route[:]
        best_cost = self.route_travel_time(best)
        improved = True
        while improved:
            improved = False
            for i in range(1, len(best) - 2):
                for k in range(i + 1, len(best) - 1):
                    new_route = best[:i] + list(reversed(best[i:k + 1])) + best[k + 1:]
                    new_cost = self.route_travel_time(new_route)
                    if new_cost + self.tol < best_cost:
                        best, best_cost = new_route, new_cost
                        improved = True
                        break
                if improved:
                    break
        return best

    # ------------------------------------------------------------------
    # Objective / schedule evaluation
    # ------------------------------------------------------------------
    def route_travel_time(self, route: Sequence[Node]) -> float:
        return sum(self.truck_leg_time(route[k], route[k + 1]) for k in range(len(route) - 1))

    def completion_time(self, route: Sequence[Node], sorties: Sequence[Sortie]) -> float:
        completion, _events = self.compute_schedule(route, sorties)
        return completion

    def compute_schedule(self, route: Sequence[Node], sorties: Sequence[Sortie]) -> Tuple[float, List[Dict[str, Any]]]:
        """
        Computes truck/drone timing along the current truck route.

        The truck may need to wait at a rendezvous if the drone has not returned yet.
        If the drone arrives earlier, the drone waits and the truck does not lose time.
        """
        launch_map: Dict[Node, Sortie] = {s.launch: s for s in sorties}
        rendezvous_map: Dict[Node, List[Sortie]] = {}
        for s in sorties:
            rendezvous_map.setdefault(s.rendezvous, []).append(s)

        # Validate that every sortie endpoint appears in the truck route.
        route_set = set(route)
        for s in sorties:
            if s.launch not in route_set or s.rendezvous not in route_set:
                raise ValueError(f"Sortie endpoint missing from truck route: {s}")

        drone_ready: Dict[Tuple[Node, Node, Node], float] = {}
        t = 0.0
        events: List[Dict[str, Any]] = []
        events.append({"node": route[0], "truck_arrival": t, "event": "start"})

        for pos in range(len(route) - 1):
            node = route[pos]

            if node in launch_map:
                s = launch_map[node]
                t_launch_start = t
                t += self.SL
                ready = t + self.drone_sortie_time_without_SL_SR(s.launch, s.drone_customer, s.rendezvous)
                drone_ready[s.as_tuple()] = ready
                events.append({
                    "node": node,
                    "event": "launch",
                    "sortie": s.as_tuple(),
                    "truck_time_before_SL": t_launch_start,
                    "truck_time_after_SL": t,
                    "drone_ready_at_rendezvous": ready,
                })

            nxt = route[pos + 1]
            t += self.truck_time(node, nxt)
            events.append({"node": nxt, "truck_arrival": t, "event": "truck_arrival"})

            service = self.truck_service_time(nxt)
            if service > self.tol:
                before_service = t
                t += service
                events.append({
                    "node": nxt,
                    "event": "truck_service",
                    "truck_time_before_service": before_service,
                    "service_time": service,
                    "truck_time_after_service": t,
                })

            if nxt in rendezvous_map:
                for s in rendezvous_map[nxt]:
                    ready = drone_ready.get(s.as_tuple())
                    if ready is None:
                        raise ValueError(f"Drone rendezvous occurs before launch for sortie {s}")
                    truck_before_wait = t
                    wait = max(0.0, ready - t)
                    t = max(t, ready) + self.SR
                    events.append({
                        "node": nxt,
                        "event": "rendezvous",
                        "sortie": s.as_tuple(),
                        "truck_arrival_before_wait": truck_before_wait,
                        "drone_ready_at_rendezvous": ready,
                        "truck_wait": wait,
                        "truck_time_after_SR": t,
                    })

        return t, events

    def arrival_times_without_service(self, route: Sequence[Node]) -> Dict[Tuple[int, Node], float]:
        """Truck arrival times along route ignoring launch/recovery service and waiting."""
        t = 0.0
        out: Dict[Tuple[int, Node], float] = {(0, route[0]): 0.0}
        for pos in range(len(route) - 1):
            t += self.truck_time(route[pos], route[pos + 1])
            out[(pos + 1, route[pos + 1])] = t
        return out

    # ------------------------------------------------------------------
    # Heuristic mechanics
    # ------------------------------------------------------------------
    def solve(
        self,
        *,
        initial_route: Optional[Sequence[Node]] = None,
        benchmark_completion_time: Optional[float] = None,
        verbose: bool = True,
    ) -> Solution:
        start = perf_counter()
        log: List[str] = []

        if initial_route is None:
            route, tsp_cost, tsp_method_name = self.solve_initial_tsp()
        else:
            route = list(initial_route)
            if route[0] != self.depot or route[-1] != self.depot:
                raise ValueError("initial_route must start and end at the depot.")
            tsp_cost = self.route_travel_time(route)
            tsp_method_name = "user-provided initial route"

        sorties: List[Sortie] = []
        eligible = set(self.initial_eligible)
        truck_only_completion = self.completion_time(route, sorties)

        self._log(log, verbose, "=" * 72)
        self._log(log, verbose, "FSTSP HEURISTIC SOLVER")
        self._log(log, verbose, "=" * 72)
        self._log(log, verbose, f"Initial TSP method: {tsp_method_name}")
        self._log(log, verbose, f"Initial truck route: {self.format_route(route)}")
        self._log(log, verbose, f"Initial truck-only completion time: {truck_only_completion:.4f}")
        self._log(log, verbose, f"Initial eligible set C': {self.format_node_set(eligible)}")
        self._log(log, verbose, "")

        iteration = 0
        while True:
            iteration += 1
            current_completion = self.completion_time(route, sorties)
            best = self.find_best_move(route, sorties, eligible)

            self._log(log, verbose, f"Iteration {iteration}")
            self._log(log, verbose, "-" * 72)

            if best is None or best.improvement <= self.tol:
                self._log(log, verbose, "No improving move found. The heuristic stops.")
                self._log(log, verbose, f"Current completion time: {current_completion:.4f}")
                self._log(log, verbose, "")
                iteration -= 1
                break

            self._log(log, verbose, best.explanation)
            self._log(log, verbose, f"Old completion time: {best.old_completion:.4f}")
            self._log(log, verbose, f"New completion time: {best.new_completion:.4f}")
            self._log(log, verbose, f"Improvement / MaxSaving: {best.improvement:.4f}")
            self._log(log, verbose, f"Updated truck route: {self.format_route(best.new_route)}")
            self._log(log, verbose, f"Updated UAV sorties: {self.format_sorties(best.new_sorties)}")

            route = best.new_route
            sorties = best.new_sorties
            if best.move_type == "uav":
                eligible.discard(best.launch_or_prev)
                eligible.discard(best.h)
                eligible.discard(best.rendezvous_or_next)
            # For a truck reinsertion, keep h eligible. This matches the idea that it
            # remains a truck customer and may still be considered in later iterations.

            self._log(log, verbose, f"Updated eligible set C': {self.format_node_set(eligible)}")
            self._log(log, verbose, "")

        final_completion = self.completion_time(route, sorties)
        improvement_abs = truck_only_completion - final_completion
        improvement_pct = 100.0 * improvement_abs / truck_only_completion if truck_only_completion > 0 else 0.0
        runtime = perf_counter() - start

        heuristic_gap_pct = None
        if benchmark_completion_time is not None and benchmark_completion_time > 0:
            heuristic_gap_pct = 100.0 * (final_completion - benchmark_completion_time) / benchmark_completion_time

        self._log(log, verbose, "=" * 72)
        self._log(log, verbose, "SOLUTION SUMMARY")
        self._log(log, verbose, "=" * 72)
        self._log(log, verbose, f"Final truck route: {self.format_route(route)}")
        self._log(log, verbose, f"Final UAV sorties: {self.format_sorties(sorties)}")
        self._log(log, verbose, f"Truck-only completion time: {truck_only_completion:.4f}")
        self._log(log, verbose, f"Final completion time: {final_completion:.4f}")
        self._log(log, verbose, f"Absolute improvement: {improvement_abs:.4f}")
        self._log(log, verbose, f"Percentage improvement: {improvement_pct:.2f}%")
        if benchmark_completion_time is not None:
            self._log(log, verbose, f"Benchmark/optimal completion time: {benchmark_completion_time:.4f}")
            self._log(log, verbose, f"Heuristic gap relative to benchmark: {heuristic_gap_pct:.2f}%")
        else:
            self._log(log, verbose, "Heuristic gap: not computed, because no optimal benchmark was supplied.")
        self._log(log, verbose, f"Iterations: {iteration}")
        self._log(log, verbose, f"Runtime: {runtime:.4f} seconds")
        self._log(log, verbose, "")

        return Solution(
            truck_route=route,
            sorties=sorties,
            completion_time=final_completion,
            truck_only_completion_time=truck_only_completion,
            improvement_abs=improvement_abs,
            improvement_pct=improvement_pct,
            eligible_remaining=sorted(eligible, key=lambda x: str(x)),
            iterations=iteration,
            runtime_seconds=runtime,
            log=log,
            benchmark_completion_time=benchmark_completion_time,
            heuristic_gap_pct=heuristic_gap_pct,
        )

    def find_best_move(self, route: List[Node], sorties: List[Sortie], eligible: set) -> Optional[Move]:
        best: Optional[Move] = None
        old_completion = self.completion_time(route, sorties)

        # Only customers that are currently on the truck route can be removed from the truck route.
        truck_customers = [node for node in route[1:-1] if node in eligible]

        for h in truck_customers:
            removal_route = self.remove_node_once(route, h)
            if removal_route is None:
                continue

            standard_saving = self.local_removal_saving(route, h)

            # Evaluate possible UAV assignment for h on non-UAV intervals.
            for i_pos in range(0, len(removal_route) - 1):
                for j_pos in range(i_pos + 1, len(removal_route)):
                    i = removal_route[i_pos]
                    j = removal_route[j_pos]
                    if i == j:
                        continue
                    if i == self.depot and not self.allow_depot_launch:
                        continue
                    if j == self.depot and not self.allow_depot_rendezvous:
                        continue
                    if not self.interval_is_free(removal_route, i_pos, j_pos, sorties):
                        continue
                    drone_flight = self.drone_sortie_time_without_SL_SR(i, h, j)
                    endurance_used = self.SL + drone_flight + self.SR
                    if endurance_used > self.endurance + self.tol:
                        continue
                    new_sorties = sorties + [Sortie(i, h, j)]
                    if not self.sorties_are_valid(removal_route, new_sorties):
                        continue
                    new_completion = self.completion_time(removal_route, new_sorties)
                    improvement = old_completion - new_completion
                    truck_segment = self.segment_truck_time(removal_route, i_pos, j_pos)
                    uav_cost = max(truck_segment + self.SL + self.SR, self.SL + drone_flight + self.SR) - truck_segment
                    explanation = (
                        f"Best move: assign customer {h} to UAV sortie {i} -> {h} -> {j}.\n"
                        f"Removal saving around h: {standard_saving:.4f}. "
                        f"UAV added cost on selected segment: {uav_cost:.4f}.\n"
                        f"Drone flight time: {drone_flight:.4f}; service included: {endurance_used:.4f}; "
                        f"endurance limit: {self.endurance:.4f}."
                    )
                    candidate = Move(
                        move_type="uav",
                        h=h,
                        launch_or_prev=i,
                        rendezvous_or_next=j,
                        improvement=improvement,
                        old_completion=old_completion,
                        new_completion=new_completion,
                        removal_saving=standard_saving,
                        insertion_or_uav_cost=uav_cost,
                        new_route=removal_route,
                        new_sorties=new_sorties,
                        explanation=explanation,
                        tie_metric=drone_flight,
                    )
                    best = self._better_move(best, candidate)

            # Evaluate truck reinsertion into existing UAV subroutes, matching Algorithm 3.
            if self.allow_truck_reinsertion_into_uav_subroutes:
                for s in sorties:
                    a_pos = self.position(removal_route, s.launch)
                    b_pos = self.position(removal_route, s.rendezvous)
                    if a_pos is None or b_pos is None or a_pos >= b_pos:
                        continue
                    for insert_pos in range(a_pos, b_pos):
                        prev_node = removal_route[insert_pos]
                        next_node = removal_route[insert_pos + 1]
                        # Avoid immediately undoing to the same local position.
                        new_route = removal_route[:insert_pos + 1] + [h] + removal_route[insert_pos + 1:]
                        if new_route == route:
                            continue
                        if not self.sorties_are_valid(new_route, sorties):
                            continue
                        new_completion = self.completion_time(new_route, sorties)
                        improvement = old_completion - new_completion
                        insertion_cost = self.truck_leg_time(prev_node, h) + self.truck_leg_time(h, next_node) - self.truck_leg_time(prev_node, next_node)
                        explanation = (
                            f"Best move: reinsert customer {h} on the truck between {prev_node} and {next_node}.\n"
                            f"Removal saving around h: {standard_saving:.4f}. "
                            f"Truck insertion cost: {insertion_cost:.4f}.\n"
                            f"The move is evaluated inside an existing UAV subroute, so synchronization is recomputed exactly."
                        )
                        candidate = Move(
                            move_type="truck",
                            h=h,
                            launch_or_prev=prev_node,
                            rendezvous_or_next=next_node,
                            improvement=improvement,
                            old_completion=old_completion,
                            new_completion=new_completion,
                            removal_saving=standard_saving,
                            insertion_or_uav_cost=insertion_cost,
                            new_route=new_route,
                            new_sorties=sorties[:],
                            explanation=explanation,
                            tie_metric=insertion_cost,
                        )
                        best = self._better_move(best, candidate)

        return best

    def _better_move(self, best: Optional[Move], candidate: Move) -> Optional[Move]:
        if candidate.improvement <= self.tol:
            return best
        if best is None or candidate.improvement > best.improvement + self.tol:
            return candidate
        if abs(candidate.improvement - best.improvement) <= self.tol:
            if candidate.tie_metric < best.tie_metric - self.tol:
                return candidate
        return best

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------
    def remove_node_once(self, route: List[Node], node: Node) -> Optional[List[Node]]:
        out = route[:]
        try:
            idx = out.index(node, 1, len(out) - 1)
        except ValueError:
            return None
        del out[idx]
        return out

    def local_removal_saving(self, route: List[Node], h: Node) -> float:
        try:
            pos = route.index(h, 1, len(route) - 1)
        except ValueError:
            return 0.0
        prev_node = route[pos - 1]
        next_node = route[pos + 1]
        return self.truck_leg_time(prev_node, h) + self.truck_leg_time(h, next_node) - self.truck_leg_time(prev_node, next_node)

    def position(self, route: Sequence[Node], node: Node) -> Optional[int]:
        for k, value in enumerate(route):
            if value == node:
                return k
        return None

    def segment_truck_time(self, route: Sequence[Node], start_pos: int, end_pos: int) -> float:
        return sum(self.truck_leg_time(route[k], route[k + 1]) for k in range(start_pos, end_pos))

    def interval_is_free(self, route: Sequence[Node], start_pos: int, end_pos: int, sorties: Sequence[Sortie]) -> bool:
        """Returns True if the candidate interval does not overlap existing UAV intervals."""
        for s in sorties:
            a = self.position(route, s.launch)
            b = self.position(route, s.rendezvous)
            if a is None or b is None:
                continue
            if a > b:
                return False
            # overlap with positive length is forbidden; touching at endpoint is allowed
            if max(start_pos, a) < min(end_pos, b):
                return False
        return True

    def sorties_are_valid(self, route: Sequence[Node], sorties: Sequence[Sortie]) -> bool:
        intervals: List[Tuple[int, int]] = []
        for s in sorties:
            a = self.position(route, s.launch)
            b = self.position(route, s.rendezvous)
            if a is None or b is None or a >= b:
                return False
            flight = self.SL + self.drone_sortie_time_without_SL_SR(s.launch, s.drone_customer, s.rendezvous) + self.SR
            if flight > self.endurance + self.tol:
                return False
            intervals.append((a, b))

        for m in range(len(intervals)):
            for n in range(m + 1, len(intervals)):
                a1, b1 = intervals[m]
                a2, b2 = intervals[n]
                if max(a1, a2) < min(b1, b2):
                    return False
        return True

    # ------------------------------------------------------------------
    # Formatting and logging
    # ------------------------------------------------------------------
    @staticmethod
    def _log(log: List[str], verbose: bool, text: str) -> None:
        log.append(text)
        if verbose:
            print(text)

    def format_route(self, route: Sequence[Node]) -> str:
        return " -> ".join(str(x) for x in route)

    def format_sorties(self, sorties: Sequence[Sortie]) -> str:
        if not sorties:
            return "none"
        return "; ".join(f"{s.launch} -> {s.drone_customer} -> {s.rendezvous}" for s in sorties)

    def format_node_set(self, nodes: Iterable[Node]) -> str:
        return "{" + ", ".join(str(x) for x in sorted(nodes, key=lambda y: str(y))) + "}"

    def format_schedule_event(self, event: Dict[str, Any]) -> str:
        """Converts one raw schedule event dictionary into a readable text line."""
        event_type = event.get("event")
        node = event.get("node")

        if event_type == "start":
            return f"Start at node {node}; truck time = {event['truck_arrival']:.4f}."

        if event_type == "truck_arrival":
            return f"Truck arrives at node {node}; truck time = {event['truck_arrival']:.4f}."

        if event_type == "launch":
            launch, drone_customer, rendezvous = event["sortie"]
            return (
                f"Launch UAV at node {node} for sortie {launch} -> {drone_customer} -> {rendezvous}; "
                f"truck time before SL = {event['truck_time_before_SL']:.4f}, "
                f"truck time after SL = {event['truck_time_after_SL']:.4f}, "
                f"UAV ready at rendezvous at time = {event['drone_ready_at_rendezvous']:.4f}."
            )

        if event_type == "rendezvous":
            launch, drone_customer, rendezvous = event["sortie"]
            return (
                f"Rendezvous at node {node} for sortie {launch} -> {drone_customer} -> {rendezvous}; "
                f"truck arrives before waiting at time = {event['truck_arrival_before_wait']:.4f}, "
                f"UAV ready at time = {event['drone_ready_at_rendezvous']:.4f}, "
                f"truck waiting time = {event['truck_wait']:.4f}, "
                f"truck time after SR = {event['truck_time_after_SR']:.4f}."
            )

        if event_type == "truck_service":
            return (
                f"Truck serves node {node}; service time = {event['service_time']:.4f}, "
                f"truck time before service = {event['truck_time_before_service']:.4f}, "
                f"truck time after service = {event['truck_time_after_service']:.4f}."
            )

        return f"Event at node {node}: {event}"

    def format_schedule(self, route: Sequence[Node], sorties: Sequence[Sortie]) -> List[str]:
        """Returns the full final schedule as readable text lines."""
        _completion, events = self.compute_schedule(route, sorties)
        return [self.format_schedule_event(event) for event in events]

    def print_schedule(self, route: Sequence[Node], sorties: Sequence[Sortie]) -> None:
        """Prints the full final schedule as readable text lines."""
        for line in self.format_schedule(route, sorties):
            print(line)




# ----------------------------------------------------------------------
# Experiment/result storage helpers
# ----------------------------------------------------------------------
def solution_to_record(
    solution: Solution,
    *,
    instance_name: str,
    n_customers: int,
    drone_speed_factor: float,
    endurance: float,
    SL: float,
    SR: float,
    tsp_method: str,
    seed: Optional[int] = None,
    notes: str = "",
) -> Dict[str, Any]:
    """
    Converts one solution into a flat dictionary.

    This is the most convenient format for comparing many experiments in Excel,
    Python, or Overleaf tables.
    """
    return {
        "instance_name": instance_name,
        "seed": seed,
        "n_customers": n_customers,
        "drone_speed_factor": drone_speed_factor,
        "endurance": endurance,
        "SL": SL,
        "SR": SR,
        "tsp_method": tsp_method,
        "truck_only_completion_time": solution.truck_only_completion_time,
        "final_completion_time": solution.completion_time,
        "absolute_improvement": solution.improvement_abs,
        "percentage_improvement": solution.improvement_pct,
        "benchmark_completion_time": solution.benchmark_completion_time,
        "heuristic_gap_pct": solution.heuristic_gap_pct,
        "iterations": solution.iterations,
        "runtime_seconds": solution.runtime_seconds,
        "final_truck_route": " -> ".join(str(x) for x in solution.truck_route),
        "final_uav_sorties": "; ".join(
            f"{s.launch} -> {s.drone_customer} -> {s.rendezvous}" for s in solution.sorties
        ),
        "eligible_remaining": ", ".join(str(x) for x in solution.eligible_remaining),
        "notes": notes,
    }


def append_record_to_csv(filepath: Union[str, Path], record: Dict[str, Any]) -> None:
    """Appends one experiment record to a CSV file, creating the header if needed."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    file_exists = filepath.exists()

    with filepath.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(record.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)


def save_solution_json(filepath: Union[str, Path], solution: Solution, record: Dict[str, Any]) -> None:
    """
    Saves the full solution in JSON format.

    Use CSV for your result table, and JSON when you want to inspect the full route,
    sorties, and iteration log later.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "record": record,
        "truck_route": solution.truck_route,
        "sorties": [s.as_tuple() for s in solution.sorties],
        "log": solution.log,
    }
    with filepath.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ----------------------------------------------------------------------
# Coordinate-based instance builders
# ----------------------------------------------------------------------
def build_travel_times_from_coordinates(
    coordinates: Coordinates,
    *,
    truck_speed: float = 1.0,
    drone_speed: float = 2.0,
    distance_scale: float = 1.0,
    round_decimals: Optional[int] = None,
) -> Tuple[Dict[Tuple[Node, Node], float], Dict[Tuple[Node, Node], float]]:
    """
    Builds truck and drone travel-time dictionaries from node coordinates.

    Parameters
    ----------
    coordinates:
        Dictionary of node -> (x, y). Include the depot as well as all customers.
        Example: {0: (0, 0), 1: (2.1, 3.0), 2: (5.0, 1.2)}.
    truck_speed:
        Truck speed in distance units per time unit.
    drone_speed:
        Drone speed in distance units per time unit. If the drone is twice as fast
        as the truck, use truck_speed=1 and drone_speed=2.
    distance_scale:
        Optional multiplier applied to all Euclidean distances.
    round_decimals:
        If not None, travel times are rounded to this number of decimals.

    Returns
    -------
    truck_times, drone_times:
        Two dictionaries with keys (i, j), ready to pass into FSTSPHeuristicSolver.
    """
    if truck_speed <= 0 or drone_speed <= 0:
        raise ValueError("truck_speed and drone_speed must be strictly positive.")

    nodes = list(coordinates.keys())
    truck: Dict[Tuple[Node, Node], float] = {}
    drone: Dict[Tuple[Node, Node], float] = {}

    for i in nodes:
        xi, yi = coordinates[i]
        for j in nodes:
            if i == j:
                continue
            xj, yj = coordinates[j]
            distance = distance_scale * hypot(float(xi) - float(xj), float(yi) - float(yj))
            truck_time = distance / truck_speed
            drone_time = distance / drone_speed
            if round_decimals is not None:
                truck_time = round(truck_time, round_decimals)
                drone_time = round(drone_time, round_decimals)
            truck[(i, j)] = truck_time
            drone[(i, j)] = drone_time

    return truck, drone


def load_coordinate_instance_from_csv(
    filepath: Union[str, Path],
    *,
    node_col: str = "node",
    x_col: str = "x",
    y_col: str = "y",
    truck_service_col: str = "truck_service_time",
    drone_service_col: str = "drone_service_time",
    drone_eligible_col: str = "drone_eligible",
) -> Tuple[Coordinates, Dict[Node, float], Dict[Node, float], Optional[List[Node]]]:
    """
    Loads coordinates and optional service-time data from a CSV file.

    Expected minimum columns:
        node,x,y

    Optional columns:
        truck_service_time,drone_service_time,drone_eligible

    Example CSV:
        node,x,y,truck_service_time,drone_service_time,drone_eligible
        0,0,0,0,0,0
        1,4.2,1.0,0.1,0.1,1
        2,2.3,5.1,0.1,0.1,1
    """
    filepath = Path(filepath)
    coordinates: Coordinates = {}
    truck_service_times: Dict[Node, float] = {}
    drone_service_times: Dict[Node, float] = {}
    eligible: List[Node] = []
    saw_eligible_col = False

    def parse_node(value: str) -> Node:
        value = value.strip()
        try:
            return int(value)
        except ValueError:
            return value

    with filepath.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {node_col, x_col, y_col}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV file is missing required columns: {sorted(missing)}")

        for row in reader:
            node = parse_node(row[node_col])
            coordinates[node] = (float(row[x_col]), float(row[y_col]))

            if truck_service_col in row and row[truck_service_col] not in {None, ""}:
                truck_service_times[node] = float(row[truck_service_col])
            if drone_service_col in row and row[drone_service_col] not in {None, ""}:
                drone_service_times[node] = float(row[drone_service_col])
            if drone_eligible_col in row and row[drone_eligible_col] not in {None, ""}:
                saw_eligible_col = True
                value = row[drone_eligible_col].strip().lower()
                if value in {"1", "true", "yes", "y"}:
                    eligible.append(node)

    return coordinates, truck_service_times, drone_service_times, eligible if saw_eligible_col else None


def make_random_coordinate_instance(
    n_customers: int,
    *,
    seed: int = 1,
    depot: Node = 0,
    width: float = 10.0,
    height: float = 10.0,
) -> Coordinates:
    """
    Creates a simple random coordinate instance for experimentation.
    The depot is placed in the center of the rectangle.
    """
    import random

    rng = random.Random(seed)
    coordinates: Coordinates = {depot: (width / 2.0, height / 2.0)}
    for customer in range(1, n_customers + 1):
        coordinates[customer] = (rng.uniform(0, width), rng.uniform(0, height))
    return coordinates

# ----------------------------------------------------------------------
# Built-in thesis example
# ----------------------------------------------------------------------
def build_thesis_example() -> Tuple[Dict[Tuple[int, int], float], Dict[Tuple[int, int], float]]:
    """
    Builds the 6-customer distance matrix from the thesis example.

    Node labels:
        0 = depot D
        1, ..., 6 = customers
    """
    labels = [0, 1, 2, 3, 4, 5, 6]
    matrix = {
        0: {0: 0.0, 1: 4.4, 2: 4.9, 3: 3.4, 4: 3.0, 5: 5.8, 6: 5.5},
        1: {0: 4.4, 1: 0.0, 2: 2.0, 3: 3.0, 4: 1.5, 5: 1.5, 6: 2.5},
        2: {0: 4.9, 1: 2.0, 2: 0.0, 3: 4.9, 4: 2.3, 5: 2.8, 6: 4.5},
        3: {0: 3.4, 1: 3.0, 2: 4.9, 3: 0.0, 4: 2.6, 5: 3.9, 6: 2.4},
        4: {0: 3.0, 1: 1.5, 2: 2.3, 3: 2.6, 4: 0.0, 5: 3.0, 6: 3.4},
        5: {0: 5.8, 1: 1.5, 2: 2.8, 3: 3.9, 4: 3.0, 5: 0.0, 6: 2.3},
        6: {0: 5.5, 1: 2.5, 2: 4.5, 3: 2.4, 4: 3.4, 5: 2.3, 6: 0.0},
    }
    truck = {(i, j): matrix[i][j] for i in labels for j in labels if i != j}
    drone = {(i, j): 0.5 * matrix[i][j] for i in labels for j in labels if i != j}
    return truck, drone


def main() -> None:
    truck, drone = build_thesis_example()

    solver = FSTSPHeuristicSolver(
        truck,
        drone,
        depot=0,
        customers=[1, 2, 3, 4, 5, 6],
        endurance=5.0,
        SL=0.2,
        SR=0.2,
        tsp_method="exact",
    )

    # To force the exact route used in the thesis example, uncomment this line:
    # initial_route = [0, 3, 6, 5, 1, 2, 4, 0]
    initial_route = None

    solution = solver.solve(initial_route=initial_route, verbose=True)

    # Optional: inspect detailed event timing of the final schedule.
    # This now prints readable text lines instead of raw tuple/dictionary objects.
    print("Detailed final schedule events:")
    solver.print_schedule(solution.truck_route, solution.sorties)

    # Optional: store this run so that different customer sizes/settings can be compared later.
    record = solution_to_record(
        solution,
        instance_name="thesis_example_6_customers",
        n_customers=6,
        drone_speed_factor=2.0,
        endurance=5.0,
        SL=0.2,
        SR=0.2,
        tsp_method="exact",
        seed=None,
        notes="Built-in thesis example",
    )
    append_record_to_csv("results/fstsp_experiment_results.csv", record)
    save_solution_json("results/thesis_example_6_customers_solution.json", solution, record)


if __name__ == "__main__":
    main()
