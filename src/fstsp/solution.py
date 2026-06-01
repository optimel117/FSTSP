from __future__ import annotations

from dataclasses import dataclass, field
from itertools import pairwise

from fstsp.instance import Instance


@dataclass(frozen=True)
class Sortie:
    """A single drone trip: launch -> customer -> rendezvous, all on the truck route."""

    launch: int
    customer: int
    rendezvous: int


@dataclass
class Subroute:
    """A maximal segment of the truck route between sortie endpoints (or route ends).

    `positions` are indices into `Solution.truck_route`. The endpoints (positions[0]
    and positions[-1]) are shared with adjacent subroutes when sorties exist.
    """

    positions: list[int]
    sortie: Sortie | None  # the sortie running over this subroute, if any

    def nodes(self, route: list[int]) -> list[int]:
        return [route[p] for p in self.positions]


@dataclass
class Solution:
    instance: Instance
    truck_route: list[int]
    sorties: list[Sortie] = field(default_factory=list)

    @property
    def drone_customers(self) -> set[int]:
        return {s.customer for s in self.sorties}

    def position_of(self, node: int) -> int:
        """Position of `node` in the truck route (first occurrence; depot starts at 0)."""
        return self.truck_route.index(node)

    def sortie_for_launch(self, node: int) -> Sortie | None:
        for s in self.sorties:
            if s.launch == node:
                return s
        return None

    def sortie_for_rendezvous(self, node: int) -> Sortie | None:
        for s in self.sorties:
            if s.rendezvous == node:
                return s
        return None

    def subroutes(self) -> list[Subroute]:
        """Partition the truck route at sortie launch/rendezvous nodes.

        Sortie endpoints are shared between adjacent subroutes, exactly like the
        vertical-bar separators in the thesis figures (e.g. Fig 4.3).
        """
        endpoints = {0, len(self.truck_route) - 1}
        sortie_by_launch_pos: dict[int, Sortie] = {}
        sortie_by_rendezvous_pos: dict[int, Sortie] = {}
        for s in self.sorties:
            lp = self.position_of(s.launch)
            rp = self.position_of(s.rendezvous)
            sortie_by_launch_pos[lp] = s
            sortie_by_rendezvous_pos[rp] = s
            endpoints.add(lp)
            endpoints.add(rp)
        ordered = sorted(endpoints)

        subroutes: list[Subroute] = []
        for a, b in pairwise(ordered):
            sortie = sortie_by_launch_pos.get(a)
            if sortie is None or sortie_by_rendezvous_pos.get(b) is not sortie:
                sortie = None
            subroutes.append(Subroute(positions=list(range(a, b + 1)), sortie=sortie))
        return subroutes

    def _simulate(self) -> tuple[list[float], list[float]]:
        """Walk the truck route and return (arrival, ready) times.

        - arrival[k] = the moment the truck physically reaches position k (no
          waiting at k yet applied).
        - ready[k] = the moment the truck is ready to leave position k (after any
          waiting for the drone, plus SR if k is a rendezvous, plus SL if k is
          also a launch — both apply when a single node closes one sortie and
          opens the next).
        """
        inst = self.instance
        route = self.truck_route
        n = len(route)
        arrival = [0.0] * n
        ready = [0.0] * n

        launch_pos = {self.position_of(s.launch): s for s in self.sorties}
        rendezvous_pos = {self.position_of(s.rendezvous): s for s in self.sorties}

        for k in range(n):
            if k == 0:
                arrival[k] = 0.0
            else:
                arrival[k] = ready[k - 1] + inst.truck_time(route[k - 1], route[k])
            ready[k] = arrival[k]
            if k in rendezvous_pos:
                s = rendezvous_pos[k]
                lp = self.position_of(s.launch)
                drone_arr = (
                    ready[lp]
                    + inst.drone_time(s.launch, s.customer)
                    + inst.drone_time(s.customer, s.rendezvous)
                )
                ready[k] = max(ready[k], drone_arr) + inst.sr
            if k in launch_pos:
                ready[k] += inst.sl
        return arrival, ready

    def truck_arrival_times(self) -> list[float]:
        """T[k] = truck arrival at position k of the route (pre-service at k).

        Matches the convention used in the thesis figures (e.g. Fig 4.3:
        T[5] = 8.1 is arrival, the 8.3 in the figure is the post-SL departure).
        """
        return self._simulate()[0]

    def truck_ready_times(self) -> list[float]:
        return self._simulate()[1]

    def total_completion_time(self) -> float:
        return self._simulate()[1][-1]
