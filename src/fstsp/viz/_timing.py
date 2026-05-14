"""Time-stepping helpers for animations.

Given a Solution, compute (x, y) for the truck and the drone at an arbitrary
time t in [0, completion_time]. The truck either sits at a node (waiting /
service) or interpolates linearly along its current arc; the drone rides the
truck except during sortie flight, where it interpolates launch → customer →
rendezvous.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fstsp.solution import Solution


@dataclass
class SortieTiming:
    launch_pos: int
    rendezvous_pos: int
    t_launch: float  # truck departs launch (drone deploys)
    t_at_customer: float  # drone reaches customer
    t_at_rendezvous: float  # drone reaches rendezvous (may wait for truck)
    t_back_on_truck: float  # truck ready to depart rendezvous (after SR)


@dataclass
class Schedule:
    arrival: list[float]
    ready: list[float]
    sortie_times: list[SortieTiming]

    @property
    def completion(self) -> float:
        return self.ready[-1]


def build_schedule(sol: Solution) -> Schedule:
    arrival = sol.truck_arrival_times()
    ready = sol.truck_ready_times()
    sortie_times: list[SortieTiming] = []
    for s in sol.sorties:
        lp = sol.position_of(s.launch)
        rp = sol.position_of(s.rendezvous)
        t_launch = ready[lp]
        t_at_customer = t_launch + sol.instance.d[s.launch, s.customer]
        t_at_rendezvous = t_at_customer + sol.instance.d[s.customer, s.rendezvous]
        sortie_times.append(
            SortieTiming(
                launch_pos=lp,
                rendezvous_pos=rp,
                t_launch=t_launch,
                t_at_customer=t_at_customer,
                t_at_rendezvous=t_at_rendezvous,
                t_back_on_truck=ready[rp],
            )
        )
    return Schedule(arrival=arrival, ready=ready, sortie_times=sortie_times)


def truck_position(sol: Solution, coords: np.ndarray, sched: Schedule, t: float) -> np.ndarray:
    route = sol.truck_route
    if t <= 0.0:
        return coords[route[0]].copy()
    for k in range(len(route) - 1):
        if t <= sched.ready[k]:
            return coords[route[k]].copy()
        if t <= sched.arrival[k + 1]:
            depart = sched.ready[k]
            span = sched.arrival[k + 1] - depart
            frac = (t - depart) / span if span > 0 else 1.0
            return coords[route[k]] + frac * (coords[route[k + 1]] - coords[route[k]])
    return coords[route[-1]].copy()


def drone_position(sol: Solution, coords: np.ndarray, sched: Schedule, t: float) -> np.ndarray:
    for s, st in zip(sol.sorties, sched.sortie_times, strict=True):
        if t < st.t_launch:
            continue
        if t < st.t_at_customer:
            span = st.t_at_customer - st.t_launch
            frac = (t - st.t_launch) / span if span > 0 else 1.0
            return coords[s.launch] + frac * (coords[s.customer] - coords[s.launch])
        if t < st.t_at_rendezvous:
            span = st.t_at_rendezvous - st.t_at_customer
            frac = (t - st.t_at_customer) / span if span > 0 else 1.0
            return coords[s.customer] + frac * (coords[s.rendezvous] - coords[s.customer])
        if t < st.t_back_on_truck:
            return coords[s.rendezvous].copy()
    return truck_position(sol, coords, sched, t)
