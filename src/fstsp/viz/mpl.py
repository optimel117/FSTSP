"""Matplotlib visualisations: static route + Gantt, and a combined animation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from fstsp.solution import Solution
from fstsp.viz._layout import coords_for
from fstsp.viz._timing import (
    Schedule,
    build_schedule,
    drone_position,
    truck_position,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from fstsp.sa import SAResult


TRUCK_COLOR = "#1f77b4"
DRONE_COLOR = "#d62728"
DEPOT_COLOR = "#2ca02c"
NODE_EDGE = "#333333"
BEST_COLOR = "#08306b"
TEMP_COLOR = "#ff7f0e"


def plot_route(
    sol: Solution,
    *,
    coords: np.ndarray | None = None,
    ax: Axes | None = None,
    title: str | None = None,
    show_node_labels: bool = True,
) -> Axes:
    """Static route plot. Truck arcs solid, drone sorties dashed.

    Launch nodes get a blue ring, rendezvous nodes get a blue diamond, drone-
    served customers are filled red. The depot is a green square.
    """
    inst = sol.instance
    coords = coords_for(inst, coords)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    drone_customers = sol.drone_customers
    launches = {s.launch for s in sol.sorties}
    rendezvous = {s.rendezvous for s in sol.sorties}

    for a, b in zip(sol.truck_route[:-1], sol.truck_route[1:], strict=True):
        _draw_arrow(ax, coords[a], coords[b], color=TRUCK_COLOR, dashed=False)
    for s in sol.sorties:
        _draw_arrow(ax, coords[s.launch], coords[s.customer], color=DRONE_COLOR, dashed=True)
        _draw_arrow(ax, coords[s.customer], coords[s.rendezvous], color=DRONE_COLOR, dashed=True)

    for i in range(inst.n_nodes):
        x, y = coords[i]
        if i == inst.depot:
            ax.scatter(
                x, y, marker="s", s=180, c=DEPOT_COLOR, zorder=3, edgecolors="black", linewidths=1.0
            )
        elif i in drone_customers:
            ax.scatter(
                x, y, marker="o", s=140, c=DRONE_COLOR, zorder=3, edgecolors="black", linewidths=1.0
            )
        else:
            ax.scatter(
                x, y, marker="o", s=140, c="white", zorder=3, edgecolors=NODE_EDGE, linewidths=1.2
            )
        if i in launches:
            ax.scatter(
                x,
                y,
                marker="o",
                s=300,
                facecolors="none",
                edgecolors=TRUCK_COLOR,
                linewidths=1.8,
                zorder=2,
            )
        if i in rendezvous:
            ax.scatter(
                x,
                y,
                marker="D",
                s=240,
                facecolors="none",
                edgecolors=TRUCK_COLOR,
                linewidths=1.8,
                zorder=2,
            )
        if show_node_labels:
            label = "D" if i == inst.depot else str(i)
            ax.annotate(label, (x, y), ha="center", va="center", fontsize=9, zorder=4)

    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title)

    handles = [
        Line2D([0], [0], color=TRUCK_COLOR, lw=2, label="Truck"),
        Line2D([0], [0], color=DRONE_COLOR, lw=2, ls="--", label="Drone"),
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor=DEPOT_COLOR,
            markeredgecolor="black",
            markersize=10,
            label="Depot",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=DRONE_COLOR,
            markeredgecolor="black",
            markersize=10,
            label="Drone customer",
        ),
    ]
    ax.legend(handles=handles, loc="best", frameon=False, fontsize=8)
    return ax


def _draw_arrow(ax: Axes, p0: np.ndarray, p1: np.ndarray, *, color: str, dashed: bool) -> None:
    ax.annotate(
        "",
        xy=p1,
        xytext=p0,
        arrowprops={
            "arrowstyle": "-|>",
            "color": color,
            "lw": 1.6,
            "shrinkA": 11,
            "shrinkB": 11,
            "linestyle": "--" if dashed else "-",
        },
        zorder=1,
    )


def plot_gantt(
    sol: Solution,
    *,
    sched: Schedule | None = None,
    ax: Axes | None = None,
    title: str | None = None,
    legend: bool = True,
) -> Axes:
    """Two-row Gantt: truck above, drone below.

    Truck: dark = travel, light = service/waiting at a node.
    Drone: solid = outbound flight, medium = inbound, light = idling at the
    rendezvous waiting for the truck.
    """
    sched = sched or build_schedule(sol)
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 2.6))

    for k in range(len(sol.truck_route) - 1):
        if sched.ready[k] > sched.arrival[k]:
            ax.barh(
                1,
                sched.ready[k] - sched.arrival[k],
                left=sched.arrival[k],
                height=0.4,
                color=TRUCK_COLOR,
                alpha=0.35,
            )
        ax.barh(
            1,
            sched.arrival[k + 1] - sched.ready[k],
            left=sched.ready[k],
            height=0.4,
            color=TRUCK_COLOR,
        )
    last = len(sol.truck_route) - 1
    if sched.ready[last] > sched.arrival[last]:
        ax.barh(
            1,
            sched.ready[last] - sched.arrival[last],
            left=sched.arrival[last],
            height=0.4,
            color=TRUCK_COLOR,
            alpha=0.35,
        )

    for st in sched.sortie_times:
        ax.barh(0, st.t_at_customer - st.t_launch, left=st.t_launch, height=0.4, color=DRONE_COLOR)
        ax.barh(
            0,
            st.t_at_rendezvous - st.t_at_customer,
            left=st.t_at_customer,
            height=0.4,
            color=DRONE_COLOR,
            alpha=0.7,
        )
        if st.t_back_on_truck > st.t_at_rendezvous:
            ax.barh(
                0,
                st.t_back_on_truck - st.t_at_rendezvous,
                left=st.t_at_rendezvous,
                height=0.4,
                color=DRONE_COLOR,
                alpha=0.3,
            )

    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Drone", "Truck"])
    ax.set_xlim(0, sched.completion * 1.02)
    ax.set_xlabel("time")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if title:
        ax.set_title(title)
    if legend:
        handles = [
            Patch(facecolor=TRUCK_COLOR, label="truck: travel"),
            Patch(facecolor=TRUCK_COLOR, alpha=0.35, label="truck: idle/service"),
            Patch(facecolor=DRONE_COLOR, label="drone: outbound"),
            Patch(facecolor=DRONE_COLOR, alpha=0.7, label="drone: inbound"),
            Patch(facecolor=DRONE_COLOR, alpha=0.3, label="drone: waiting"),
        ]
        ax.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncols=5,
            frameon=False,
            fontsize=8,
            handlelength=1.2,
            handleheight=1.0,
            columnspacing=1.0,
        )
    return ax


def animate(
    sol: Solution,
    *,
    coords: np.ndarray | None = None,
    fps: int = 30,
    duration: float | None = None,
    figsize: tuple[float, float] = (8, 9),
) -> FuncAnimation:
    """Combined animation: route plot on top with moving truck/drone markers,
    Gantt below with a moving time cursor.

    `duration` is the wall-clock length of the animation in seconds. If left
    unset it defaults to the simulated completion time, clamped to [3, 20] sec
    so generators with real-world units (e.g. ~18000 sec sim) don't render
    a multi-hour MP4.
    """
    coords = coords_for(sol.instance, coords)
    sched = build_schedule(sol)
    end_t = sched.completion
    if duration is None:
        duration = min(20.0, max(3.0, end_t))
    n_frames = max(2, int(duration * fps))

    fig, (ax_route, ax_gantt) = plt.subplots(
        2,
        1,
        figsize=figsize,
        gridspec_kw={"height_ratios": [4, 1]},
    )
    plot_route(sol, coords=coords, ax=ax_route)
    plot_gantt(sol, sched=sched, ax=ax_gantt)
    cursor = ax_gantt.axvline(0, color="black", lw=1.4, zorder=10)

    truck_marker = ax_route.scatter(
        [coords[sol.truck_route[0], 0]],
        [coords[sol.truck_route[0], 1]],
        marker="s",
        s=180,
        c=TRUCK_COLOR,
        edgecolors="black",
        linewidths=1.2,
        zorder=10,
    )
    drone_marker = ax_route.scatter(
        [coords[sol.truck_route[0], 0]],
        [coords[sol.truck_route[0], 1]],
        marker="^",
        s=130,
        c=DRONE_COLOR,
        edgecolors="black",
        linewidths=1.2,
        zorder=10,
    )
    time_text = ax_route.text(
        0.02,
        0.98,
        "",
        transform=ax_route.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 3.0},
    )

    def update(frame: int):
        t = (frame / (n_frames - 1)) * end_t
        tp = truck_position(sol, coords, sched, t)
        dp = drone_position(sol, coords, sched, t)
        truck_marker.set_offsets([tp])
        drone_marker.set_offsets([dp])
        cursor.set_xdata([t, t])
        time_text.set_text(f"t = {t:5.2f}")
        return truck_marker, drone_marker, cursor, time_text

    return FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps, blit=False)


def plot_convergence(
    result: SAResult,
    *,
    ax: Axes | None = None,
    title: str | None = None,
    show_temperature: bool = True,
) -> Axes:
    """Plot SA convergence: current and best-so-far objective against iteration.

    Requires a result from ``simulated_annealing(..., record=True)``. The thin
    line is the accepted current objective (the exploration), the bold line is
    the best found so far. With `show_temperature`, the geometric cooling
    schedule is drawn on a twin axis.
    """
    trace = result.trace
    if trace is None:
        raise ValueError("result has no trace; run simulated_annealing(..., record=True)")
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4.5))

    it = trace.iteration
    ax.plot(it, trace.z_current, color=TRUCK_COLOR, lw=0.6, alpha=0.45, label="current")
    ax.plot(it, trace.z_best, color=BEST_COLOR, lw=1.8, label="best so far")
    ax.set_xlabel("iteration")
    ax.set_ylabel("completion time (s)")
    ax.set_xlim(it[0], it[-1])
    ax.spines["top"].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()

    if show_temperature:
        ax2 = ax.twinx()
        ax2.plot(it, trace.temperature, color=TEMP_COLOR, lw=1.2, ls="--", label="temperature")
        ax2.set_ylabel("temperature")
        ax2.set_ylim(bottom=0)
        ax2.spines["top"].set_visible(False)
        h2, l2 = ax2.get_legend_handles_labels()
        handles += h2
        labels += l2

    ax.legend(handles, labels, loc="upper right", frameon=False, fontsize=9)
    if title:
        ax.set_title(title)
    return ax
