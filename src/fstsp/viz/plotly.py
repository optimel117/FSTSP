"""Plotly visualisations: interactive HTML route plot and time-slider animation."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from fstsp.solution import Solution
from fstsp.viz._layout import coords_for
from fstsp.viz._timing import build_schedule, drone_position, truck_position

TRUCK_COLOR = "#1f77b4"
DRONE_COLOR = "#d62728"
DEPOT_COLOR = "#2ca02c"


def plot_route(
    sol: Solution,
    *,
    coords: np.ndarray | None = None,
    title: str | None = None,
    height: int = 600,
) -> go.Figure:
    """Static interactive route plot with hover tooltips."""
    inst = sol.instance
    coords = coords_for(inst, coords)
    fig = go.Figure()

    xs: list[float | None] = []
    ys: list[float | None] = []
    for a, b in zip(sol.truck_route[:-1], sol.truck_route[1:], strict=True):
        xs += [float(coords[a, 0]), float(coords[b, 0]), None]
        ys += [float(coords[a, 1]), float(coords[b, 1]), None]
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            line={"color": TRUCK_COLOR, "width": 2.5},
            name="Truck",
            hoverinfo="skip",
        )
    )

    for i, s in enumerate(sol.sorties, start=1):
        hover = f"sortie {i}: {s.launch} → {s.customer} → {s.rendezvous}<extra></extra>"
        for src, dst, leg in [(s.launch, s.customer, "out"), (s.customer, s.rendezvous, "back")]:
            fig.add_trace(
                go.Scatter(
                    x=[float(coords[src, 0]), float(coords[dst, 0])],
                    y=[float(coords[src, 1]), float(coords[dst, 1])],
                    mode="lines",
                    line={"color": DRONE_COLOR, "width": 2.2, "dash": "dash"},
                    name=f"Sortie {i}",
                    legendgroup=f"sortie-{i}",
                    showlegend=(leg == "out"),
                    hovertemplate=hover,
                )
            )

    drone_customers = sol.drone_customers
    for i in range(inst.n_nodes):
        if i == inst.depot:
            color, symbol = DEPOT_COLOR, "square"
        elif i in drone_customers:
            color, symbol = DRONE_COLOR, "circle"
        else:
            color, symbol = "white", "circle"
        fig.add_trace(
            go.Scatter(
                x=[float(coords[i, 0])],
                y=[float(coords[i, 1])],
                mode="markers+text",
                text=["D" if i == inst.depot else str(i)],
                textposition="middle center",
                textfont={"size": 11, "color": "black"},
                marker={
                    "size": 22,
                    "color": color,
                    "symbol": symbol,
                    "line": {"color": "black", "width": 1},
                },
                hovertemplate=f"node {i}<extra></extra>",
                showlegend=False,
            )
        )

    fig.update_layout(
        title=title,
        xaxis={"visible": False, "scaleanchor": "y", "scaleratio": 1},
        yaxis={"visible": False},
        margin={"l": 10, "r": 10, "t": 50 if title else 20, "b": 10},
        plot_bgcolor="white",
        height=height,
        legend={"x": 0.01, "y": 0.99, "bgcolor": "rgba(255,255,255,0.7)"},
    )
    return fig


def animate(
    sol: Solution,
    *,
    coords: np.ndarray | None = None,
    n_frames: int = 200,
    title: str | None = None,
    height: int = 700,
) -> go.Figure:
    """Time-slider animation: static route as backdrop, two moving markers."""
    inst = sol.instance
    coords = coords_for(inst, coords)
    sched = build_schedule(sol)
    end_t = sched.completion
    times = np.linspace(0.0, end_t, n_frames)

    fig = plot_route(sol, coords=coords, title=title, height=height)

    truck_idx = len(fig.data)
    drone_idx = truck_idx + 1
    start_xy = coords[sol.truck_route[0]]
    fig.add_trace(
        go.Scatter(
            x=[float(start_xy[0])],
            y=[float(start_xy[1])],
            mode="markers",
            marker={
                "size": 20,
                "symbol": "square",
                "color": TRUCK_COLOR,
                "line": {"color": "black", "width": 1.5},
            },
            name="Truck (now)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[float(start_xy[0])],
            y=[float(start_xy[1])],
            mode="markers",
            marker={
                "size": 16,
                "symbol": "triangle-up",
                "color": DRONE_COLOR,
                "line": {"color": "black", "width": 1.5},
            },
            name="Drone (now)",
        )
    )

    frames: list[go.Frame] = []
    for t in times:
        tp = truck_position(sol, coords, sched, float(t))
        dp = drone_position(sol, coords, sched, float(t))
        frames.append(
            go.Frame(
                data=[
                    go.Scatter(x=[float(tp[0])], y=[float(tp[1])]),
                    go.Scatter(x=[float(dp[0])], y=[float(dp[1])]),
                ],
                traces=[truck_idx, drone_idx],
                name=f"{t:.3f}",
            )
        )
    fig.frames = frames

    slider_step = max(1, n_frames // 40)
    fig.update_layout(
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "buttons": [
                    {
                        "label": "▶ Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 1000 * end_t / n_frames, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                    {
                        "label": "⏸ Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                ],
                "x": 0.02,
                "y": 1.08,
                "xanchor": "left",
                "yanchor": "top",
                "pad": {"l": 0, "r": 0, "t": 0, "b": 0},
            }
        ],
        sliders=[
            {
                "active": 0,
                "currentvalue": {"prefix": "t = ", "font": {"size": 12}},
                "steps": [
                    {
                        "method": "animate",
                        "args": [
                            [f"{times[i]:.3f}"],
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "mode": "immediate",
                            },
                        ],
                        "label": f"{times[i]:.1f}",
                    }
                    for i in range(0, n_frames, slider_step)
                ],
                "x": 0.10,
                "len": 0.85,
                "pad": {"l": 0, "r": 0, "t": 30, "b": 10},
            }
        ],
    )
    return fig
