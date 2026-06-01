"""Smoke tests for the viz subpackage.

We do not snapshot images — these tests only verify that figure construction
runs to completion and that the time-stepping helpers return the expected
endpoints. The heavy lifting (image fidelity) is left to manual review.
"""

from __future__ import annotations

import numpy as np
import pytest

from fstsp import murray_chu
from fstsp.examples import thesis_4_1_2

pytest.importorskip("matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from fstsp.viz import mpl as vmpl
from fstsp.viz._layout import classical_mds, coords_for
from fstsp.viz._timing import (
    build_schedule,
    drone_position,
    truck_position,
)


@pytest.fixture
def solved():
    inst, route = thesis_4_1_2()
    return murray_chu(inst, route)


def test_classical_mds_recovers_distances():
    inst, _ = thesis_4_1_2()
    coords = classical_mds(inst.t)
    assert coords.shape == (inst.n_nodes, 2)
    # MDS approximates pairwise distances; allow ~25% slack since §4.1.2's
    # matrix isn't perfectly Euclidean.
    recovered = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    mask = inst.t > 0
    relative_err = np.abs(recovered[mask] - inst.t[mask]) / inst.t[mask]
    assert relative_err.mean() < 0.25


def test_coords_for_passthrough():
    inst, _ = thesis_4_1_2()
    explicit = np.arange(inst.n_nodes * 2, dtype=float).reshape(inst.n_nodes, 2)
    out = coords_for(inst, explicit)
    # Physical rows pass through unchanged; a trailing end-depot row mirrors the depot.
    assert out.shape == (inst.n_nodes + 1, 2)
    np.testing.assert_array_equal(out[: inst.n_nodes], explicit)
    np.testing.assert_array_equal(out[inst.end_depot], explicit[inst.depot])
    # Idempotent: re-passing the resolved coords returns them unchanged.
    np.testing.assert_array_equal(coords_for(inst, out), out)


def test_coords_for_rejects_wrong_shape():
    inst, _ = thesis_4_1_2()
    with pytest.raises(ValueError):
        coords_for(inst, np.zeros((3, 2)))


def test_truck_position_endpoints(solved):
    coords = coords_for(solved.instance)
    sched = build_schedule(solved)
    np.testing.assert_allclose(
        truck_position(solved, coords, sched, 0.0),
        coords[solved.truck_route[0]],
    )
    np.testing.assert_allclose(
        truck_position(solved, coords, sched, sched.completion),
        coords[solved.truck_route[-1]],
    )


def test_drone_position_riding_truck_at_t0(solved):
    coords = coords_for(solved.instance)
    sched = build_schedule(solved)
    np.testing.assert_allclose(
        drone_position(solved, coords, sched, 0.0),
        coords[solved.truck_route[0]],
    )


def test_drone_at_customer_midflight(solved):
    if not solved.sorties:
        pytest.skip("solution has no sorties to check")
    coords = coords_for(solved.instance)
    sched = build_schedule(solved)
    s = solved.sorties[0]
    st = sched.sortie_times[0]
    pos = drone_position(solved, coords, sched, st.t_at_customer)
    np.testing.assert_allclose(pos, coords[s.customer], atol=1e-9)


def test_plot_route_smoke(solved):
    ax = vmpl.plot_route(solved, title="t")
    assert ax.has_data()
    plt.close(ax.figure)


def test_plot_gantt_smoke(solved):
    ax = vmpl.plot_gantt(solved)
    assert ax.has_data()
    plt.close(ax.figure)


def test_animate_constructs_and_updates(solved, tmp_path):
    from matplotlib.animation import PillowWriter

    anim = vmpl.animate(solved, fps=10, duration=0.5)
    update = anim._func  # FuncAnimation.func is exposed as _func
    assert update(0)
    assert update(2)
    # Actually render to disk so matplotlib doesn't warn about the
    # animation being deleted without ever drawing a frame.
    anim.save(tmp_path / "anim.gif", writer=PillowWriter(fps=5))
    plt.close("all")


def test_plot_convergence_smoke():
    from fstsp import initial_truck_solution, random_euclidean, simulated_annealing

    inst = random_euclidean(n_customers=8, area_side_km=12.0, seed=4)
    result = simulated_annealing(initial_truck_solution(inst), iterations=600, seed=0, record=True)
    ax = vmpl.plot_convergence(result, title="convergence")
    assert ax.has_data()
    plt.close(ax.figure)


def test_plot_convergence_requires_trace():
    from fstsp import initial_truck_solution, random_euclidean, simulated_annealing

    inst = random_euclidean(n_customers=8, area_side_km=12.0, seed=4)
    result = simulated_annealing(initial_truck_solution(inst), iterations=200, seed=0)
    with pytest.raises(ValueError, match="trace"):
        vmpl.plot_convergence(result)


def test_plotly_plot_route_smoke(solved):
    pytest.importorskip("plotly")
    from fstsp.viz import plotly as vplotly

    fig = vplotly.plot_route(solved, title="t")
    assert len(fig.data) > 0


def test_plotly_animate_has_frames(solved):
    pytest.importorskip("plotly")
    from fstsp.viz import plotly as vplotly

    fig = vplotly.animate(solved, n_frames=30)
    assert len(fig.frames) == 30
