"""Render the §4.1.2 example to PNG + HTML + MP4/GIF.

Run with:
    uv run --extra viz python scripts/render_example.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from fstsp import murray_chu
from fstsp.examples import thesis_4_1_2
from fstsp.viz import mpl as vmpl
from fstsp.viz import plotly as vplotly


def main() -> None:
    out = Path("renders")
    out.mkdir(exist_ok=True)

    inst, init_route = thesis_4_1_2()
    sol = murray_chu(inst, init_route)
    print(f"heuristic completion time: {sol.total_completion_time():.2f}")
    print(f"  truck route: {sol.truck_route}")
    for s in sol.sorties:
        print(f"  sortie: {s.launch} → {s.customer} → {s.rendezvous}")

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(7.5, 9),
        gridspec_kw={"height_ratios": [4, 1]},
    )
    vmpl.plot_route(sol, ax=axes[0], title="§4.1.2 - Murray-Chu heuristic")
    vmpl.plot_gantt(sol, ax=axes[1])
    fig.tight_layout()
    png_path = out / "thesis_4_1_2.png"
    fig.savefig(png_path, dpi=200)
    plt.close(fig)
    print(f"wrote {png_path}")

    anim = vmpl.animate(sol, fps=30)
    mp4_path = out / "thesis_4_1_2.mp4"
    try:
        anim.save(mp4_path, fps=30, dpi=150)
        print(f"wrote {mp4_path}")
    except Exception as exc:
        from matplotlib.animation import PillowWriter

        gif_path = out / "thesis_4_1_2.gif"
        anim.save(gif_path, writer=PillowWriter(fps=20))
        print(f"mp4 unavailable ({exc.__class__.__name__}: {exc}); wrote {gif_path}")
    plt.close("all")

    html_fig = vplotly.animate(sol, title="§4.1.2 — interactive")
    html_path = out / "thesis_4_1_2.html"
    html_fig.write_html(html_path, include_plotlyjs="cdn")
    print(f"wrote {html_path}")


if __name__ == "__main__":
    main()
