"""Run the FSTSP method comparison and write a CSV, summary tables, and figures.

    uv run python scripts/run_experiments.py --seeds 10 --out experiments

Small instances (default n=5..12) get MILP + heuristic + SA so the exact optimum
anchors an optimality gap; large instances (default n=15,20,25,30) get heuristic
+ SA only. WLS credentials are read from .env / the environment (see
fstsp.gurobi_env); without a licence the MILP rows are skipped with a warning.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

from fstsp.experiments import (
    CSV_FIELDS,
    RunRecord,
    done_keys,
    optimality_gaps,
    read_csv,
    record_to_dict,
    run_suite,
)


def _sizes(spec: str) -> tuple[int, ...]:
    return tuple(int(x) for x in spec.split(",") if x.strip())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--small", default="5,6,7,8,9,10,11,12", help="sizes with MILP ground truth")
    p.add_argument("--large", default="15,20,25,30", help="heuristic+SA-only sizes")
    p.add_argument("--seeds", type=int, default=10, help="instances per size (seeds 0..S-1)")
    p.add_argument("--sa-iters", type=int, default=50_000, help="SA iterations per run")
    p.add_argument("--milp-time-limit", type=float, default=60.0, help="MILP seconds/instance")
    p.add_argument("--milp-max-n", type=int, default=12, help="largest n to run the MILP on")
    p.add_argument("--out", type=Path, default=Path("experiments"), help="output directory")
    p.add_argument("--no-plots", action="store_true", help="skip figure generation")
    p.add_argument(
        "--fresh", action="store_true", help="ignore any existing results.csv (don't resume)"
    )
    return p.parse_args()


def _fmt(x: float | None, width: int = 9, prec: int = 1) -> str:
    return f"{x:>{width}.{prec}f}" if x is not None else f"{'-':>{width}}"


def print_summary(records: list[RunRecord]) -> None:
    by_key: dict[tuple[str, int], list[RunRecord]] = defaultdict(list)
    for r in records:
        by_key[(r.method, r.n)].append(r)

    print("\n=== per (method, n): mean objective / mean runtime(s) / optimality ===")
    print(f"{'method':>10} {'n':>3} {'count':>5} {'mean_obj':>10} {'mean_rt':>9} {'opt':>5}")
    for (method, n) in sorted(by_key, key=lambda k: (k[0], k[1])):
        rs = by_key[(method, n)]
        objs = [r.objective for r in rs if r.objective is not None]
        mean_obj = statistics.mean(objs) if objs else None
        mean_rt = statistics.mean(r.runtime_s for r in rs)
        opt = sum(r.proven_optimal for r in rs)
        opt_str = f"{opt}/{len(rs)}" if method == "milp" else "-"
        print(
            f"{method:>10} {n:>3} {len(rs):>5} "
            f"{_fmt(mean_obj, 10)} {_fmt(mean_rt, 9, 3)} {opt_str:>5}"
        )

    gaps = optimality_gaps(records)
    if not gaps:
        print("\n(no optimality gaps: no proven MILP optima available)")
        return
    print("\n=== optimality gap vs MILP (%), small instances only ===")
    print(f"{'method':>10} {'n':>3} {'count':>5} {'mean%':>8} {'median%':>8} {'max%':>8}")
    by_mn: dict[tuple[str, int], list[float]] = defaultdict(list)
    for g in gaps:
        by_mn[(g["method"], g["n"])].append(g["gap"] * 100)
    for (method, n) in sorted(by_mn):
        vals = by_mn[(method, n)]
        print(
            f"{method:>10} {n:>3} {len(vals):>5} "
            f"{statistics.mean(vals):>8.2f} {statistics.median(vals):>8.2f} {max(vals):>8.2f}"
        )


def make_plots(records: list[RunRecord], out: Path, *, sa_iters: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1) Optimality-gap boxplot per method (small instances).
    gaps = optimality_gaps(records)
    if gaps:
        by_method: dict[str, list[float]] = defaultdict(list)
        for g in gaps:
            by_method[g["method"]].append(g["gap"] * 100)
        methods = sorted(by_method)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.boxplot([by_method[m] for m in methods], tick_labels=methods, showmeans=True)
        ax.set_ylabel("optimality gap vs MILP (%)")
        ax.set_title("Heuristic vs SA — optimality gap (small instances)")
        ax.axhline(0, color="grey", lw=0.8, ls="--")
        fig.tight_layout()
        fig.savefig(out / "gap_boxplot.png", dpi=150)
        plt.close(fig)

    # 2) Mean runtime vs n, per method (log y — shows the MILP's exponential wall).
    by_key: dict[tuple[str, int], list[float]] = defaultdict(list)
    for r in records:
        by_key[(r.method, r.n)].append(r.runtime_s)
    fig, ax = plt.subplots(figsize=(6, 4))
    for method in ("milp", "heuristic", "sa"):
        pts = sorted((n, statistics.mean(v)) for (m, n), v in by_key.items() if m == method)
        if pts:
            xs, ys = zip(*pts, strict=True)
            ax.plot(xs, ys, marker="o", label=method)
    ax.set_yscale("log")
    ax.set_xlabel("number of customers n")
    ax.set_ylabel("mean runtime (s, log scale)")
    ax.set_title("Runtime vs instance size")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "runtime_vs_n.png", dpi=150)
    plt.close(fig)

    # 3) An SA convergence trace on a representative mid-size instance.
    from fstsp import initial_truck_solution, random_euclidean, simulated_annealing
    from fstsp.viz.mpl import plot_convergence

    inst = random_euclidean(n_customers=20, seed=0)
    res = simulated_annealing(
        initial_truck_solution(inst), iterations=sa_iters, seed=0, record=True
    )
    ax = plot_convergence(res, title="SA convergence (n=20, seed=0)")
    ax.figure.savefig(out / "sa_convergence.png", dpi=150)
    plt.close(ax.figure)


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / "results.csv"

    # Resume: load existing rows and skip the (method, n, seed) keys already done.
    existing: list[RunRecord] = []
    if csv_path.exists() and not args.fresh:
        existing = read_csv(csv_path)
        print(f"resuming: {len(existing)} rows already in {csv_path}")
    elif args.fresh and csv_path.exists():
        csv_path.unlink()
    skip = done_keys(existing)

    env = None
    small = _sizes(args.small)
    try:
        from fstsp.gurobi_env import make_env

        env = make_env()
    except Exception as exc:  # any licence/import failure -> skip MILP
        print(f"warning: no Gurobi env ({exc}); skipping MILP rows")
        small = ()

    # Append each record as it lands, flushing so a crash/disconnect keeps progress.
    f = csv_path.open("a", newline="")
    writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
    if not existing:
        writer.writeheader()
    n_done = 0

    def progress(rec: RunRecord) -> None:
        nonlocal n_done
        n_done += 1
        writer.writerow(record_to_dict(rec))
        f.flush()
        obj = f"{rec.objective:.1f}" if rec.objective is not None else "none"
        print(f"[{n_done:>4}] {rec.method:>10} n={rec.n:>2} seed={rec.seed} "
              f"obj={obj:>9} rt={rec.runtime_s:6.2f}s")

    try:
        run_suite(
            small_sizes=small,
            large_sizes=_sizes(args.large),
            seeds=args.seeds,
            sa_iterations=args.sa_iters,
            milp_time_limit=args.milp_time_limit,
            milp_max_n=args.milp_max_n,
            env=env,
            progress=progress,
            skip=skip,
        )
    finally:
        f.close()
        if env is not None:
            env.dispose()

    # Summarise / plot from the full CSV (existing + newly appended).
    records = read_csv(csv_path)
    print(f"\n{len(records)} rows total -> {csv_path}")
    print_summary(records)

    if not args.no_plots:
        make_plots(records, args.out, sa_iters=args.sa_iters)
        print(f"\nfigures -> {args.out}/gap_boxplot.png, runtime_vs_n.png, sa_convergence.png")


if __name__ == "__main__":
    main()
