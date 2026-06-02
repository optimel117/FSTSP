"""Run the FSTSP method comparison over the thesis grid and write CSVs + figures.

    uv run python scripts/run_experiments.py --out experiments

The grid is size x replication x endurance x hub (default 3 x 10 x 3 x 2 = 180
configs), each solved by the MILP (exact, lazy subtour cuts), the Murray-Chu
heuristic, and SA. Outputs in --out:

  results.csv    one row per (method, config): objective, runtime, status,
                 proven_optimal, gap, truck route, drone sorties, seed.
  instances.csv  customer/depot coordinates for every (n, replication, hub).
  *.png          summary figures.

WLS credentials are read from .env / the environment (see fstsp.gurobi_env);
without a licence the MILP rows are skipped with a warning and only the
heuristic + SA run (e.g. on a worker without Gurobi).
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

from fstsp.experiments import (
    CSV_FIELDS,
    ENDURANCES,
    HUBS,
    REPLICATIONS,
    RunRecord,
    done_keys,
    optimality_gaps,
    read_csv,
    record_to_dict,
    run_suite,
    write_instances_csv,
    write_results_with_gaps,
)


def _floats(spec: str) -> tuple[float, ...]:
    return tuple(float(x) for x in spec.split(",") if x.strip())


def _ints(spec: str) -> tuple[int, ...]:
    return tuple(int(x) for x in spec.split(",") if x.strip())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sizes", default="10,20,30,40", help="customer-set sizes n")
    p.add_argument("--reps", type=int, default=len(REPLICATIONS), help="replications per size")
    p.add_argument("--endurances", default=",".join(f"{e:g}" for e in ENDURANCES),
                   help="drone endurance D_tl values (seconds)")
    p.add_argument("--hubs", default=",".join(HUBS), help="depot positions")
    p.add_argument("--sa-iters", type=int, default=100_000, help="SA iterations per run")
    p.add_argument("--sa-reps", type=int, default=10, help="SA runs per config (seeds 0..R-1)")
    p.add_argument("--hybrid-reps", type=int, default=None,
                   help="hybrid-SA runs per config (default: same as --sa-reps)")
    p.add_argument("--milp-time-limit", type=float, default=3600.0, help="MILP seconds/config")
    p.add_argument("--milp-max-n", type=int, default=None, help="largest n to run the MILP on")
    p.add_argument("--methods", default="milp,heuristic,sa,hybrid_sa",
                   help="subset of methods to run")
    p.add_argument("--out", type=Path, default=Path("experiments"), help="output directory")
    p.add_argument("--no-plots", action="store_true", help="skip figure generation")
    p.add_argument("--fresh", action="store_true", help="ignore existing results.csv; don't resume")
    return p.parse_args()


def _fmt(x: float | None, width: int = 9, prec: int = 1) -> str:
    return f"{x:>{width}.{prec}f}" if x is not None else f"{'-':>{width}}"


def print_summary(records: list[RunRecord]) -> None:
    by_key: dict[tuple[str, int], list[RunRecord]] = defaultdict(list)
    for r in records:
        by_key[(r.method, r.n)].append(r)

    print("\n=== per (method, n): mean objective / mean runtime(s) / MILP optimality ===")
    print(f"{'method':>10} {'n':>3} {'count':>5} {'mean_obj':>10} {'mean_rt':>9} {'opt':>7}")
    for (method, n) in sorted(by_key, key=lambda k: (k[0], k[1])):
        rs = by_key[(method, n)]
        objs = [r.objective for r in rs if r.objective is not None]
        mean_obj = statistics.mean(objs) if objs else None
        mean_rt = statistics.mean(r.runtime_s for r in rs)
        opt = sum(r.proven_optimal for r in rs)
        opt_str = f"{opt}/{len(rs)}" if method == "milp" else "-"
        print(f"{method:>10} {n:>3} {len(rs):>5} "
              f"{_fmt(mean_obj, 10)} {_fmt(mean_rt, 9, 3)} {opt_str:>7}")

    gaps = optimality_gaps(records)
    if not gaps:
        print("\n(no optimality gaps: no proven MILP optima available)")
        return
    print("\n=== optimality gap vs MILP (%), where MILP proved optimal ===")
    print(f"{'method':>10} {'n':>3} {'count':>5} {'mean%':>8} {'median%':>8} {'max%':>8}")
    by_mn: dict[tuple[str, int], list[float]] = defaultdict(list)
    for g in gaps:
        by_mn[(g["method"], g["n"])].append(g["gap"] * 100)
    for (method, n) in sorted(by_mn):
        vals = by_mn[(method, n)]
        print(f"{method:>10} {n:>3} {len(vals):>5} "
              f"{statistics.mean(vals):>8.2f} {statistics.median(vals):>8.2f} {max(vals):>8.2f}")


def make_plots(records: list[RunRecord], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1) Optimality-gap boxplot per method.
    gaps = optimality_gaps(records)
    if gaps:
        by_method: dict[str, list[float]] = defaultdict(list)
        for g in gaps:
            by_method[g["method"]].append(g["gap"] * 100)
        methods = sorted(by_method)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.boxplot([by_method[m] for m in methods], tick_labels=methods, showmeans=True)
        ax.set_ylabel("optimality gap vs MILP (%)")
        ax.set_title("Heuristic vs SA -- optimality gap")
        ax.axhline(0, color="grey", lw=0.8, ls="--")
        fig.tight_layout()
        fig.savefig(out / "gap_boxplot.png", dpi=150)
        plt.close(fig)

    # 2) Mean runtime vs n, per method (log y -- the MILP's exponential wall).
    by_key: dict[tuple[str, int], list[float]] = defaultdict(list)
    for r in records:
        by_key[(r.method, r.n)].append(r.runtime_s)
    fig, ax = plt.subplots(figsize=(6, 4))
    for method in ("milp", "heuristic", "sa", "hybrid_sa"):
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

    # 3) Mean objective vs endurance, split by hub (MILP where available, else SA).
    ref_method = "milp" if any(r.method == "milp" for r in records) else "sa"
    by_eh: dict[tuple[str, float], list[float]] = defaultdict(list)
    for r in records:
        if r.method == ref_method and r.objective is not None:
            by_eh[(r.hub, r.endurance)].append(r.objective)
    if by_eh:
        hubs = sorted({h for (h, _) in by_eh})
        fig, ax = plt.subplots(figsize=(6, 4))
        for hub in hubs:
            pts = sorted((e, statistics.mean(v)) for (h, e), v in by_eh.items() if h == hub)
            xs, ys = zip(*pts, strict=True)
            ax.plot(xs, ys, marker="o", label=f"{hub} hub")
        ax.set_xlabel("drone endurance D_tl (s)")
        ax.set_ylabel(f"mean completion time ({ref_method})")
        ax.set_title("Effect of endurance and hub position")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "endurance_hub.png", dpi=150)
        plt.close(fig)

    # 4) Run-to-run spread of SA and Hybrid SA, averaged by size (Rafael's spec):
    # for each instance take the mean completion time over its seeds, express each
    # run's max/min as a % deviation from that mean, then average those %s over all
    # instances of a size -- so the sizes can be compared. One series per method.
    spread_methods = [m for m in ("sa", "hybrid_sa") if any(r.method == m for r in records)]
    if spread_methods:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        styles = {"sa": ("C0", "SA"), "hybrid_sa": ("C1", "Hybrid SA")}
        all_ns: set[int] = set()
        for mi, method in enumerate(spread_methods):
            runs: dict[tuple, list[float]] = defaultdict(list)
            for r in records:
                if r.method == method and r.objective is not None:
                    runs[(r.n, r.replication, r.endurance, r.hub)].append(r.objective)
            by_n_max: dict[int, list[float]] = defaultdict(list)
            by_n_min: dict[int, list[float]] = defaultdict(list)
            for (n, *_), objs in runs.items():
                mean = statistics.mean(objs)
                if len(objs) < 2 or mean <= 0:
                    continue
                by_n_max[n].append((max(objs) - mean) / mean * 100)
                by_n_min[n].append((min(objs) - mean) / mean * 100)
            ns = sorted(by_n_max)
            if not ns:
                continue
            all_ns.update(ns)
            up = [statistics.mean(by_n_max[n]) for n in ns]
            lo = [statistics.mean(by_n_min[n]) for n in ns]
            color, label = styles.get(method, ("C2", method))
            xs = [n + (mi - 0.5) * 0.6 for n in ns]  # offset so the two methods don't overlap
            ax.errorbar(xs, [0] * len(ns), yerr=[[-v for v in lo], up], fmt="o",
                        capsize=5, lw=1.5, color=color, label=label)
        ax.axhline(0, color="grey", lw=0.8, ls="--")
        ax.set_xlabel("number of customers n")
        ax.set_ylabel("avg % deviation from per-instance mean\n(min .. max over seeds)")
        ax.set_title("Run-to-run spread of SA / Hybrid SA, averaged by size")
        if all_ns:
            ax.set_xticks(sorted(all_ns))
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "sa_spread.png", dpi=150)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / "results.csv"
    sizes = _ints(args.sizes)
    reps = tuple(range(1, args.reps + 1))
    hubs = tuple(h.strip() for h in args.hubs.split(",") if h.strip())
    methods = tuple(m.strip() for m in args.methods.split(",") if m.strip())

    # Dump the instance coordinates (deterministic; cheap to always refresh).
    n_coords = write_instances_csv(
        args.out / "instances.csv", sizes=sizes, replications=reps, hubs=tuple(hubs)
    )
    print(f"instances -> {args.out}/instances.csv ({n_coords} coordinate rows)")

    # Resume: load existing rows and skip the keys already done.
    existing: list[RunRecord] = []
    if csv_path.exists() and not args.fresh:
        existing = read_csv(csv_path)
        print(f"resuming: {len(existing)} rows already in {csv_path}")
    elif args.fresh and csv_path.exists():
        csv_path.unlink()
    skip = done_keys(existing)

    env = None
    if "milp" in methods:
        try:
            from fstsp.gurobi_env import make_env

            env = make_env()
        except Exception as exc:  # any licence/import failure -> skip MILP
            print(f"warning: no Gurobi env ({exc}); skipping MILP rows")

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
        obj = f"{rec.objective:.0f}" if rec.objective is not None else "none"
        tag = f"opt={rec.proven_optimal}" if rec.method == "milp" else ""
        print(f"[{n_done:>4}] {rec.method:>10} n={rec.n:>2} r={rec.replication:>2} "
              f"{rec.hub:>6} E={rec.endurance:>6.0f} obj={obj:>8} "
              f"ns={rec.n_sorties} rt={rec.runtime_s:7.2f}s {tag}")

    try:
        run_suite(
            sizes=sizes,
            replications=reps,
            endurances=_floats(args.endurances),
            hubs=tuple(hubs),
            methods=methods,
            sa_iterations=args.sa_iters,
            sa_repetitions=args.sa_reps,
            hybrid_repetitions=args.hybrid_reps,
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

    records = read_csv(csv_path)
    print(f"\n{len(records)} rows total -> {csv_path}")

    # Deliverable for Rafael: the same rows plus a per-row gap_vs_milp column.
    deliverable = args.out / "results_with_gaps.csv"
    write_results_with_gaps(records, deliverable)
    print(f"deliverable (with per-row gap) -> {deliverable}")

    print_summary(records)

    if not args.no_plots:
        make_plots(records, args.out)
        print(f"\nfigures -> {args.out}/*.png")


if __name__ == "__main__":
    main()
