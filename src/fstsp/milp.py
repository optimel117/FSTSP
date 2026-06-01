"""Exact Boccia-style FSTSP MILP, solved with Gurobi.

This is the formulation written up in §3 of the thesis (variables y, x, gamma,
theta, omega, delta, sigma), ported from ``legacy/milp_v0.py`` to operate on a
package :class:`~fstsp.instance.Instance` and return a
:class:`~fstsp.solution.Solution`.

The formulation uses a *split depot*: a start depot ``s`` and an end depot ``t``
that are two copies of the single physical depot. We map the instance's depot
to both: travel to/from the synthetic end-depot id reuses the depot's row/column
in the travel-time matrices, and on extraction the end depot collapses back to
the real depot so the truck route starts and ends at it.

The subtour-elimination constraints are enumerated over all customer subsets, so
this is only practical for the small instances the thesis studies (a dozen-ish
customers).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import gurobipy as gp
from gurobipy import GRB

from fstsp.gurobi_env import make_env
from fstsp.instance import Instance
from fstsp.solution import Solution, Sortie

_STATUS_NAMES = {
    GRB.OPTIMAL: "optimal",
    GRB.TIME_LIMIT: "time_limit",
    GRB.INFEASIBLE: "infeasible",
    GRB.INF_OR_UNBD: "infeasible_or_unbounded",
    GRB.UNBOUNDED: "unbounded",
    GRB.INTERRUPTED: "interrupted",
}


@dataclass
class MilpResult:
    """Outcome of an exact solve."""

    solution: Solution | None
    objective: float | None
    bound: float | None
    gap: float | None
    runtime: float
    status: str

    @property
    def is_optimal(self) -> bool:
        return self.status == "optimal"


def solve_milp(
    instance: Instance,
    *,
    time_limit: float | None = None,
    mip_gap: float = 0.0,
    seed: int = 1,
    env: gp.Env | None = None,
    verbose: bool = False,
) -> MilpResult:
    """Solve the FSTSP exactly for ``instance``.

    Parameters
    ----------
    time_limit:
        Wall-clock limit in seconds (None = no limit).
    mip_gap:
        Relative MIP gap to stop at (0.0 = prove optimality).
    seed:
        Gurobi seed, for reproducibility.
    env:
        A started Gurobi environment to reuse. When None, a WLS/local env is
        created from credentials (see :func:`fstsp.gurobi_env.make_env`) and
        disposed before returning.
    verbose:
        Stream the Gurobi log.

    Returns a :class:`MilpResult`. ``solution`` is None when no feasible solution
    was found.
    """
    owns_env = env is None
    if env is None:
        env = make_env(verbose=verbose)
    try:
        return _solve(
            instance,
            env=env,
            time_limit=time_limit,
            mip_gap=mip_gap,
            seed=seed,
            verbose=verbose,
        )
    finally:
        if owns_env:
            env.dispose()


def _solve(
    instance: Instance,
    *,
    env: gp.Env,
    time_limit: float | None,
    mip_gap: float,
    seed: int,
    verbose: bool,
) -> MilpResult:
    s = instance.depot
    C = list(instance.customers)
    t = instance.end_depot  # second copy of the depot that closes the truck route
    V = [s, *C, t]

    SL = instance.sl
    SR = instance.sr
    D_tl = instance.drone_endurance

    # A = {(i,j): i in {s} u C, j in C u {t}, i != j} excluding the direct s->t arc
    A = [
        (i, j)
        for i in [s, *C]
        for j in [*C, t]
        if i != j and not (i == s and j == t)
    ]
    truck_time = {(i, j): instance.truck_time(i, j) for (i, j) in A}
    drone_time = {(i, j): instance.drone_time(i, j) for (i, j) in A}

    out_arcs: dict[int, list[tuple[int, int]]] = {i: [] for i in V}
    in_arcs: dict[int, list[tuple[int, int]]] = {j: [] for j in V}
    for i, j in A:
        out_arcs[i].append((i, j))
        in_arcs[j].append((i, j))

    model = gp.Model("fstsp_boccia", env=env)

    # ---- decision variables -------------------------------------------------
    y = model.addVars(A, vtype=GRB.BINARY, name="y")  # truck arc used
    x = model.addVars(A, vtype=GRB.BINARY, name="x")  # drone arc used
    gamma = model.addVars(C, A, vtype=GRB.BINARY, name="gamma")  # truck arc during sortie h
    theta = model.addVars(C, vtype=GRB.BINARY, name="theta")  # customer drone-served
    omega = model.addVars(C, V, vtype=GRB.BINARY, name="omega")  # launch node of sortie h
    delta = model.addVars(C, V, vtype=GRB.BINARY, name="delta")  # rendezvous node of sortie h
    sigma = model.addVars(C, lb=0.0, vtype=GRB.CONTINUOUS, name="sigma")  # truck wait

    # ---- objective (3.7) ----------------------------------------------------
    model.setObjective(
        gp.quicksum(truck_time[i, j] * y[i, j] for (i, j) in A)
        + gp.quicksum((SL + SR) * theta[h] for h in C)
        - gp.quicksum(SL * omega[h, s] for h in C)
        + gp.quicksum(sigma[h] for h in C),
        GRB.MINIMIZE,
    )

    # ---- truck routing (3.8)-(3.10) -----------------------------------------
    model.addConstr(gp.quicksum(y[s, j] for (_, j) in out_arcs[s]) == 1, name="truck_leaves_s")
    model.addConstr(gp.quicksum(y[i, t] for (i, _) in in_arcs[t]) == 1, name="truck_enters_t")
    for i in C:
        model.addConstr(
            gp.quicksum(y[i, j] for (_, j) in out_arcs[i])
            == gp.quicksum(y[j, i] for (j, _) in in_arcs[i]),
            name=f"truck_flow_balance_{i}",
        )
        model.addConstr(
            gp.quicksum(y[i, j] for (_, j) in out_arcs[i]) <= 1,
            name=f"truck_visit_once_{i}",
        )
    # subtour elimination over customer subsets (exponential; small instances only)
    for r in range(2, len(C) + 1):
        for subset in itertools.combinations(C, r):
            S = set(subset)
            inner = gp.quicksum(y[i, j] for i in S for j in S if (i, j) in A)
            for q in S:
                model.addConstr(
                    inner <= gp.quicksum(1 - theta[h] for h in S if h != q),
                    name=f"subtour_{'_'.join(map(str, sorted(S)))}_q{q}",
                )

    # ---- truck-drone linking (3.11)-(3.13) ----------------------------------
    for h in C:
        model.addConstr(
            gp.quicksum(gamma[h, s, j] for (_, j) in out_arcs[s]) == omega[h, s],
            name=f"gamma_start_{h}",
        )
        model.addConstr(
            gp.quicksum(gamma[h, i, t] for (i, _) in in_arcs[t]) == delta[h, t],
            name=f"gamma_end_{h}",
        )
        for i in C:
            model.addConstr(
                gp.quicksum(gamma[h, i, j] for (_, j) in out_arcs[i])
                - gp.quicksum(gamma[h, j, i] for (j, _) in in_arcs[i])
                == omega[h, i] - delta[h, i],
                name=f"gamma_flow_{h}_{i}",
            )

    # ---- assignment (3.14)-(3.17) -------------------------------------------
    for i, j in A:
        if i == s and j in C:
            model.addConstr(y[s, j] + x[s, j] <= 1, name=f"no_truck_drone_start_{j}")
        if j == t and i in C:
            model.addConstr(y[i, t] + x[i, t] <= 1, name=f"no_truck_drone_end_{i}")
    for i in C:
        for j in C:
            if i != j and (i, j) in A and (j, i) in A:
                model.addConstr(
                    y[i, j] + x[i, j] + x[j, i] <= 1, name=f"no_conflict_arc_{i}_{j}"
                )
    for h in C:
        model.addConstr(
            gp.quicksum(y[h, j] for (_, j) in out_arcs[h]) + theta[h] == 1,
            name=f"served_once_{h}",
        )

    # ---- consistency (3.18)-(3.23) ------------------------------------------
    for i, j in A:
        model.addConstr(
            gp.quicksum(gamma[h, i, j] for h in C) <= y[i, j], name=f"gamma_needs_y_{i}_{j}"
        )
    for h in C:
        model.addConstr(
            gp.quicksum(omega[h, i] for i in V if i not in (t, h)) == theta[h],
            name=f"one_launch_{h}",
        )
        model.addConstr(
            gp.quicksum(delta[h, j] for j in V if j not in (s, h)) == theta[h],
            name=f"one_rendezvous_{h}",
        )
        model.addConstr(omega[h, t] == 0, name=f"no_launch_at_t_{h}")
        model.addConstr(omega[h, h] == 0, name=f"no_launch_at_self_{h}")
        model.addConstr(delta[h, s] == 0, name=f"no_return_at_s_{h}")
        model.addConstr(delta[h, h] == 0, name=f"no_return_at_self_{h}")
    for i, j in A:
        rhs = gp.LinExpr()
        if i in C:
            rhs += theta[i]
        if j in C:
            rhs += theta[j]
        model.addConstr(x[i, j] <= rhs, name=f"x_needs_drone_cust_{i}_{j}")
    for i, j in A:
        # every arc touches {s} u C on one side and C u {t} on the other, so at
        # least one endpoint is always a customer -> rhs_terms is never empty.
        rhs_terms = []
        if j in C:
            rhs_terms.append(omega[j, i])
        if i in C:
            rhs_terms.append(delta[i, j])
        model.addConstr(x[i, j] <= gp.quicksum(rhs_terms), name=f"x_launch_or_return_{i}_{j}")
    for i in C:
        launch_for_other = gp.quicksum(omega[h, i] for h in C if h != i)
        model.addConstr(
            gp.quicksum(x[i, j] for (_, j) in out_arcs[i]) == launch_for_other + theta[i],
            name=f"drone_out_balance_{i}",
        )
        model.addConstr(launch_for_other + theta[i] <= 1, name=f"drone_out_single_{i}")
    for j in C:
        rendezvous_for_other = gp.quicksum(delta[h, j] for h in C if h != j)
        model.addConstr(
            gp.quicksum(x[i, j] for (i, _) in in_arcs[j]) == rendezvous_for_other + theta[j],
            name=f"drone_in_balance_{j}",
        )
        model.addConstr(rendezvous_for_other + theta[j] <= 1, name=f"drone_in_single_{j}")

    # ---- drone endurance (3.24)-(3.25) --------------------------------------
    for h in C:
        model.addConstr(
            gp.quicksum(truck_time[i, j] * gamma[h, i, j] for (i, j) in A)
            <= (D_tl - SR) * theta[h],
            name=f"truck_duration_sortie_{h}",
        )
        launch_leg = gp.quicksum(
            drone_time[i, h] * omega[h, i] for i in V if i not in (t, h) and (i, h) in A
        )
        return_leg = gp.quicksum(
            drone_time[h, j] * delta[h, j] for j in V if j not in (s, h) and (h, j) in A
        )
        model.addConstr(
            launch_leg + return_leg <= (D_tl - SR) * theta[h], name=f"drone_endurance_{h}"
        )

        # ---- waiting time (3.26) --------------------------------------------
        truck_during = gp.quicksum(truck_time[i, j] * gamma[h, i, j] for (i, j) in A)
        model.addConstr(
            launch_leg + return_leg - truck_during <= sigma[h], name=f"wait_{h}"
        )

    # ---- single-drone validity ----------------------------------------------
    for v in C:
        model.addConstr(
            theta[v]
            + gp.quicksum(omega[h, v] for h in C if h != v)
            + gp.quicksum(delta[h, v] for h in C if h != v)
            <= 1,
            name=f"node_drone_once_{v}",
        )

    # ---- solve --------------------------------------------------------------
    if time_limit is not None:
        model.Params.TimeLimit = time_limit
    model.Params.MIPGap = mip_gap
    model.Params.Seed = seed
    model.Params.OutputFlag = 1 if verbose else 0
    model.optimize()

    status = _STATUS_NAMES.get(model.Status, f"status_{model.Status}")
    runtime = model.Runtime

    if model.SolCount == 0:
        return MilpResult(
            solution=None,
            objective=None,
            bound=model.ObjBound if model.Status != GRB.INFEASIBLE else None,
            gap=None,
            runtime=runtime,
            status=status,
        )

    solution = _extract_solution(instance, s, t, V, A, y, theta, omega, delta)
    return MilpResult(
        solution=solution,
        objective=model.ObjVal,
        bound=model.ObjBound,
        gap=model.MIPGap,
        runtime=runtime,
        status=status,
    )


def _extract_solution(
    instance: Instance,
    s: int,
    t: int,
    V: list[int],
    A: list[tuple[int, int]],
    y: gp.tupledict,
    theta: gp.tupledict,
    omega: gp.tupledict,
    delta: gp.tupledict,
) -> Solution:
    """Rebuild a package Solution from the solved variables.

    The route runs from the start depot ``s`` to the end-depot ``t`` (==
    ``instance.end_depot``); both are first-class ids in the Solution, so a sortie
    may launch at ``s`` and/or rendezvous at ``t``.
    """
    next_node = {i: j for (i, j) in A if y[i, j].X > 0.5}
    route: list[int] = [s]
    cur = s
    while cur != t:
        cur = next_node[cur]
        route.append(cur)

    sorties: list[Sortie] = []
    for h in instance.customers:
        if theta[h].X <= 0.5:
            continue
        launch = next(i for i in V if i not in (t, h) and omega[h, i].X > 0.5)
        rendezvous = next(j for j in V if j not in (s, h) and delta[h, j].X > 0.5)
        sorties.append(Sortie(launch=launch, customer=h, rendezvous=rendezvous))

    return Solution(instance=instance, truck_route=route, sorties=sorties)
