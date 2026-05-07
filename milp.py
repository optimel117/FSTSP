# ============================================================
# Boccia-style FSTSP MILP
# Based on the formulation written in the thesis:
# variables y, x, gamma, theta, omega, delta, sigma
# ============================================================

import math
import itertools
import gurobipy as gp
from gurobipy import GRB


# ============================================================
# 1. INSTANCE DATA
# ============================================================

# Customers are 1,...,n
# s = 0 is the start depot
# t = n+1 is the end depot

coords = {
    0: (0.0, 0.0),    # start depot s
    1: (4.0, 3.0),
    2: (8.0, 2.0),
    3: (3.0, 7.0),
    4: (7.0, 8.0),
    5: (10.0, 5.0),
    6: (5.0, 10.0),
    7: (12.0, 3.0),
    8: (14.0, 7.0),
    9: (9.0, 11.0),
    10: (2.0, 12.0),
    11: (6.0, 14.0),
    12: (13.0, 12.0),
    13: (16.0, 4.0),
    14: (18.0, 9.0),
    15: (11.0, 15.0),
    16: (0.0, 0.0),  # end depot t
}

n = len(coords) - 2
C = list(range(1, n + 1))
s = 0
t = n + 1
V = [s] + C + [t]

truck_speed = 1.0
drone_speed = 2.0

D_tl = 8.0
SL = 0.2
SR = 0.2


# ============================================================
# 2. TRAVEL TIMES
# ============================================================

def euclidean_distance(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


# A = {(i,j): i in C union {s}, j in C union {t}, i != j}
A = [
    (i, j)
    for i in [s] + C
    for j in C + [t]
    if i != j and not (i == s and j == t)
]

truck_time = {}
drone_time = {}

for i, j in A:
    dist = euclidean_distance(coords[i], coords[j])
    truck_time[i, j] = dist / truck_speed
    drone_time[i, j] = dist / drone_speed


# ============================================================
# 3. HELPER SETS
# ============================================================

def outgoing_arcs(i):
    return [(a, b) for (a, b) in A if a == i]


def incoming_arcs(j):
    return [(a, b) for (a, b) in A if b == j]


def powerset_customers():
    """
    Generates subsets S of C with size at least 2.
    Used for subtour elimination constraints.
    """
    for r in range(2, len(C) + 1):
        for subset in itertools.combinations(C, r):
            yield set(subset)


# ============================================================
# 4. CREATE MODEL
# ============================================================

model = gp.Model("Boccia_style_FSTSP")


# ============================================================
# 5. DECISION VARIABLES
# ============================================================

# y[i,j] = 1 if truck uses arc (i,j)
y = model.addVars(A, vtype=GRB.BINARY, name="y")

# x[i,j] = 1 if drone uses arc (i,j)
x = model.addVars(A, vtype=GRB.BINARY, name="x")

# gamma[h,i,j] = 1 if truck arc (i,j) is travelled during the sortie serving h
gamma = model.addVars(C, A, vtype=GRB.BINARY, name="gamma")

# theta[h] = 1 if customer h is served by drone
theta = model.addVars(C, vtype=GRB.BINARY, name="theta")

# omega[h,i] = 1 if node i is the launch node for sortie serving h
omega = model.addVars(C, V, vtype=GRB.BINARY, name="omega")

# delta[h,j] = 1 if node j is the rendezvous node for sortie serving h
delta = model.addVars(C, V, vtype=GRB.BINARY, name="delta")

# sigma[h] = truck waiting time for sortie h
sigma = model.addVars(C, lb=0.0, vtype=GRB.CONTINUOUS, name="sigma")


# ============================================================
# 6. OBJECTIVE FUNCTION
# Corresponds to thesis equation (3.7)
# ============================================================

model.setObjective(
    gp.quicksum(truck_time[i, j] * y[i, j] for (i, j) in A)
    + gp.quicksum((SL + SR) * theta[h] for h in C)
    - gp.quicksum(SL * omega[h, s] for h in C)
    + gp.quicksum(sigma[h] for h in C),
    GRB.MINIMIZE
)


# ============================================================
# 7. TRUCK ROUTING CONSTRAINTS
# Corresponds to equations (3.8), (3.9), (3.10)
# ============================================================

# (3.8) Truck leaves s once and enters t once
model.addConstr(
    gp.quicksum(y[s, j] for j in C if (s, j) in A) == 1,
    name="truck_leaves_start_depot"
)

model.addConstr(
    gp.quicksum(y[i, t] for i in C if (i, t) in A) == 1,
    name="truck_enters_end_depot"
)

# (3.9) Flow conservation for each customer
for i in C:
    model.addConstr(
        gp.quicksum(y[i, j] for (i2, j) in outgoing_arcs(i))
        ==
        gp.quicksum(y[j, i] for (j, i2) in incoming_arcs(i)),
        name=f"truck_flow_balance_{i}"
    )

    model.addConstr(
        gp.quicksum(y[i, j] for (i2, j) in outgoing_arcs(i)) <= 1,
        name=f"truck_visit_at_most_once_{i}"
    )

# (3.10) Subtour elimination
# Written over customer subsets. This is exponential, but valid for small thesis instances.
for S in powerset_customers():
    for q in S:
        lhs = gp.quicksum(
            y[i, j]
            for i in S
            for j in S
            if (i, j) in A
        )

        rhs = gp.quicksum(
            1 - theta[h]
            for h in S
            if h != q
        )

        model.addConstr(
            lhs <= rhs,
            name=f"subtour_elim_S_{'_'.join(map(str, sorted(S)))}_q_{q}"
        )


# ============================================================
# 8. TRUCK-DRONE LINKING CONSTRAINTS
# Corresponds to equations (3.11), (3.12), (3.13)
# ============================================================

for h in C:

    # (3.11)
    model.addConstr(
        gp.quicksum(gamma[h, s, j] for j in C if (s, j) in A)
        == omega[h, s],
        name=f"gamma_start_flow_h_{h}"
    )

    # (3.12)
    model.addConstr(
        gp.quicksum(gamma[h, i, t] for i in C if (i, t) in A)
        == delta[h, t],
        name=f"gamma_end_flow_h_{h}"
    )

    # (3.13)
    for i in C:
        model.addConstr(
            gp.quicksum(gamma[h, i, j] for (i2, j) in outgoing_arcs(i))
            -
            gp.quicksum(gamma[h, j, i] for (j, i2) in incoming_arcs(i))
            ==
            omega[h, i] - delta[h, i],
            name=f"gamma_flow_h_{h}_node_{i}"
        )


# ============================================================
# 9. ASSIGNMENT CONSTRAINTS
# Corresponds to equations (3.14), (3.15), (3.16), (3.17)
# ============================================================

# (3.14)
for j in C:
    if (s, j) in A:
        model.addConstr(
            y[s, j] + x[s, j] <= 1,
            name=f"no_truck_and_drone_start_arc_{j}"
        )

# (3.15)
for i in C:
    if (i, t) in A:
        model.addConstr(
            y[i, t] + x[i, t] <= 1,
            name=f"no_truck_and_drone_end_arc_{i}"
        )

# (3.16)
for i in C:
    for j in C:
        if i != j and (i, j) in A and (j, i) in A:
            model.addConstr(
                y[i, j] + x[i, j] + x[j, i] <= 1,
                name=f"no_conflicting_customer_arc_{i}_{j}"
            )

# (3.17) Each customer is served either by truck or by drone
for h in C:
    model.addConstr(
        gp.quicksum(y[h, j] for (h2, j) in outgoing_arcs(h))
        + theta[h]
        == 1,
        name=f"customer_served_once_{h}"
    )


# ============================================================
# 10. CONSISTENCY CONSTRAINTS
# Corresponds to equations (3.18), (3.19), (3.20), (3.21),
# (3.22), (3.23)
# ============================================================

# (3.18) gamma can only use truck arcs actually selected by y
for i, j in A:
    model.addConstr(
        gp.quicksum(gamma[h, i, j] for h in C) <= y[i, j],
        name=f"gamma_only_if_truck_arc_{i}_{j}"
    )

# (3.19) If h is drone-served, exactly one launch and one rendezvous
for h in C:
    model.addConstr(
        gp.quicksum(omega[h, i] for i in V if i not in [t, h]) == theta[h],
        name=f"one_launch_if_drone_served_{h}"
    )

    model.addConstr(
        gp.quicksum(delta[h, j] for j in V if j not in [s, h]) == theta[h],
        name=f"one_rendezvous_if_drone_served_{h}"
    )

    # Explicitly forbid invalid launch/rendezvous positions
    model.addConstr(omega[h, t] == 0, name=f"no_launch_at_end_depot_h_{h}")
    model.addConstr(omega[h, h] == 0, name=f"no_launch_at_served_customer_h_{h}")
    model.addConstr(delta[h, s] == 0, name=f"no_return_at_start_depot_h_{h}")
    model.addConstr(delta[h, h] == 0, name=f"no_return_at_served_customer_h_{h}")

# (3.20)
# x_ij can only exist if at least one endpoint is drone-served.
# Need to handle depot cases because theta is only defined for customers.
for i, j in A:
    rhs = 0

    if i in C:
        rhs += theta[i]

    if j in C:
        rhs += theta[j]

    model.addConstr(
        x[i, j] <= rhs,
        name=f"x_only_related_to_drone_customer_{i}_{j}"
    )

# (3.21)
# x_ij <= omega_i^j + delta_j^i, with depot cases carefully handled.
# Interpretation:
# - arc i -> j can be a launch-to-customer arc if j is drone-served and i is its launch
# - or it can be a customer-to-rendezvous arc if i is drone-served and j is its rendezvous
for i, j in A:

    rhs_terms = []

    if j in C:
        rhs_terms.append(omega[j, i])

    if i in C:
        rhs_terms.append(delta[i, j])

    if rhs_terms:
        model.addConstr(
            x[i, j] <= gp.quicksum(rhs_terms),
            name=f"x_consistent_with_launch_or_return_{i}_{j}"
        )
    else:
        # This only occurs for impossible depot-depot arcs, which are not in A anyway.
        model.addConstr(
            x[i, j] <= 0,
            name=f"x_forbidden_{i}_{j}"
        )

# (3.22) Drone outgoing degree / single-use node restriction
for i in C:
    outgoing_x = gp.quicksum(x[i, j] for (i2, j) in outgoing_arcs(i))

    launch_from_i_for_other_customer = gp.quicksum(
        omega[h, i]
        for h in C
        if h != i
    )

    model.addConstr(
        outgoing_x == launch_from_i_for_other_customer + theta[i],
        name=f"drone_outgoing_balance_{i}"
    )

    model.addConstr(
        launch_from_i_for_other_customer + theta[i] <= 1,
        name=f"drone_outgoing_single_use_{i}"
    )

# (3.23) Drone incoming degree / single-use node restriction
for j in C:
    incoming_x = gp.quicksum(x[i, j] for (i, j2) in incoming_arcs(j))

    rendezvous_at_j_for_other_customer = gp.quicksum(
        delta[h, j]
        for h in C
        if h != j
    )

    model.addConstr(
        incoming_x == rendezvous_at_j_for_other_customer + theta[j],
        name=f"drone_incoming_balance_{j}"
    )

    model.addConstr(
        rendezvous_at_j_for_other_customer + theta[j] <= 1,
        name=f"drone_incoming_single_use_{j}"
    )


# ============================================================
# 11. DRONE ENDURANCE CONSTRAINTS
# Corresponds to equations (3.24), (3.25)
# ============================================================

for h in C:

    # (3.24) Truck path duration during sortie h
    model.addConstr(
        gp.quicksum(truck_time[i, j] * gamma[h, i, j] for (i, j) in A)
        <=
        (D_tl - SR) * theta[h],
        name=f"truck_duration_during_sortie_h_{h}"
    )

    # (3.25) Drone travel duration during sortie h
    drone_launch_leg = gp.quicksum(
        drone_time[i, h] * omega[h, i]
        for i in V
        if i != t and i != h and (i, h) in A
    )

    drone_return_leg = gp.quicksum(
        drone_time[h, j] * delta[h, j]
        for j in V
        if j != s and j != h and (h, j) in A
    )

    model.addConstr(
        drone_launch_leg + drone_return_leg
        <=
        (D_tl - SR) * theta[h],
        name=f"drone_endurance_h_{h}"
    )


# ============================================================
# 12. WAITING TIME CONSTRAINTS
# Corresponds to equation (3.26)
# ============================================================

for h in C:

    drone_launch_leg = gp.quicksum(
        drone_time[i, h] * omega[h, i]
        for i in V
        if i != t and i != h and (i, h) in A
    )

    drone_return_leg = gp.quicksum(
        drone_time[h, j] * delta[h, j]
        for j in V
        if j != s and j != h and (h, j) in A
    )

    truck_during_sortie = gp.quicksum(
        truck_time[i, j] * gamma[h, i, j]
        for (i, j) in A
    )

    model.addConstr(
        drone_launch_leg + drone_return_leg - truck_during_sortie
        <=
        sigma[h],
        name=f"truck_waiting_time_h_{h}"
    )


# ============================================================
# 13. EXTRA SINGLE-DRONE VALIDITY CHECKS
# These are not extra drones. They strengthen the single-UAV interpretation.
# They prevent a customer from being both launch/rendezvous and drone-served.
# ============================================================

for v in C:
    model.addConstr(
        theta[v]
        + gp.quicksum(omega[h, v] for h in C if h != v)
        + gp.quicksum(delta[h, v] for h in C if h != v)
        <= 1,
        name=f"node_used_by_drone_at_most_once_{v}"
    )


# ============================================================
# 14. SOLVER SETTINGS
# ============================================================

model.Params.TimeLimit = 3600
model.Params.MIPGap = 0.0
model.Params.OutputFlag = 1

# Optional: makes results more reproducible
model.Params.Seed = 1


# ============================================================
# 15. OPTIMIZE
# ============================================================

model.optimize()


# ============================================================
# 16. SOLUTION EXTRACTION
# ============================================================

def reconstruct_route(selected_arcs, start, end):
    route = [start]
    current = start

    while current != end:
        next_nodes = [j for (i, j) in selected_arcs if i == current]

        if len(next_nodes) == 0:
            print("Warning: route reconstruction stopped early.")
            break

        if len(next_nodes) > 1:
            print(f"Warning: multiple outgoing arcs from node {current}: {next_nodes}")

        current = next_nodes[0]
        route.append(current)

        if len(route) > len(V) + 5:
            print("Warning: possible cycle in route reconstruction.")
            break

    return route


if model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:

    completion_time = model.ObjVal
    best_bound = model.ObjBound
    gurobi_gap = model.MIPGap
    runtime = model.Runtime

    print("\n================================================")
    print("SOLUTION SUMMARY")
    print("================================================")

    print(f"Completion time / objective value: {completion_time:.4f}")
    print(f"Best lower bound: {best_bound:.4f}")
    print(f"Gurobi MIP gap: {100 * gurobi_gap:.4f}%")
    print(f"Runtime: {runtime:.2f} seconds")

    if abs(best_bound) > 1e-9:
        boccia_gap = ((completion_time - best_bound) / best_bound) * 100
        print(f"Boccia-style gap: {boccia_gap:.4f}%")
    else:
        print("Boccia-style gap: not available because lower bound is zero.")
   
    print("\n================================================")
    print("SOLUTION")
    print("================================================")

    if model.status == GRB.OPTIMAL:
        print("Status: OPTIMAL")
    else:
        print("Status: TIME LIMIT")

    print(f"Objective value: {model.ObjVal:.4f}")

    selected_truck_arcs = [
        (i, j)
        for (i, j) in A
        if y[i, j].X > 0.5
    ]

    selected_drone_arcs = [
        (i, j)
        for (i, j) in A
        if x[i, j].X > 0.5
    ]

    drone_customers = [
        h
        for h in C
        if theta[h].X > 0.5
    ]

    print("\nTruck arcs:")
    for i, j in selected_truck_arcs:
        print(f"  {i} -> {j}")

    truck_route = reconstruct_route(selected_truck_arcs, s, t)

    print("\nTruck route:")
    print("  " + " -> ".join(map(str, truck_route)))

    print("\nDrone arcs:")
    if selected_drone_arcs:
        for i, j in selected_drone_arcs:
            print(f"  {i} -> {j}")
    else:
        print("  No drone arcs used.")

    print("\nDrone-served customers:")
    if drone_customers:
        for h in drone_customers:
            print(f"  Customer {h}")
    else:
        print("  No customers served by drone.")

    print("\nDrone sorties:")
    for h in drone_customers:
        launch_nodes = [
            i for i in V
            if omega[h, i].X > 0.5
        ]

        rendezvous_nodes = [
            j for j in V
            if delta[h, j].X > 0.5
        ]

        if len(launch_nodes) == 1 and len(rendezvous_nodes) == 1:
            i = launch_nodes[0]
            j = rendezvous_nodes[0]
            drone_duration = drone_time[i, h] + drone_time[h, j] + SL + SR
            print(
                f"  Sortie serving {h}: {i} -> {h} -> {j} "
                f"| drone duration incl. service = {drone_duration:.4f} "
                f"| sigma = {sigma[h].X:.4f}"
            )
        else:
            print(f"  Warning: sortie for customer {h} has invalid launch/rendezvous.")

    print("\nGamma arcs by sortie:")
    for h in C:
        gamma_arcs = [
            (i, j)
            for (i, j) in A
            if gamma[h, i, j].X > 0.5
        ]

        if gamma_arcs:
            print(f"  During sortie serving {h}, truck uses:")
            for i, j in gamma_arcs:
                print(f"    {i} -> {j}")

    print("\nCustomer assignment:")
    for h in C:
        if theta[h].X > 0.5:
            print(f"  Customer {h}: drone")
        else:
            print(f"  Customer {h}: truck")

else:
    print("\nNo feasible solution found.")
    print(f"Model status: {model.status}")
    