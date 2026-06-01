from fstsp.heuristic import murray_chu
from fstsp.instance import Instance
from fstsp.instances import (
    initial_truck_solution,
    nearest_neighbour_route,
    random_euclidean,
    two_opt,
)
from fstsp.sa import SAResult, SATrace, simulated_annealing
from fstsp.solution import Solution, Sortie
from fstsp.validate import FeasibilityError, is_feasible, validate

__all__ = [
    "FeasibilityError",
    "Instance",
    "SAResult",
    "SATrace",
    "Solution",
    "Sortie",
    "initial_truck_solution",
    "is_feasible",
    "murray_chu",
    "nearest_neighbour_route",
    "random_euclidean",
    "simulated_annealing",
    "two_opt",
    "validate",
]

# The exact solver needs gurobipy (the optional "exact" extra). Expose it at the
# top level when available, but let `import fstsp` work without it so the
# heuristic/SA path runs on machines with numpy only.
try:
    from fstsp.milp import MilpResult, solve_milp
except ModuleNotFoundError:
    pass
else:
    __all__ += ["MilpResult", "solve_milp"]
