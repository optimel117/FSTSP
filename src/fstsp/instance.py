from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Instance:
    """An FSTSP instance.

    Nodes are integer ids 0..N-1 where 0 is the depot. The truck starts and ends
    at the depot. Customers are nodes 1..N-1 by default but `customers` can be
    overridden if the instance includes drone-ineligible nodes.
    """

    depot: int
    customers: tuple[int, ...]
    t: np.ndarray
    d: np.ndarray
    drone_endurance: float
    sl: float
    sr: float

    def __post_init__(self) -> None:
        n = self.t.shape[0]
        if self.t.shape != (n, n) or self.d.shape != (n, n):
            raise ValueError("travel-time matrices must be square and same shape")
        if self.depot in self.customers:
            raise ValueError("depot must not appear in customers")

    @property
    def n_nodes(self) -> int:
        return self.t.shape[0]

    @classmethod
    def from_truck_matrix(
        cls,
        t: np.ndarray,
        *,
        depot: int = 0,
        drone_speed_ratio: float = 2.0,
        drone_endurance: float,
        sl: float,
        sr: float,
        customers: tuple[int, ...] | None = None,
    ) -> Instance:
        """Build an instance from a truck travel-time matrix.

        Drone times default to t / drone_speed_ratio (matches the §4.1.2 example
        where the drone travels twice as fast as the truck).
        """
        d = t / drone_speed_ratio
        if customers is None:
            customers = tuple(i for i in range(t.shape[0]) if i != depot)
        return cls(
            depot=depot,
            customers=customers,
            t=t.astype(float),
            d=d.astype(float),
            drone_endurance=drone_endurance,
            sl=sl,
            sr=sr,
        )
