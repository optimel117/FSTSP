"""Reference instances from the thesis. Keep these stable so they can be used as tests."""

from __future__ import annotations

import numpy as np

from fstsp.instance import Instance


def thesis_4_1_2() -> tuple[Instance, list[int]]:
    """The §4.1.2 worked example.

    Nodes: 0 = depot D, 1..6 = customers. Truck travel times are read directly
    from the distance matrix in Figure 4.1b. The drone is twice as fast as the
    truck (d_ij = t_ij / 2), endurance e = 5, and SL = SR = 0.2.

    The initial truck-only TSP (Figure 4.2) is D - 3 - 6 - 5 - 1 - 2 - 4 - D
    with total time 16.9.

    Returns (instance, initial_tsp_route).
    """
    # Symmetric matrix; rows/cols 0..6 correspond to D, 1, 2, 3, 4, 5, 6.
    t = np.array(
        [
            [0.0, 4.4, 4.9, 3.4, 3.0, 5.8, 5.5],
            [4.4, 0.0, 2.0, 3.0, 1.5, 1.5, 2.5],
            [4.9, 2.0, 0.0, 4.9, 2.3, 2.8, 4.5],
            [3.4, 3.0, 4.9, 0.0, 2.6, 3.9, 2.4],
            [3.0, 1.5, 2.3, 2.6, 0.0, 3.0, 3.4],
            [5.8, 1.5, 2.8, 3.9, 3.0, 0.0, 2.3],
            [5.5, 2.5, 4.5, 2.4, 3.4, 2.3, 0.0],
        ]
    )
    instance = Instance.from_truck_matrix(
        t,
        depot=0,
        drone_speed_ratio=2.0,
        drone_endurance=5.0,
        sl=0.2,
        sr=0.2,
    )
    initial_route = [0, 3, 6, 5, 1, 2, 4, 0]
    return instance, initial_route
