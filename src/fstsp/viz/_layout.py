"""Coordinate helpers for visualisation.

Most FSTSP instances arrive as distance matrices without explicit (x, y)
positions. Classical MDS reconstructs a 2D embedding that preserves pairwise
distances as well as possible — good enough for visualisation.
"""

from __future__ import annotations

import numpy as np

from fstsp.instance import Instance


def classical_mds(D: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Classical multidimensional scaling for a symmetric distance matrix.

    Returns an (n, n_components) array of coordinates. Distances in the
    embedding approximate the input distances; orientation is arbitrary.
    """
    n = D.shape[0]
    D2 = D**2
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D2 @ J
    eigvals, eigvecs = np.linalg.eigh(B)
    order = np.argsort(eigvals)[::-1][:n_components]
    return eigvecs[:, order] * np.sqrt(np.maximum(eigvals[order], 0.0))


def coords_for(inst: Instance, coords: np.ndarray | None = None) -> np.ndarray:
    """Resolve (n, 2) coords for an Instance.

    Precedence: explicit `coords` argument > `inst.coords` > classical-MDS
    embedding of the truck travel-time matrix.
    """
    if coords is not None:
        coords = np.asarray(coords, dtype=float)
        if coords.shape != (inst.n_nodes, 2):
            raise ValueError(f"coords shape {coords.shape} != ({inst.n_nodes}, 2)")
        return coords
    if inst.coords is not None:
        return np.asarray(inst.coords, dtype=float)
    return classical_mds(inst.t)
