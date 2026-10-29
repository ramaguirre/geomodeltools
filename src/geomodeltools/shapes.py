"""Plain 3D shapes in real coordinates, for exports and viewers (numpy only).

Coordinates are float64 X = east, Y = north, Z = elevation, never shifted to a local origin: float64 keeps well under a
millimetre at UTM magnitudes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _vertices(v, what: str, at_least: int) -> np.ndarray:
    v = np.array(v, dtype=np.float64)                                  # a copy: the shape owns its vertices
    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError(f"{what} vertices must be N x 3, got shape {v.shape}")
    if len(v) < at_least:
        raise ValueError(f"{what} needs at least {at_least} vertices, got {len(v)}")
    return v


@dataclass(eq=False)
class Polyline:
    """A named 3D polyline: ``vertices`` is N x 3 float64 (N >= 2), in order along the line."""
    name: str
    vertices: np.ndarray

    def __post_init__(self):
        self.name = str(self.name)
        self.vertices = _vertices(self.vertices, f"polyline {self.name!r}", 2)

    @property
    def length(self) -> float:
        """Total length along the vertices."""
        return float(np.linalg.norm(np.diff(self.vertices, axis=0), axis=1).sum())
