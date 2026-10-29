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


def _indices(i, n_vertices: int, width: int, what: str) -> np.ndarray:
    i = np.array(i, dtype=np.int64).reshape(-1, width) if np.size(i) else np.empty((0, width), np.int64)
    if len(i) and (i.min() < 0 or i.max() >= n_vertices):
        raise ValueError(f"{what} refers to a vertex outside 0..{n_vertices - 1}")
    return i


@dataclass(eq=False)
class PointSet:
    """Named 3D points: ``vertices`` N x 3 float64, and ``data``, one value per point for each name (float64; NaN = no value)."""
    name: str
    vertices: np.ndarray
    data: dict = None

    def __post_init__(self):
        self.name = str(self.name)
        self.vertices = _vertices(self.vertices, f"point set {self.name!r}", 0)
        self.data = {str(k): np.array(v, dtype=np.float64).ravel() for k, v in (self.data or {}).items()}
        for k, v in self.data.items():
            if len(v) != len(self.vertices):
                raise ValueError(f"point set {self.name!r}: data {k!r} has {len(v)} values for {len(self.vertices)} points")


@dataclass(eq=False)
class LineSet:
    """Named 3D line segments: ``vertices`` N x 3 float64 and ``segments`` M x 2 int64 (vertex indices, from 0)."""
    name: str
    vertices: np.ndarray
    segments: np.ndarray

    def __post_init__(self):
        self.name = str(self.name)
        self.vertices = _vertices(self.vertices, f"line set {self.name!r}", 0)
        self.segments = _indices(self.segments, len(self.vertices), 2, f"line set {self.name!r}")

    @classmethod
    def from_polylines(cls, name: str, polylines) -> LineSet:
        """One line set holding several polylines (drillhole traces, say), each a chain of segments."""
        verts, segs, start = [], [], 0
        for p in polylines:
            n = len(p.vertices)
            verts.append(p.vertices)
            segs.append(start + np.column_stack([np.arange(n - 1), np.arange(1, n)]))
            start += n
        return cls(name, np.vstack(verts) if verts else np.empty((0, 3)), np.vstack(segs) if segs else np.empty((0, 2)))

    def to_polylines(self) -> list[Polyline]:
        """The segments as polylines: a run of segments each starting where the one before ends is one polyline. Named after the
        line set, with ``_1``, ``_2``, ... when there is more than one."""
        s = self.segments
        if not len(s):
            return []
        breaks = np.flatnonzero(s[1:, 0] != s[:-1, 1]) + 1
        runs = np.split(s, breaks)
        names = [self.name] if len(runs) == 1 else [f"{self.name}_{k}" for k in range(1, len(runs) + 1)]
        return [Polyline(n, self.vertices[np.append(r[:, 0], r[-1, 1])]) for n, r in zip(names, runs)]


@dataclass(eq=False)
class Surface:
    """A named triangulated surface: ``vertices`` N x 3 float64 and ``triangles`` M x 3 int64 (vertex indices, from 0)."""
    name: str
    vertices: np.ndarray
    triangles: np.ndarray

    def __post_init__(self):
        self.name = str(self.name)
        self.vertices = _vertices(self.vertices, f"surface {self.name!r}", 0)
        self.triangles = _indices(self.triangles, len(self.vertices), 3, f"surface {self.name!r}")
