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
    """A named 3D polyline: ``vertices`` is N x 3 float64 (N >= 2), in order along the line.

    Optionally a curve and sides, as drawn with Bezier handles (geoview's ``edit_polyline(curve=True, sides=True)``):

    - ``controls`` N x 2 x 3: the [incoming, outgoing] handle of each vertex. Segment k is the cubic Bezier curve through
      ``vertices[k]``, ``controls[k, 1]``, ``controls[k + 1, 0]``, ``vertices[k + 1]``; it passes through the vertices. ``None``
      (the default): straight segments. The first vertex's incoming and the last vertex's outgoing handle are not used.
    - ``normals`` N x 3: the side each vertex faces (e.g. the younger side of a contact). ``None``: no sides.

    Files that only hold straight segments (DXF, OBJ, the OMF geometry) get the curve as points along it (`curve_points`), so
    other software shows its true shape; OMF also keeps the handles and sides for geomodeltools to read back (`write_omf`).
    """
    name: str
    vertices: np.ndarray
    controls: np.ndarray | None = None
    normals: np.ndarray | None = None

    def __post_init__(self):
        self.name = str(self.name)
        self.vertices = _vertices(self.vertices, f"polyline {self.name!r}", 2)
        n = len(self.vertices)
        if self.controls is not None:
            c = np.array(self.controls, dtype=np.float64)
            if c.shape != (n, 2, 3):
                raise ValueError(f"polyline {self.name!r}: controls must be N x 2 x 3 = ({n}, 2, 3), got shape {c.shape}")
            self.controls = c
        if self.normals is not None:
            m = np.array(self.normals, dtype=np.float64)
            if m.shape != (n, 3):
                raise ValueError(f"polyline {self.name!r}: normals must be N x 3 = ({n}, 3), got shape {m.shape}")
            self.normals = m
        for what, a in (("vertices", self.vertices), ("controls", self.controls), ("normals", self.normals)):
            if a is not None and not np.isfinite(a).all():
                raise ValueError(f"polyline {self.name!r}: {what} must be finite")

    @classmethod
    def from_edit(cls, edit) -> Polyline:
        """A polyline from anything with ``name``, ``vertices`` and optionally ``controls`` / ``normals`` (geoview's
        ``PolylineEdit``, say), handles and sides included."""
        return cls(edit.name, edit.vertices, getattr(edit, "controls", None), getattr(edit, "normals", None))

    @property
    def curved(self) -> bool:
        """True if the polyline has Bezier handles (``controls``)."""
        return self.controls is not None

    def curve_points(self, step: float = 1.0) -> np.ndarray:
        """The line as points along it (M x 3): the vertices themselves if it is straight; for a curve, each segment cut into
        equal steps of its parameter, about ``step`` metres apart along the curve, every vertex included exactly."""
        if not self.curved:
            return self.vertices.copy()
        if step <= 0:
            raise ValueError("step must be > 0")
        v, c = self.vertices, self.controls
        out = [v[:1]]
        for k in range(len(v) - 1):
            seg = (v[k], c[k, 1], c[k + 1, 0], v[k + 1])
            n = max(1, int(np.ceil(_bezier_length(seg) / step)))
            out.append(_bezier(seg, np.arange(1, n + 1) / n))
        out = np.vstack(out)
        out[-1] = v[-1]
        return out

    @property
    def length(self) -> float:
        """Total length along the line (along the curve if it has handles)."""
        p = self.curve_points(0.05) if self.curved else self.vertices
        return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def _bezier(seg, t) -> np.ndarray:
    """Points of the cubic Bezier ``seg`` = (p0, c0, c1, p1) at parameters ``t``."""
    t = np.asarray(t, dtype=np.float64)[:, None]
    u = 1 - t
    p0, c0, c1, p1 = seg
    return u ** 3 * p0 + 3 * u * u * t * c0 + 3 * u * t * t * c1 + t ** 3 * p1


def _bezier_length(seg) -> float:
    return float(np.linalg.norm(np.diff(_bezier(seg, np.linspace(0, 1, 129)), axis=0), axis=1).sum())


def _indices(i, n_vertices: int, width: int, what: str) -> np.ndarray:
    i = np.array(i, dtype=np.int64).reshape(-1, width) if np.size(i) else np.empty((0, width), np.int64)
    if len(i) and (i.min() < 0 or i.max() >= n_vertices):
        raise ValueError(f"{what} refers to a vertex outside 0..{n_vertices - 1}")
    return i


def _segment_values(v) -> np.ndarray:
    """Per-segment values: float64 if they are all numbers, otherwise text (an object array of str)."""
    a = np.asarray(v).ravel()
    if a.dtype.kind in "biuf":
        return a.astype(np.float64)
    return np.array(["" if x is None else str(x) for x in a], dtype=object)


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
    """Named 3D line segments: ``vertices`` N x 3 float64 and ``segments`` M x 2 int64 (vertex indices, from 0). ``data``: one value
    per segment for each name, numbers (float64; NaN = no value) or text (e.g. the hole each trace segment belongs to)."""
    name: str
    vertices: np.ndarray
    segments: np.ndarray
    data: dict = None

    def __post_init__(self):
        self.name = str(self.name)
        self.vertices = _vertices(self.vertices, f"line set {self.name!r}", 0)
        self.segments = _indices(self.segments, len(self.vertices), 2, f"line set {self.name!r}")
        self.data = {str(k): _segment_values(v) for k, v in (self.data or {}).items()}
        for k, v in self.data.items():
            if len(v) != len(self.segments):
                raise ValueError(f"line set {self.name!r}: data {k!r} has {len(v)} values for {len(self.segments)} segments")

    @classmethod
    def from_polylines(cls, name: str, polylines, *, names_as: str | None = "name", curve_step: float = 1.0) -> LineSet:
        """One line set holding several polylines (drillhole traces, say), each a chain of segments. Each segment keeps the name of
        its polyline in ``data[names_as]`` (``None``: not kept), so the polylines can be told apart (e.g. ``names_as="HOLEID"``).
        A curved polyline enters as points along its curve, about ``curve_step`` m apart (`Polyline.curve_points`)."""
        polylines = list(polylines)
        verts, segs, labels, start = [], [], [], 0
        for p in polylines:
            pts = p.curve_points(curve_step)
            n = len(pts)
            verts.append(pts)
            segs.append(start + np.column_stack([np.arange(n - 1), np.arange(1, n)]))
            labels += [p.name] * (n - 1)
            start += n
        data = {names_as: np.array(labels, dtype=object)} if names_as else None
        return cls(name, np.vstack(verts) if verts else np.empty((0, 3)), np.vstack(segs) if segs else np.empty((0, 2)), data)

    def to_polylines(self, names_from: str | None = "name") -> list[Polyline]:
        """The segments as polylines: a run of segments each starting where the one before ends, and with the same text in
        ``data[names_from]`` if the line set has it, is one polyline, named by that text. Otherwise they are named after the line
        set, with ``_1``, ``_2``, ... when there is more than one."""
        s = self.segments
        if not len(s):
            return []
        labels = self.data.get(names_from) if names_from else None
        if labels is not None and labels.dtype != object:
            labels = None  # numbers do not name anything
        cut = s[1:, 0] != s[:-1, 1]
        if labels is not None:
            cut |= labels[1:] != labels[:-1]
        runs = np.split(np.arange(len(s)), np.flatnonzero(cut) + 1)
        if labels is not None:
            names = [str(labels[r[0]]) for r in runs]
        else:
            names = [self.name] if len(runs) == 1 else [f"{self.name}_{k}" for k in range(1, len(runs) + 1)]
        return [Polyline(n, self.vertices[np.append(s[r, 0], s[r[-1], 1])]) for n, r in zip(names, runs)]


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
