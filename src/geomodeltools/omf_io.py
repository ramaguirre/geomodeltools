"""Open Mining Format (OMF) v1 files: point sets, line sets and surfaces, written and read with the standard library and numpy.

OMF v1 is the format of the `omf` 1.x package (file version string ``OMF-v0.9.0``), which Leapfrog Geo, Micromine, Deswik and
others import. A file is a 60-byte header (magic number, version, the project's uid, where the JSON starts), then every array
zlib-compressed (little-endian float64 or int64), then a UTF-8 JSON index of every object keyed by uid. Coordinates are written as
given (real coordinates, origin 0). `omf` itself is not needed; the tests check that it opens what this module writes.
"""
from __future__ import annotations

import json
import struct
import uuid
import warnings
import zlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .shapes import LineSet, PointSet, Polyline, Surface

MAGIC = b"\x84\x83\x82\x81"
VERSION = b"OMF-v0.9.0"
HEADER_SIZE = 60
# a fixed colour per element, in order (Tableau 10); OMF requires one
COLOURS = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189), (140, 86, 75), (227, 119, 194),
           (127, 127, 127), (188, 189, 34), (23, 190, 207)]


class _Writer:
    def __init__(self, f):
        self.f = f
        self.index: dict = {}
        self.now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def add(self, cls: str, **fields) -> str:
        uid = str(uuid.uuid4())
        self.index[uid] = {"date_created": self.now, "date_modified": self.now, **fields, "__class__": cls}
        return uid

    def array(self, cls: str, a: np.ndarray, dtype: str) -> str:
        start = self.f.tell()
        self.f.write(zlib.compress(np.ascontiguousarray(a, dtype=dtype).tobytes()))
        return self.add(cls, array={"start": start, "dtype": dtype, "length": self.f.tell() - start})

    def element(self, e, colour) -> str:
        common = {"name": e.name, "description": "", "color": list(colour)}
        vertices = self.array("Vector3Array", e.vertices, "<f8")
        if isinstance(e, PointSet):
            data = [self.add("ScalarData", name=k, description="", array=self.array("ScalarArray", v, "<f8"), location="vertices")
                    for k, v in e.data.items()]
            geometry = self.add("PointSetGeometry", origin=[0.0, 0.0, 0.0], vertices=vertices)
            return self.add("PointSetElement", **common, data=data, textures=[], subtype="point", geometry=geometry)
        if isinstance(e, LineSet):
            geometry = self.add("LineSetGeometry", origin=[0.0, 0.0, 0.0], vertices=vertices,
                                segments=self.array("Int2Array", e.segments, "<i8"))
            return self.add("LineSetElement", **common, data=[], subtype="line", geometry=geometry)
        geometry = self.add("SurfaceGeometry", origin=[0.0, 0.0, 0.0], vertices=vertices,
                            triangles=self.array("Int3Array", e.triangles, "<i8"))
        return self.add("SurfaceElement", **common, data=[], textures=[], subtype="surface", geometry=geometry)


def write_omf(path, elements, *, name: str = "", description: str = "", units: str = "") -> Path:
    """Write `PointSet`, `LineSet`, `Surface` and `Polyline` objects to an OMF v1 file; returns the path.

    Each object is one OMF element, in order. A `Polyline` becomes a line set of its own; to put many polylines (drillhole traces,
    say) in one element, pass ``LineSet.from_polylines(name, polylines)``. ``name``, ``description`` and ``units`` (e.g. "m")
    describe the project.
    """
    path = Path(path)
    elements = [LineSet.from_polylines(e.name, [e]) if isinstance(e, Polyline) else e for e in elements]
    for e in elements:
        if not isinstance(e, (PointSet, LineSet, Surface)):
            raise TypeError(f"cannot write {type(e).__name__} to OMF; expected PointSet, LineSet, Surface or Polyline")
    project = uuid.uuid4()
    with open(path, "wb") as f:
        f.write(MAGIC + struct.pack("<32s", VERSION) + project.bytes + struct.pack("<Q", 0))
        w = _Writer(f)
        uids = [w.element(e, COLOURS[k % len(COLOURS)]) for k, e in enumerate(elements)]
        w.index[str(project)] = {"date_created": w.now, "date_modified": w.now, "name": name, "description": description,
                                 "author": "", "revision": "", "units": units, "elements": uids, "origin": [0.0, 0.0, 0.0],
                                 "__class__": "Project"}
        json_start = f.tell()
        f.write(json.dumps(w.index).encode("utf-8"))
        f.seek(HEADER_SIZE - 8)
        f.write(struct.pack("<Q", json_start))
    return path


def read_omf(path) -> list:
    """The point sets, line sets and triangulated surfaces of an OMF v1 file, in project order, as `PointSet`, `LineSet` and
    `Surface` objects in real coordinates (the project and geometry origins added). A point set keeps its scalar data on vertices;
    other data, and elements of other kinds (volumes, gridded surfaces), are left out with a warning."""
    raw = Path(path).read_bytes()
    if raw[:4] != MAGIC:
        raise ValueError(f"{path} is not an OMF file")
    version = struct.unpack("<32s", raw[4:36])[0].rstrip(b"\x00")
    if version != VERSION:
        raise ValueError(f"{path} is OMF {version.decode(errors='replace')}; only {VERSION.decode()} (OMF v1) is read")
    project = str(uuid.UUID(bytes=raw[36:52]))
    index = json.loads(raw[struct.unpack("<Q", raw[52:60])[0]:].decode("utf-8"))

    def array(uid, width):
        a = index[uid]["array"]
        if a["dtype"] not in ("<f8", "<i8"):
            raise ValueError(f"unsupported OMF array type {a['dtype']!r}")
        out = np.frombuffer(zlib.decompress(raw[a["start"]:a["start"] + a["length"]]), a["dtype"])
        return out.reshape(-1, width) if width else out

    origin = np.asarray(index[project].get("origin", [0.0, 0.0, 0.0]), float)
    out, left_out = [], []
    for uid in index[project]["elements"]:
        e = index[uid]
        cls, name = e["__class__"], e.get("name", "")
        g = index.get(e.get("geometry"), {})
        if g.get("__class__") not in ("PointSetGeometry", "LineSetGeometry", "SurfaceGeometry"):
            left_out.append(f"{name!r} ({g.get('__class__', cls)})")
            continue
        vertices = array(g["vertices"], 3) + origin + np.asarray(g.get("origin", [0.0, 0.0, 0.0]), float)
        if cls == "PointSetElement":
            data = {}
            for d in map(index.get, e.get("data", [])):
                if d["__class__"] == "ScalarData" and d.get("location") == "vertices":
                    data[d["name"]] = array(d["array"], 0)
                else:
                    left_out.append(f"data {d.get('name', '')!r} of {name!r} ({d['__class__']})")
            out.append(PointSet(name, vertices, data))
        elif cls == "LineSetElement":
            out.append(LineSet(name, vertices, array(g["segments"], 2)))
        else:
            out.append(Surface(name, vertices, array(g["triangles"], 3)))
        if cls != "PointSetElement" and e.get("data"):
            left_out.append(f"the data of {name!r}")
    if left_out:
        warnings.warn(f"left out of {Path(path).name}: {', '.join(left_out)}", stacklevel=2)
    return out
