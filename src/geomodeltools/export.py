"""Export polylines, line sets and surfaces to DXF and OBJ, and tables to CSV, in real coordinates.

DXF needs the optional extra ``[dxf]`` (ezdxf); OBJ and CSV need nothing beyond the base install. Polylines become DXF 3D polylines
and OBJ ``l`` lines; surfaces become DXF 3D faces and OBJ faces. A `LineSet` is written as its polylines (`LineSet.to_polylines`).
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .shapes import LineSet, Polyline, Surface


def _shapes(elements) -> list:
    out = []
    for e in elements:
        if isinstance(e, LineSet):
            out.extend(e.to_polylines())
        elif isinstance(e, (Polyline, Surface)):
            out.append(e)
        else:
            raise TypeError(f"cannot export {type(e).__name__}; expected Polyline, LineSet or Surface")
    return out


def dxf_layer_name(name: str) -> str:
    """``name`` as a valid DXF layer name: the characters DXF forbids (< > / \\ " : ; ? * | = `) become ``_``."""
    name = re.sub(r'[<>/\\":;?*|=`]', "_", str(name)).strip()
    return name or "0"


def write_dxf(path, elements, *, layer: str | None = None) -> Path:
    """Write polylines (3D polylines) and surfaces (one 3D face per triangle) to a DXF file; returns the path.

    Each element goes on the layer named after it (made valid with `dxf_layer_name`), or all on ``layer`` if given. Needs ezdxf:
    ``pip install geomodeltools[dxf]``.
    """
    try:
        import ezdxf
    except ModuleNotFoundError as e:
        raise ImportError("write_dxf needs the optional package 'ezdxf': pip install geomodeltools[dxf]") from e
    shapes = _shapes(elements)
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()
    for s in shapes:
        name = dxf_layer_name(layer if layer is not None else s.name)
        if name not in doc.layers:
            doc.layers.add(name)
        attribs = {"layer": name}
        if isinstance(s, Polyline):
            msp.add_polyline3d(s.vertices.tolist(), dxfattribs=attribs)
        else:
            for tri in s.vertices[s.triangles].tolist():
                msp.add_3dface(tri, dxfattribs=attribs)
    path = Path(path)
    doc.saveas(path)
    return path


def write_obj(path, elements) -> Path:
    """Write polylines (``l`` lines) and surfaces (``f`` faces) to a Wavefront OBJ file, one ``o`` object per element named after
    it (spaces become ``_``); returns the path. Coordinates are written in full (shortest exact decimal of each float64)."""
    shapes = _shapes(elements)
    lines, start = [], 1
    for s in shapes:
        lines.append("o " + (re.sub(r"\s+", "_", s.name.strip()) or "object"))
        lines.extend("v " + " ".join(map(repr, v)) for v in s.vertices.tolist())
        if isinstance(s, Polyline):
            lines.append("l " + " ".join(str(start + k) for k in range(len(s.vertices))))
        else:
            lines.extend("f " + " ".join(str(start + k) for k in t) for t in s.triangles.tolist())
        start += len(s.vertices)
    path = Path(path)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_csv(path, table: pd.DataFrame, **kwargs) -> Path:
    """Write a table (`interval_table`, `point_table`, ...) to CSV with no index column, floats in full; returns the path.
    ``kwargs`` go to `pandas.DataFrame.to_csv` (``sep=";"``, say)."""
    path = Path(path)
    table.to_csv(path, index=False, **kwargs)
    return path
