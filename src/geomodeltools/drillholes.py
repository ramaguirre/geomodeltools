"""Drillhole tables and traces in real coordinates, built on `desurvey` (numpy and pandas only).

`interval_table` and `point_table` are the depth tables a 3D viewer or an export needs: one row per interval (or per point), with
its position along the desurveyed hole; `traces` draws each hole as a `Polyline`. Column detection, the dip flag (``dip_positive_down``, True by default) and the handling of
holes that cannot be placed are those of `desurvey_intervals`, which both wrap.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .columns import find_column
from .desurvey import (DEPTH_CANDIDATES, HOLE_CANDIDATES, _prepare, _Prepared, _warn_left_out, desurvey, desurvey_intervals,
                       trace_depths)
from .shapes import Polyline

INTERVAL_POSITIONS = {"start": "TOP", "mid": "MID", "end": "BOTTOM"}


def _positions(out: pd.DataFrame) -> dict:
    return {f"{label}_{ax.upper()}": out[f"{name}_{ax}"] for name, label in INTERVAL_POSITIONS.items() for ax in "xyz"}


def interval_table(collars: pd.DataFrame, surveys: pd.DataFrame | None, intervals: pd.DataFrame, *,
                   hole: str | None = None, xyz: Sequence[str] | None = None, length: str | None = None,
                   at: str | None = None, azimuth: str | None = None, dip: str | None = None,
                   from_: str | None = None, to: str | None = None, attributes: Sequence[str] | None = None,
                   dip_positive_down: bool = True, duplicates="skip", warn: bool = True) -> pd.DataFrame:
    """An interval table in real coordinates: HOLEID, FROM, TO, the interval columns (all others, or ``attributes``), then
    TOP_X/Y/Z (at FROM), MID_X/Y/Z (half way along the hole, on its arc) and BOTTOM_X/Y/Z (at TO).

    Arguments are those of `desurvey_intervals`, which this wraps: column names are found automatically unless given, and
    ``dip_positive_down`` is True by default (pass False for tables that store dip negative-down). Holes that cannot be placed
    are left out with a warning; call `desurvey_intervals` itself for the `.skipped` and `.report` tables.
    """
    d = desurvey_intervals(collars, surveys, intervals, hole=hole, xyz=xyz, length=length, at=at, azimuth=azimuth, dip=dip,
                           from_=from_, to=to, attributes=attributes, dip_positive_down=dip_positive_down,
                           duplicates=duplicates, warn=warn, stacklevel=3)
    out = d.intervals
    table = out.iloc[:, :out.columns.get_loc("_row")]          # HOLEID, FROM, TO and the interval columns, then _row, positions
    return pd.concat([table, pd.DataFrame(_positions(out))], axis=1)


def point_table(collars: pd.DataFrame, surveys: pd.DataFrame | None, points: pd.DataFrame, *,
                hole: str | None = None, xyz: Sequence[str] | None = None, length: str | None = None,
                at: str | None = None, azimuth: str | None = None, dip: str | None = None, depth: str | None = None,
                attributes: Sequence[str] | None = None, dip_positive_down: bool = True, duplicates="skip",
                warn: bool = True) -> pd.DataFrame:
    """A point (depth) table in real coordinates: HOLEID, DEPTH, the point columns (all others, or ``attributes``), then X, Y, Z.

    ``points`` has a hole id and a depth along the hole (``depth``; found among DEPTH, AT, MD, ... unless given). ``at`` is the
    survey's depth column, as in `desurvey_intervals`; the other arguments are that function's too, including
    ``dip_positive_down`` (True by default). Holes that cannot be placed are left out with a warning.
    """
    p_hole = find_column(points, hole, HOLE_CANDIDATES, "point hole id")
    p_depth = find_column(points, depth, DEPTH_CANDIDATES, "point depth")
    extra = list(attributes) if attributes is not None else [c for c in points.columns if c not in (p_hole, p_depth)]
    d = points[p_depth].to_numpy(float)
    # a point is an interval of no length: its start is its position
    as_intervals = pd.concat([points[[p_hole]].reset_index(drop=True),
                              pd.DataFrame({"__FROM": d, "__TO": d}), points[extra].reset_index(drop=True)], axis=1)
    out = desurvey_intervals(collars, surveys, as_intervals, hole=hole, xyz=xyz, length=length, at=at, azimuth=azimuth, dip=dip,
                             from_="__FROM", to="__TO", attributes=extra, dip_positive_down=dip_positive_down,
                             duplicates=duplicates, warn=warn, stacklevel=3).intervals
    return pd.concat([out[["HOLEID"]], out["FROM"].rename("DEPTH"), out[extra],
                      out[["start_x", "start_y", "start_z"]].set_axis(["X", "Y", "Z"], axis=1)], axis=1)


def traces(collars: pd.DataFrame, surveys: pd.DataFrame | None = None, *, hole: str | None = None,
           xyz: Sequence[str] | None = None, length: str | None = None, at: str | None = None, azimuth: str | None = None,
           dip: str | None = None, dip_positive_down: bool = True, duplicates="skip", max_sag: float = 0.05,
           max_turn: float = 3.0, warn: bool = True) -> list[Polyline]:
    """One `Polyline` per hole, from the collar to the end of the hole, named by the hole id (as text), in collar order.

    The vertices are `trace_depths` (the collar, every survey station, the end, and extra vertices wherever the hole curves, so
    the polyline is within ``max_sag`` metres of the desurveyed arc), placed by `desurvey`. A hole's end is its collar length,
    else its deepest survey station. Column detection and ``dip_positive_down`` (True by default) are those of
    `desurvey_intervals`; holes that cannot be placed, or have no length, are left out with a warning.
    """
    prep = _prepare(collars, surveys, None, hole=hole, xyz=xyz, length=length, at=at, azimuth=azimuth, dip=dip, from_=None,
                    to=None, dip_positive_down=dip_positive_down, duplicates=duplicates)
    skipped = dict(prep.skipped)
    out = []
    for _, c in collars.drop_duplicates(prep.hole_c).iterrows():   # duplicated collar row: first wins
        hid = c[prep.hole_c]
        if hid in skipped:
            continue
        if hid in prep.stations:
            sd, az, dp = prep.stations[hid]
        else:                                                      # the collar's own azimuth/dip (checked in _prepare)
            sd, az, dp = np.array([0.0]), np.array([float(c[prep.c_az])]), np.array([prep.sign * float(c[prep.c_dip])])
        total = float(c[prep.c_len]) if prep.c_len is not None and pd.notna(c[prep.c_len]) else float(sd.max())
        if not np.isfinite(total) or total <= 0:
            skipped[hid] = "no hole length"
            continue
        depths = trace_depths(sd, az, dp, total, max_sag=max_sag, max_turn=max_turn)
        out.append(Polyline(str(hid), desurvey(c[list(prep.xyz_c)].to_numpy(float), sd, az, dp, depths)))
    if warn:
        _warn_left_out(_Prepared(**{**prep.__dict__, "skipped": skipped}))
    return out
