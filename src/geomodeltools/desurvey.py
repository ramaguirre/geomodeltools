"""Desurvey drillholes: collar / survey / interval tables to positions along the holes, by minimum curvature.

Moved here from geoview (2026-10-14) so that one Python implementation serves the notebooks, geoview and the browser tool
`geoplots_local` (contract: docs/desurvey_contract.md, golden data in tests/golden/desurvey/). numpy and pandas only.

Conventions
- Azimuth: degrees clockwise from grid north.
- Dip. The low-level functions (`directions`, `desurvey`, `trace_depths`, `dogleg_severity`) take the angle **below horizontal as a
  negative number** (-90 = straight down; the elevation angle). The table-level functions (`desurvey_intervals`, `check_tables`
  and the `_prepare` they share) take a flag, ``dip_positive_down``, **True by default here**: Leapfrog Geo writes dip positive-down
  in its own survey export, and so do many tables; set it False for tables that store negative-down dips (acQuire exports often do).
  geoview keeps False as its own default and passes it explicitly.
- Coordinates: X = east, Y = north, Z = elevation (up).

A hole that cannot be placed is **left out and reported, never guessed**, as Leapfrog Geo does: one with two
survey rows at the same depth is not desurveyed (unless you pick a deduplication rule), and one with no survey and no
azimuth/dip on the collar is not assumed to be vertical. `check_tables` lists the problems.

Desurvey is minimum curvature. Between two survey stations the hole follows a circular arc,
along which the direction turns at a constant rate with depth. So directions at intermediate
depths are spherical interpolations (slerp) of the station directions, and splitting a
segment at any depth reproduces the same arc exactly. On geoview's anonymised Centinela test set this reproduces Leapfrog Geo
2026.1.2's desurveyed intervals to under a quarter of a millimetre in every hole (notebook desurvey_vs_leapfrog_anonymised in geoview).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np
import pandas as pd

from .columns import XYZ_CANDIDATES, find_column, find_columns

HOLE_CANDIDATES = ["HOLEID", "HOLE_ID", "BHID", "DHID", "HOLE", "SONDAJE", "ID"]
DEPTH_CANDIDATES = ["DEPTH", "AT", "MD", "DISTANCE", "PROF", "PROFUNDIDAD"]
LENGTH_CANDIDATES = ["MAX_DEPTH", "MAXDEPTH", "LENGTH", "DEPTH", "EOH", "TD", "LARGO", "LONGITUD", "PROFUNDIDAD_TOTAL"]
AZIMUTH_CANDIDATES = ["AZIMUTH", "AZI", "AZ", "BRG", "BEARING", "AZIMUT", "RUMBO"]
DIP_CANDIDATES = ["DIP", "INCLINACION", "INCLINACIÓN", "INCL", "BUZAMIENTO", "BUZ"]
FROM_CANDIDATES = ["FROM", "MFROM", "DESDE"]
TO_CANDIDATES = ["TO", "MTO", "HASTA"]


def _collar_column(collars, given, candidates, what):
    """The collar's own direction column, or None. A name given for the *survey's* columns (``azimuth="BRG"``) may not exist in the
    collar table, which usually has no direction columns at all: then the usual names are tried instead of failing."""
    if given is not None and given in collars.columns:
        return given
    return find_column(collars, None, candidates, what, required=False)


def directions(azimuth, dip) -> np.ndarray:
    """Unit direction vectors (east, north, up) for azimuth/dip in degrees (dip < 0 = down)."""
    az, dp = np.radians(np.asarray(azimuth, float)), np.radians(np.asarray(dip, float))
    return np.stack([np.cos(dp) * np.sin(az), np.cos(dp) * np.cos(az), np.sin(dp)], axis=-1)


def _slerp(t1: np.ndarray, t2: np.ndarray, f: np.ndarray) -> np.ndarray:
    dot = np.clip(np.sum(t1 * t2, axis=-1), -1.0, 1.0)
    om = np.arccos(dot)[..., None]
    f = f[..., None]
    small = om < 1e-9
    so = np.where(small, 1.0, np.sin(om))
    out = np.where(small, t1 * (1 - f) + t2 * f, (np.sin((1 - f) * om) * t1 + np.sin(f * om) * t2) / so)
    return out / np.linalg.norm(out, axis=-1, keepdims=True)


def desurvey(collar, station_depths, azimuths, dips, depths) -> np.ndarray:
    """(len(depths), 3) positions along one hole, by minimum curvature.

    Before the first station the first direction applies; beyond the last, the last one.
    """
    collar = np.asarray(collar, float)
    sd = np.asarray(station_depths, float)
    order = np.argsort(sd, kind="stable")
    sd = sd[order]
    t = directions(np.asarray(azimuths, float)[order], np.asarray(dips, float)[order])
    depths = np.asarray(depths, float)
    q = np.unique(np.concatenate([[0.0], sd[sd > 0], depths]))  # every station and target, sorted
    q = q[q >= 0]
    # direction at each query depth: slerp between the stations that bracket it
    k = np.clip(np.searchsorted(sd, q, side="right") - 1, 0, len(sd) - 1)
    k1 = np.minimum(k + 1, len(sd) - 1)
    span = sd[k1] - sd[k]
    f = np.where(span > 0, np.clip((q - sd[k]) / np.where(span > 0, span, 1), 0, 1), 0.0)
    tq = _slerp(t[k], t[k1], f)
    # minimum curvature between consecutive query depths
    md = np.diff(q)
    a, b = tq[:-1], tq[1:]
    dl = np.arccos(np.clip(np.sum(a * b, axis=-1), -1.0, 1.0))
    rf = np.where(dl > 1e-9, 2 / np.where(dl > 1e-9, dl, 1) * np.tan(dl / 2), 1.0)
    steps = (md * rf / 2)[:, None] * (a + b)
    pos = collar + np.vstack([np.zeros(3), np.cumsum(steps, axis=0)])
    return pos[np.searchsorted(q, depths)]


def trace_depths(station_depths, azimuths, dips, length: float, *, max_sag: float = 0.05, max_turn: float = 3.0) -> np.ndarray:
    """Depths of the vertices that draw a hole as a polyline, as Leapfrog Geo places them in its trace export.

    The collar (0), every survey station inside the hole, and the end of the hole (``length``), plus
    equally spaced extra vertices inside any interval until each chord is within ``max_sag`` metres of the
    true arc and turns less than ``max_turn`` degrees. A straight interval gets none. The vertices all lie on the
    desurveyed hole, so the polyline is the hole to within ``max_sag``. Dips are negative downwards.
    """
    sd = np.asarray(station_depths, float)
    order = np.argsort(sd, kind="stable")
    sd = sd[order]
    t = directions(np.asarray(azimuths, float)[order], np.asarray(dips, float)[order])
    nodes = np.unique(np.concatenate([[0.0], sd[(sd > 0) & (sd < length)], [float(length)]]))
    tn = t[np.clip(np.searchsorted(sd, nodes, side="right") - 1, 0, len(sd) - 1)]   # direction at each node
    a, b = tn[:-1], tn[1:]
    theta = np.arctan2(np.linalg.norm(np.cross(a, b), axis=1), np.sum(a * b, axis=1))
    gap = np.diff(nodes)
    n_turn = np.ceil(theta / np.radians(max_turn) - 1e-9)
    # a chord of an arc of angle theta and length gap is max_sag off when theta / n = 4 asin(sqrt(max_sag * theta / (2 gap)))
    half = np.arcsin(np.minimum(1.0, np.sqrt(max_sag * theta / (2 * gap))))
    with np.errstate(divide="ignore", invalid="ignore"):
        n_sag = np.where(half > 0, np.ceil(theta / (4 * half) - 1e-9), 1.0)
    n = np.maximum(1, np.maximum(n_turn, n_sag)).astype(int)
    extra = n - 1
    k = np.arange(extra.sum()) - np.repeat(np.cumsum(extra) - extra, extra) + 1
    inner = np.repeat(nodes[:-1], extra) + k * np.repeat(gap / n, extra)
    return np.sort(np.concatenate([nodes, inner]))


def dogleg_severity(azimuths, dips, depths, per: float = 30.0) -> np.ndarray:
    """Dogleg severity at each survey station of one hole, in degrees per ``per`` metres.

    The angle between a station's direction and the previous station's, divided by the depth
    between them. ``per=30`` is Leapfrog's convention, and this reproduces its column. Stations are
    taken in depth order (a stable sort, as Leapfrog does) and the result comes back in the order
    given. The first station is 0. A station at the same depth as the one before has no defined
    severity: NaN. ``dips`` are negative downwards, as everywhere in geoview.
    """
    az, dp, md = (np.asarray(a, float) for a in (azimuths, dips, depths))
    out = np.empty(len(md))
    if len(md) == 0:
        return out
    order = np.argsort(md, kind="stable")
    t, d = directions(az[order], dp[order]), md[order]
    # atan2(|a x b|, a . b) stays accurate for the tiny turns that arccos(a . b) rounds to zero
    turn = np.degrees(np.arctan2(np.linalg.norm(np.cross(t[:-1], t[1:]), axis=-1), np.sum(t[:-1] * t[1:], axis=-1)))
    step = np.diff(d)
    with np.errstate(divide="ignore", invalid="ignore"):
        sev = np.where(step > 0, turn / np.where(step > 0, step, 1.0) * per, np.nan)
    out[order] = np.concatenate([[0.0], sev])
    return out


REPORT_COLUMNS = ["type", "kind", "hole", "table", "row", "conflicting_row", "depth", "message"]


def check_tables(collars: pd.DataFrame, surveys: pd.DataFrame | None = None, intervals: pd.DataFrame | None = None, *,
                 hole: str | None = None, length: str | None = None, at: str | None = None,
                 azimuth: str | None = None, dip: str | None = None, from_: str | None = None, to: str | None = None,
                 tolerance: float = 0.0) -> pd.DataFrame:
    """The problems in collar / survey / interval tables that stop a hole being desurveyed, as a table.

    These are the checks Leapfrog Geo makes on import. Row numbers count from 1 in the table as given.
    ``type`` is ``"error"`` or ``"warning"``; ``kind`` is one of:

    * ``duplicate_survey_depth`` (error): two survey rows of a hole at the same depth. ``row`` and
      ``conflicting_row`` are consecutive rows sharing the depth. The hole is not desurveyed by default.
    * ``no_survey`` (warning): a collar with no survey rows and no azimuth/dip of its own.
    * ``survey_without_collar`` (warning): survey rows of a hole that has no collar.
    * ``no_intervals`` (warning): a collar with no rows in ``intervals``.
    * ``interval_without_collar`` (warning): intervals of a hole that has no collar.
    * ``interval_beyond_depth`` (error): an interval ending deeper than the collar's hole length, by more
      than ``tolerance`` metres. The interval is kept.

    Leapfrog also warns about "wedge" holes and "duplicate collar and surveys"; geoview does not check those.
    """
    hole_c = find_column(collars, hole, HOLE_CANDIDATES, "hole id")
    c_az = _collar_column(collars, azimuth, AZIMUTH_CANDIDATES, "collar azimuth")
    c_dip = _collar_column(collars, dip, DIP_CANDIDATES, "collar dip")
    c_len = find_column(collars, length, LENGTH_CANDIDATES, "hole length", required=False)

    ch = collars[hole_c].to_numpy()
    first_row = pd.Series(np.arange(1, len(ch) + 1), index=ch)
    first_row = first_row[~first_row.index.duplicated()]                       # duplicated collar: first wins
    own_dir = set()
    if c_az is not None and c_dip is not None:
        own_dir = set(ch[collars[c_az].notna().to_numpy() & collars[c_dip].notna().to_numpy()])

    parts = []

    def add(df: pd.DataFrame, **const):
        df = df.assign(**const)
        parts.append(df.reindex(columns=REPORT_COLUMNS))

    survey_holes: set = set()
    if surveys is not None and len(surveys):
        s_hole = find_column(surveys, hole, HOLE_CANDIDATES, "survey hole id")
        s_at = find_column(surveys, at, DEPTH_CANDIDATES, "survey depth")
        sdf = pd.DataFrame({"hole": surveys[s_hole].to_numpy(), "depth": surveys[s_at].to_numpy(float),
                            "row": np.arange(1, len(surveys) + 1)})
        survey_holes = set(sdf.hole)
        dup = sdf[sdf.duplicated(["hole", "depth"], keep=False)]               # already in table order
        nxt = dup.groupby(["hole", "depth"], sort=False)["row"].shift(-1)
        pairs = dup[nxt.notna()].assign(conflicting_row=nxt[nxt.notna()].astype(int))
        add(pairs, type="error", kind="duplicate_survey_depth", table="survey",
            message="duplicate survey depth, row " + pairs.row.astype(str))
        orphan = sdf[~sdf.hole.isin(first_row.index)].drop_duplicates("hole")
        add(orphan, type="warning", kind="survey_without_collar", table="survey", conflicting_row=pd.NA,
            message="survey rows of a hole with no collar, row " + orphan.row.astype(str))

    unplaced = first_row[[h not in survey_holes and h not in own_dir for h in first_row.index]]
    add(pd.DataFrame({"hole": unplaced.index, "row": unplaced.to_numpy()}), type="warning", kind="no_survey", table="collar",
        message=[f"no surveys for collar, row {r}" for r in unplaced.to_numpy()])

    if intervals is not None:
        i_hole = find_column(intervals, hole, HOLE_CANDIDATES, "interval hole id")
        i_to = find_column(intervals, to, TO_CANDIDATES, "interval to")
        idf = pd.DataFrame({"hole": intervals[i_hole].to_numpy(), "end": intervals[i_to].to_numpy(float),
                            "row": np.arange(1, len(intervals) + 1)})
        orphan = idf[~idf.hole.isin(first_row.index)].drop_duplicates("hole")
        add(orphan, type="warning", kind="interval_without_collar", table="intervals", conflicting_row=pd.NA,
            message="intervals of a hole with no collar, row " + orphan.row.astype(str))
        empty = first_row[~first_row.index.isin(set(idf.hole))]
        add(pd.DataFrame({"hole": empty.index, "row": empty.to_numpy()}), type="warning", kind="no_intervals", table="collar",
            message=[f"no samples for collar, row {r}" for r in empty.to_numpy()])
        if c_len is not None:
            depth = collars.loc[~collars[hole_c].duplicated(), [hole_c, c_len]].set_index(hole_c)[c_len]
            lim = idf.hole.map(depth)
            far = idf[(idf["end"] > lim + tolerance).to_numpy()]
            add(far.assign(conflicting_row=far.hole.map(first_row)), type="error", kind="interval_beyond_depth",
                table="intervals", depth=far["end"],
                message="to value (row " + far.row.astype(str) + ") exceeds max depth in collar table, row "
                        + far.hole.map(first_row).astype(str))

    parts = [d for d in parts if len(d)]
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=REPORT_COLUMNS)
    return out.reindex(columns=REPORT_COLUMNS)


def _deduplicate(surveys: pd.DataFrame, hole_c: str, at_c: str, rule) -> pd.DataFrame:
    """Survey rows sharing a (hole, depth) reduced to one by ``rule``: "first", "last" (in table order) or a
    function that gets the rows sharing a depth (a DataFrame) and returns one row (a Series or 1-row DataFrame)."""
    if rule in ("first", "last"):
        return surveys.drop_duplicates([hole_c, at_c], keep=rule)
    dup = surveys.duplicated([hole_c, at_c], keep=False)
    rows = []
    for _, g in surveys[dup].groupby([hole_c, at_c], sort=False):
        r = rule(g)
        rows.append(r.to_frame().T if isinstance(r, pd.Series) else r)
    return pd.concat([surveys[~dup], *rows]).sort_index(kind="stable")


class _Prepared:
    """What `Drillholes.from_tables` and the DXF trace export share: which holes can be placed, and how."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def _prepare(collars, surveys, intervals, *, hole, xyz, length, at, azimuth, dip, from_, to, dip_positive_down,
             duplicates) -> _Prepared:
    if not (duplicates in ("skip", "first", "last") or callable(duplicates)):
        raise ValueError(f'duplicates must be "skip", "first", "last" or a function, got {duplicates!r}')
    hole_c = find_column(collars, hole, HOLE_CANDIDATES, "hole id")
    xyz_c = tuple(xyz) if xyz else find_columns(collars, XYZ_CANDIDATES)
    if not xyz_c:
        raise ValueError(f"could not find collar X/Y/Z in {list(collars.columns)}; pass xyz=")
    sign = -1.0 if dip_positive_down else 1.0
    c_az = _collar_column(collars, azimuth, AZIMUTH_CANDIDATES, "collar azimuth")
    c_dip = _collar_column(collars, dip, DIP_CANDIDATES, "collar dip")
    c_len = find_column(collars, length, LENGTH_CANDIDATES, "hole length", required=False)

    report = check_tables(collars, surveys, intervals, hole=hole, length=length, at=at, azimuth=azimuth,
                          dip=dip, from_=from_, to=to)
    dup_holes = set(report.loc[report.kind == "duplicate_survey_depth", "hole"])
    skipped: dict = {}

    # orientation per hole: its survey stations, else the collar's own azimuth/dip, else the hole is left out
    stations: dict = {}
    if surveys is not None and len(surveys):
        s_hole = find_column(surveys, hole, HOLE_CANDIDATES, "survey hole id")
        s_at = find_column(surveys, at, DEPTH_CANDIDATES, "survey depth")
        s_az = find_column(surveys, azimuth, AZIMUTH_CANDIDATES, "survey azimuth")
        s_dip = find_column(surveys, dip, DIP_CANDIDATES, "survey dip")
        if dup_holes and duplicates != "skip":
            surveys = _deduplicate(surveys, s_hole, s_at, duplicates)
        for hid, g in surveys.groupby(s_hole, sort=False):
            if hid in dup_holes and duplicates == "skip":
                skipped[hid] = "a repeated survey depth"
                continue
            stations[hid] = (g[s_at].to_numpy(float), g[s_az].to_numpy(float), sign * g[s_dip].to_numpy(float))
    own_dir = set()
    if c_az is not None and c_dip is not None:
        own_dir = set(collars.loc[collars[c_az].notna() & collars[c_dip].notna(), hole_c])
    for hid in collars[hole_c].unique():
        if hid not in stations and hid not in skipped and hid not in own_dir:
            skipped[hid] = "no survey"
    return _Prepared(hole_c=hole_c, xyz_c=xyz_c, sign=sign, c_az=c_az, c_dip=c_dip, c_len=c_len, report=report,
                     dup_holes=dup_holes, stations=stations, skipped=skipped, duplicates=duplicates)


def _left_out_text(skipped: dict) -> str:
    why = pd.Series(skipped).value_counts()
    return ", ".join(f"{n} with {r}" for r, n in why.items())


def _warn_left_out(prep: _Prepared, stacklevel: int = 3) -> None:
    """The warnings every entry point gives: holes left out, and survey rows deduplicated by a rule."""
    if prep.skipped:
        hint = (" Pass duplicates='first' or 'last' (or a function) to deduplicate the survey instead of skipping them."
                if "a repeated survey depth" in set(prep.skipped.values()) else "")
        warnings.warn(f"{len(prep.skipped)} holes were not desurveyed ({_left_out_text(prep.skipped)}); "
                      f"they are listed in .skipped and every problem in .report.{hint}", stacklevel=stacklevel)
    if prep.duplicates != "skip" and prep.dup_holes:
        rule = prep.duplicates if isinstance(prep.duplicates, str) else "a function"
        warnings.warn(f"{len(prep.dup_holes)} holes have repeated survey depths; deduplicated with duplicates={rule!r}.",
                      stacklevel=stacklevel)


@dataclass
class Desurveyed:
    """Result of `desurvey_intervals`. `intervals` has the kept rows: HOLEID, FROM, TO, the attribute columns, `_row` (0-based row of
    the interval in the table given; -1 for the cuts made when no intervals were given) and the positions in real coordinates
    `start_x/y/z`, `mid_x/y/z` (half way along the hole, on its arc) and `end_x/y/z`. `skipped` maps each hole left out to the
    reason; `report` is `check_tables` of the input."""
    intervals: pd.DataFrame
    report: pd.DataFrame
    skipped: pd.Series


def desurvey_intervals(collars: pd.DataFrame, surveys: pd.DataFrame | None = None, intervals: pd.DataFrame | None = None, *,
                       hole: str | None = None, xyz: Sequence[str] | None = None, length: str | None = None,
                       at: str | None = None, azimuth: str | None = None, dip: str | None = None,
                       from_: str | None = None, to: str | None = None, attributes: Sequence[str] | None = None,
                       dip_positive_down: bool = True, trace_step: float = 10.0, duplicates="skip",
                       warn: bool = True, stacklevel: int = 2) -> Desurveyed:
    """Desurvey collar / survey / interval tables: the position of the start, middle and end of every interval. Column names are
    found automatically (case-insensitive) unless given.

    ``collars``: hole id, X/Y/Z, and optionally length and azimuth/dip (used for a hole that has no survey rows).
    ``surveys``: hole id, depth (``at``), azimuth, dip. ``intervals``: hole id, from, to, plus attribute columns (all others by
    default; ``attributes`` picks). Without ``intervals``, each hole is cut every ``trace_step`` metres.
    ``dip_positive_down``: True (default) if the tables store dip positive when the hole points down, as Leapfrog's survey export
    does; False for negative-down tables.

    Holes that cannot be placed are left out and reported (a warning, `.skipped`, `.report`), as Leapfrog Geo does; nothing is
    guessed. A hole with no survey rows and no azimuth/dip on its collar is never assumed vertical. A hole with two survey rows at the
    same depth is skipped unless ``duplicates`` says how to deduplicate: "first" or "last" keep that row of each depth (table order);
    a function gets the rows sharing a depth (a DataFrame) and returns the one row to use.
    """
    prep = _prepare(collars, surveys, intervals, hole=hole, xyz=xyz, length=length, at=at, azimuth=azimuth, dip=dip,
                    from_=from_, to=to, dip_positive_down=dip_positive_down, duplicates=duplicates)
    hole_c, xyz_c, sign, c_az, c_dip, c_len = prep.hole_c, prep.xyz_c, prep.sign, prep.c_az, prep.c_dip, prep.c_len
    report, stations, skipped = prep.report, prep.stations, prep.skipped

    if intervals is not None:
        i_hole = find_column(intervals, hole, HOLE_CANDIDATES, "interval hole id")
        i_from = find_column(intervals, from_, FROM_CANDIDATES, "interval from")
        i_to = find_column(intervals, to, TO_CANDIDATES, "interval to")
        known = set(collars[hole_c])
        unknown = ~intervals[i_hole].isin(known)
        if unknown.any() and warn:
            warnings.warn(f"{int(unknown.sum())} intervals belong to holes with no collar and were dropped "
                          f"(e.g. {intervals.loc[unknown, i_hole].iloc[0]!r})", stacklevel=stacklevel)
        extra = list(attributes) if attributes is not None else [c for c in intervals.columns if c not in (i_hole, i_from, i_to)]
        table = intervals[[i_hole, i_from, i_to, *extra]].rename(
            columns={i_hole: "HOLEID", i_from: "FROM", i_to: "TO"}).assign(_row=np.arange(len(intervals)))
        table = table[~unknown.to_numpy()]
        table = table[~table["HOLEID"].isin(set(skipped))].reset_index(drop=True)
    else:
        rows = []
        for _, c in collars.iterrows():
            hid = c[hole_c]
            if hid in skipped:
                continue
            total = float(c[c_len]) if c_len is not None and pd.notna(c[c_len]) else (
                float(stations[hid][0].max()) if hid in stations else np.nan)
            if not np.isfinite(total) or total <= 0:
                continue
            cuts = np.unique(np.append(np.arange(0.0, total, trace_step), total))
            rows.append(pd.DataFrame({"HOLEID": hid, "FROM": cuts[:-1], "TO": cuts[1:], "_row": -1}))
        table = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["HOLEID", "FROM", "TO", "_row"])

    if warn:
        _warn_left_out(prep, stacklevel + 1)
    if not len(table):
        raise ValueError("no hole could be desurveyed" + (f" ({len(skipped)} left out: {_left_out_text(skipped)})" if skipped else ""))

    pos = {k: np.empty((len(table), 3)) for k in ("start", "mid", "end")}
    collar_rows = collars.set_index(hole_c)
    for hid, idx in table.groupby("HOLEID", sort=False).indices.items():
        c = collar_rows.loc[hid]
        if isinstance(c, pd.DataFrame):
            c = c.iloc[0]  # duplicated collar row: first wins
        if hid in stations:
            sd, az, dp = stations[hid]
        else:                                               # the collar's own azimuth/dip (checked in _prepare)
            sd, az, dp = np.array([0.0]), np.array([float(c[c_az])]), np.array([sign * float(c[c_dip])])
        f = table["FROM"].to_numpy(float)[idx]
        t = table["TO"].to_numpy(float)[idx]
        n = len(idx)
        p = desurvey(c[list(xyz_c)].to_numpy(float), sd, az, dp, np.concatenate([f, t, (f + t) / 2]))
        pos["start"][idx], pos["end"][idx], pos["mid"][idx] = p[:n], p[n:2 * n], p[2 * n:]
    cols = {f"{name}_{ax}": pos[name][:, k] for name in ("start", "mid", "end") for k, ax in enumerate("xyz")}
    out = pd.concat([table, pd.DataFrame(cols)], axis=1)
    return Desurveyed(out, report, pd.Series(skipped, dtype=object, name="reason"))
