"""The problems in a drillhole set, before it is used: `DrillholeSet.validate` (this module's `validate_set`).

It runs `desurvey.check_tables` (the checks Leapfrog Geo makes on import: duplicate survey depths, collars with no survey, rows of
holes with no collar, intervals beyond the hole's length) on every table of the set, and adds:

* ``duplicate_collar`` (error): a hole with two collar rows (the first is used).
* ``from_ge_to`` (error): an interval whose FROM is not before its TO.
* ``overlapping_intervals`` (warning): an interval starting before the previous one of its hole ends.
* ``gap`` (info): an interval starting after the previous one ends.
* ``point_without_collar`` (warning), ``point_beyond_depth`` (warning): depth points of a hole with no collar, or deeper than it.
* ``value_out_of_range`` (warning): an azimuth outside 0-360 or a dip outside -90..90.
* ``unparsed_values`` (warning): values of a number or date column that could not be read as one (``<0.01``, ``ND``) and became empty.
* ``hole_id_mismatch`` (warning): a hole id that matches a collar only once spaces, case and dashes are ignored (``DDH-01`` / ``ddh01 ``).

Row numbers count from 1 in the table as given.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

LEVELS = ("error", "warning", "info")


@dataclass
class Issue:
    """One problem: ``level`` ("error", "warning" or "info"), ``kind``, the ``table`` (its name in the set), the ``hole`` and ``row`` it is
    about (None when it is about a whole column), and ``detail`` in plain words."""
    level: str
    kind: str
    table: str
    hole: str | None
    row: int | None
    detail: str

    def __str__(self) -> str:
        where = " ".join(x for x in (self.hole or "", f"row {self.row}" if self.row else "") if x)
        return f"{self.level:<8} {self.kind:<22} {self.table:<16} {where:<18} {self.detail}"


def issues_frame(issues) -> pd.DataFrame:
    """Issues as a table (one row each), errors first."""
    df = pd.DataFrame([asdict(i) for i in issues], columns=["level", "kind", "table", "hole", "row", "detail"])
    return df.sort_values("level", key=lambda s: s.map(LEVELS.index), kind="stable").reset_index(drop=True)


def _norm(h: str) -> str:
    return re.sub(r"[\s\-_]+", "", str(h)).casefold()


def _intervals(name: str, t: pd.DataFrame) -> list:
    out = []
    f, to = t["FROM"].to_numpy(float), t["TO"].to_numpy(float)
    rows = np.arange(1, len(t) + 1)
    for k in np.flatnonzero(~(f < to) & np.isfinite(f) & np.isfinite(to)):
        out.append(Issue("error", "from_ge_to", name, t["HOLEID"].iat[k], int(rows[k]), f"FROM {f[k]:g} is not before TO {to[k]:g}"))
    ok = t.assign(_row=rows)[f < to].sort_values(["HOLEID", "FROM"], kind="stable")
    prev_to = ok.groupby("HOLEID", sort=False)["TO"].shift()
    over = ok[(ok["FROM"] < prev_to - 1e-9).to_numpy()]
    for _, r in over.iterrows():
        out.append(Issue("warning", "overlapping_intervals", name, r["HOLEID"], int(r["_row"]),
                         f"{r['FROM']:g}-{r['TO']:g} starts before the previous interval ends ({prev_to[_]:g})"))
    gap = ok[(ok["FROM"] > prev_to + 1e-9).to_numpy()]
    for _, r in gap.iterrows():
        out.append(Issue("info", "gap", name, r["HOLEID"], int(r["_row"]), f"no interval from {prev_to[_]:g} to {r['FROM']:g}"))
    return out


def _ranges(name: str, t: pd.DataFrame) -> list:
    out = []
    for col, lo, hi in (("AZIMUTH", 0, 360), ("DIP", -90, 90)):
        if col not in t:
            continue
        v = t[col]
        bad = v.notna() & ~v.between(lo, hi)
        for k in np.flatnonzero(bad.to_numpy())[:50]:
            out.append(Issue("warning", "value_out_of_range", name, t["HOLEID"].iat[k], int(k + 1), f"{col.lower()} {v.iat[k]:g} is outside {lo}..{hi}"))
    return out


def validate_set(s) -> list:
    """Every problem of a `DrillholeSet` (see the module), errors first."""
    from .desurvey import check_tables

    tables = s.tables()                                           # (name, role), collars first
    frame = {name: s.canonical(name) for name, _ in tables}
    cname = s.collars_name
    collars = frame[cname]
    surveys = frame[s.surveys_name] if s.surveys is not None else None
    out: list = []

    def from_report(rep: pd.DataFrame, names: dict, keep=None):
        for r in rep.itertuples(index=False):
            if keep and r.kind not in keep:
                continue
            row = None if pd.isna(r.row) else int(r.row)
            out.append(Issue(r.type, r.kind, names.get(r.table, r.table), None if pd.isna(r.hole) else str(r.hole), row, str(r.message)))

    from_report(check_tables(collars, surveys), {"collar": cname, "survey": s.surveys_name})
    for name, role in tables:
        if role == "intervals":
            from_report(check_tables(collars, None, frame[name]), {"intervals": name, "collar": name},
                        keep={"interval_without_collar", "interval_beyond_depth"})

    dup = collars["HOLEID"].duplicated(keep="first")
    for k in np.flatnonzero(dup.to_numpy()):
        out.append(Issue("error", "duplicate_collar", cname, collars["HOLEID"].iat[k], int(k + 1), "a second collar row for this hole (the first is used)"))
    known = set(collars["HOLEID"])
    norm = {_norm(h): h for h in known}
    depth = collars.drop_duplicates("HOLEID").set_index("HOLEID")["LENGTH"] if "LENGTH" in collars else pd.Series(dtype=float)
    out += _ranges(cname, collars)
    if surveys is not None:
        out += _ranges(s.surveys_name, surveys)
    for name, role in tables:
        t = frame[name]
        if role == "intervals":
            out += _intervals(name, t)
        elif role == "points":
            orphan = t[~t["HOLEID"].isin(known)].drop_duplicates("HOLEID")
            for k, r in orphan.iterrows():
                out.append(Issue("warning", "point_without_collar", name, r["HOLEID"], int(k + 1), "depth points of a hole with no collar"))
            if len(depth):
                lim = t["HOLEID"].map(depth)
                for k in np.flatnonzero((t["DEPTH"] > lim).to_numpy()):
                    out.append(Issue("warning", "point_beyond_depth", name, t["HOLEID"].iat[k], int(k + 1),
                                     f"depth {t['DEPTH'].iat[k]:g} is beyond the hole's length {lim.iat[k]:g}"))
        if role != "collars":
            for h in sorted(set(t["HOLEID"]) - known):
                match = norm.get(_norm(h))
                if match is not None:
                    out.append(Issue("warning", "hole_id_mismatch", name, h, None, f"{h!r} is written {match!r} in the collars"))
    for name, bad in s.unparsed.items():
        for col, (n, examples) in bad.items():
            out.append(Issue("warning", "unparsed_values", name, None, None,
                             f"{n:,} values of {col} could not be read and became empty (e.g. {', '.join(map(repr, examples))})"))
    return sorted(out, key=lambda i: LEVELS.index(i.level))
