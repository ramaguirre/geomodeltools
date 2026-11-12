"""Drillhole sets: one collar table, at most one survey table, and any number of interval and point tables that belong together.

A set is how Leapfrog Geo groups drillhole data, and how a project with two campaigns (or two deposits) keeps them apart:

    s = DrillholeSet("campaign 2025", collars, surveys)
    s.add("assays", assays)                      # an interval table (hole id, from, to, values)
    s.add("structures", structures)              # a point table (hole id, depth along the hole, values)
    s.interval_table("assays")                   # TOP/MID/BOTTOM X/Y/Z, by minimum curvature
    s.traces()                                   # one Polyline per hole

Tables are told apart by their columns (`drillhole_role`): hole id + from + to are **intervals**; hole id + depth + azimuth +
dip (no X/Y/Z) a **survey**; hole id + X/Y/Z with one row per hole a **collar** table; hole id + depth (no from/to, no X/Y/Z) a
**point** table, i.e. depth points along the holes. A table with a hole id and X/Y/Z on many rows per hole is free points that
happen to carry a hole id (one per photo, say): not part of a set (`drillhole_role` returns None).

`group_tables` sorts loaded tables into sets by the holes they share: each collar table starts a set; a survey, interval or
point table joins the set whose collars hold most of its holes. It never guesses where the answer is unclear: a table whose
holes match no set well enough, or two sets about equally, or a second survey table for a set, comes back in ``unsure`` with
the scores, for the user to place (`DrillholeSet.add` / `remove`).
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import pandas as pd

from .columns import XYZ_CANDIDATES, find_column, find_columns
from .desurvey import AZIMUTH_CANDIDATES, DEPTH_CANDIDATES, DIP_CANDIDATES, FROM_CANDIDATES, HOLE_CANDIDATES, TO_CANDIDATES

ROLES = ("collars", "surveys", "intervals", "points")
MIN_SHARE = 0.5   # a table joins a set when at least this share of its holes are in the set's collars...
MARGIN = 0.2      # ...and that share beats the next set's by at least this much


def _has(df: pd.DataFrame, candidates) -> str | None:
    return find_column(df, None, candidates, "", required=False)


def drillhole_role(df: pd.DataFrame) -> str | None:
    """What part of a drillhole set a table is, from its columns (usual names, any case): ``"collars"``, ``"surveys"``,
    ``"intervals"``, ``"points"`` (depth points along the holes), or None (not drillhole data, or free X/Y/Z points)."""
    hole = _has(df, HOLE_CANDIDATES)
    if hole is None:
        return None
    xyz = find_columns(df, XYZ_CANDIDATES)
    if _has(df, FROM_CANDIDATES) and _has(df, TO_CANDIDATES):
        return "intervals"
    if xyz is None and _has(df, DEPTH_CANDIDATES) and _has(df, AZIMUTH_CANDIDATES) and _has(df, DIP_CANDIDATES):
        return "surveys"
    if xyz is not None:
        n = len(df)
        repeats = n >= 20 and df[hole].nunique() < 0.9 * n  # several rows per hole, each with a position: free points, not collars
        return None if repeats else "collars"
    if _has(df, DEPTH_CANDIDATES):
        return "points"
    return None


def hole_ids(df: pd.DataFrame) -> set[str]:
    """The holes a table names (as text, stripped), empty if it has no hole id column."""
    hole = _has(df, HOLE_CANDIDATES)
    if hole is None:
        return set()
    return {str(h).strip() for h in df[hole].dropna()}


@dataclass(eq=False)
class DrillholeSet:
    """One collar table, at most one survey table (without one, each hole is oriented by its collar's azimuth and dip), and named
    interval and point tables. ``dip_positive_down``: how the survey (or collar) dips are stored (geomodeltools' default, as
    Leapfrog's survey export; pass False for -60 = 60 degrees down)."""
    name: str
    collars: pd.DataFrame
    surveys: pd.DataFrame | None = None
    intervals: dict = field(default_factory=dict)
    points: dict = field(default_factory=dict)
    dip_positive_down: bool = True
    #: the names the collar and survey tables were loaded under (e.g. their files), for lists and messages
    collars_name: str = ""
    surveys_name: str = ""

    def __post_init__(self):
        if _has(self.collars, HOLE_CANDIDATES) is None:
            raise ValueError(f"set {self.name!r}: the collar table has no hole id column (one of {', '.join(HOLE_CANDIDATES)})")
        self.collars_name = self.collars_name or f"{self.name} collars"

    # ---- what is in it ---------------------------------------------------------------------------------------------------------

    def holes(self) -> set[str]:
        """The holes of the set: those of its collar table."""
        return hole_ids(self.collars)

    def tables(self) -> list[tuple[str, str]]:
        """(name, role) of every table in the set, collars first."""
        out = [(self.collars_name, "collars")]
        if self.surveys is not None:
            out.append((self.surveys_name, "surveys"))
        out += [(n, "intervals") for n in self.intervals] + [(n, "points") for n in self.points]
        return out

    def share(self, df: pd.DataFrame) -> float:
        """The share of a table's holes that are in this set's collars (0 to 1; 0 for a table with no holes)."""
        ids = hole_ids(df)
        return len(ids & self.holes()) / len(ids) if ids else 0.0

    def missing(self, df: pd.DataFrame) -> set[str]:
        """Holes of a table that this set's collars do not have (their rows cannot be placed)."""
        return hole_ids(df) - self.holes()

    # ---- changing it ------------------------------------------------------------------------------------------------------------

    def add(self, name: str, df: pd.DataFrame, role: str | None = None) -> str:
        """Put a table in the set, as ``role`` (found from its columns unless given): a survey replaces the set's survey table, an
        interval or point table is added under ``name`` (replacing one of the same name). Returns the role. Collars cannot be added:
        a set has exactly one collar table (make a new set for another)."""
        role = role or drillhole_role(df)
        if role not in ("surveys", "intervals", "points"):
            raise ValueError(f"{name}: a set takes a survey, interval or point table, not {role or 'this table'}; a collar table starts its own set")
        if _has(df, HOLE_CANDIDATES) is None:
            raise ValueError(f"{name}: no hole id column (one of {', '.join(HOLE_CANDIDATES)})")
        self.remove(name)
        if role == "surveys":
            self.surveys, self.surveys_name = df, name
        else:
            getattr(self, role)[name] = df
        return role

    def remove(self, name: str) -> pd.DataFrame | None:
        """Take a survey, interval or point table out of the set (by the name it has in the set); returns it, or None if absent."""
        if self.surveys is not None and name == self.surveys_name:
            df, self.surveys, self.surveys_name = self.surveys, None, ""
            return df
        for group in (self.intervals, self.points):
            if name in group:
                return group.pop(name)
        return None

    # ---- desurveyed ---------------------------------------------------------------------------------------------------------------

    def interval_table(self, name: str, **kw) -> pd.DataFrame:
        """An interval table of the set in real coordinates (`geomodeltools.drillholes.interval_table`: TOP/MID/BOTTOM X/Y/Z)."""
        from .drillholes import interval_table

        return interval_table(self.collars, self.surveys, self.intervals[name], dip_positive_down=self.dip_positive_down, **kw)

    def point_table(self, name: str, **kw) -> pd.DataFrame:
        """A point table of the set in real coordinates (`geomodeltools.drillholes.point_table`: X/Y/Z at each depth)."""
        from .drillholes import point_table

        return point_table(self.collars, self.surveys, self.points[name], dip_positive_down=self.dip_positive_down, **kw)

    def traces(self, **kw) -> list:
        """One `Polyline` per hole of the set (`geomodeltools.drillholes.traces`)."""
        from .drillholes import traces

        return traces(self.collars, self.surveys, dip_positive_down=self.dip_positive_down, **kw)

    def desurvey(self) -> dict[str, pd.DataFrame]:
        """Every interval and point table of the set in real coordinates, by name."""
        out = {n: self.interval_table(n) for n in self.intervals}
        out.update({n: self.point_table(n) for n in self.points})
        return out


@dataclass
class Unsure:
    """A table `group_tables` would not place by itself: its ``name``, ``role``, the share of its holes in each set's collars
    (``scores``, best first) and ``why``."""
    name: str
    role: str
    scores: list
    why: str


@dataclass
class Grouping:
    """What `group_tables` made: the ``sets`` (the ones given, then one per new collar table), the tables it was ``unsure`` about,
    and the tables that are not drillhole data (``other``: name -> why)."""
    sets: list
    unsure: list
    other: dict


def _unique(name: str, taken: set[str]) -> str:
    out, k = name, 2
    while out.casefold() in {t.casefold() for t in taken}:
        out, k = f"{name} ({k})", k + 1
    return out


def group_tables(tables: Mapping[str, pd.DataFrame], sets: Sequence[DrillholeSet] = (), *, roles: Mapping[str, str] | None = None,
                 min_share: float = MIN_SHARE, margin: float = MARGIN, dip_positive_down: bool = True) -> Grouping:
    """Sort tables (name -> DataFrame, e.g. files just loaded) into drillhole sets, the existing ``sets`` first.

    Each collar table starts a new set, named after it. A survey, interval or point table joins the set whose collars hold the
    largest share of its holes, when that share is at least ``min_share`` and beats the next set's by ``margin``; otherwise, or
    when the set already has a survey table, it is returned in ``unsure`` (nothing guessed). ``roles`` (name -> role) overrides what
    `drillhole_role` finds, e.g. after the user said what a table is. ``dip_positive_down`` is given to the new sets."""
    roles = dict(roles or {})
    out = list(sets)
    unsure, other = [], {}
    found = {}
    for name, df in tables.items():
        role = roles.get(name) or drillhole_role(df)
        if role not in ROLES:
            other[name] = "not drillhole data (no hole id with from/to, depth, or a collar position)" if role is None else f"unknown role {role!r}"
            continue
        found[name] = role
    for name, role in found.items():          # collars first: they make the sets the other tables join
        if role == "collars":
            taken = {s.name for s in out}
            out.append(DrillholeSet(_unique(name, taken), tables[name], dip_positive_down=dip_positive_down, collars_name=name))
    order = sorted((n for n, r in found.items() if r != "collars"), key=lambda n: ("surveys", "intervals", "points").index(found[n]))
    for name in order:
        role, df = found[name], tables[name]
        scores = sorted(((s.name, round(s.share(df), 3)) for s in out), key=lambda x: -x[1])
        if not scores:
            unsure.append(Unsure(name, role, [], "there is no collar table to join (load one, or say which set it belongs to)"))
            continue
        best, share = scores[0]
        second = scores[1][1] if len(scores) > 1 else 0.0
        if share < min_share:
            unsure.append(Unsure(name, role, scores, f"only {share:.0%} of its holes are in the collars of {best!r}"))
            continue
        if share - second < margin:
            unsure.append(Unsure(name, role, scores, f"its holes are in the collars of {best!r} and {scores[1][0]!r} about equally"))
            continue
        target = next(s for s in out if s.name == best)
        if role == "surveys" and target.surveys is not None:
            unsure.append(Unsure(name, role, scores, f"{best!r} already has a survey table ({target.surveys_name})"))
            continue
        target.add(name, df, role)
    return Grouping(out, unsure, other)
