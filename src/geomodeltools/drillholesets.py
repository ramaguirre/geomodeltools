"""Drillhole sets: one collar table, at most one survey table, and any number of interval and point tables that belong together.

A set is how Leapfrog Geo groups drillhole data, and how a project with two campaigns (or two deposits) keeps them apart. Each table
has a `TableSpec` (what its columns are: `schema`), so nothing downstream guesses column names again:

    s = DrillholeSet.onboard(folder, ["collar.csv", "survey.csv", "intervals.csv"])   # read, guess, confirm (stops if unsure)
    s.validate()                                  # every problem, errors first (`validate`)
    s.save(folder / "campaign.dhset.yaml")        # the specs, beside the data
    s = DrillholeSet.open(folder / "campaign.dhset.yaml")     # next time: no questions
    s.interval_table("intervals")                 # TOP/MID/BOTTOM X/Y/Z, by minimum curvature
    s.traces()                                    # one Polyline per hole

Built from tables only, a set guesses each table's spec from its names (``DrillholeSet("campaign", collars, surveys)``), as before.

Tables are told apart by their columns (`drillhole_role`, the rules of `guess.table_kind`). `group_tables` sorts loaded tables into
sets by the holes they share (`best_set`): each collar table starts a set; a survey, interval or point table joins the set whose
collars hold most of its holes. It never guesses where the answer is unclear: a table whose holes match no set well enough, or two
sets about equally, or a second survey table for a set, comes back in ``unsure`` with the scores, for the user to place.
"""
from __future__ import annotations

import pathlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import pandas as pd

from .columns import HOLE_CANDIDATES, find_column
from .guess import REVIEW_BELOW, detect_dip_convention, guess_table, table_kind, unit_of, value_dtype
from .schema import DRILLHOLE_KINDS, REQUIRED, ROLE_LABELS, ColumnSpec, SetSpec, TableSpec

ROLES = DRILLHOLE_KINDS   # "collars", "surveys", "intervals", "points"
MIN_SHARE = 0.5   # a table joins a set when at least this share of its holes are in the set's collars...
MARGIN = 0.2      # ...and that share beats the next set's by at least this much


class SpecMismatch(ValueError):
    """A saved set's tables no longer have the columns its specs name. ``changes``: table -> {"missing", "new"}."""

    def __init__(self, message: str, changes: dict):
        super().__init__(message)
        self.changes = changes


class NeedsReview(ValueError):
    """`DrillholeSet.onboard` is not sure about some columns. ``guesses``: file -> `TableGuess`; ``unsure``: file -> the columns to look at.
    Confirm them with ``edits={file: {column: {...}}}`` (an empty ``{}`` accepts the guess)."""

    def __init__(self, message: str, guesses: dict, unsure: dict):
        super().__init__(message)
        self.guesses, self.unsure = guesses, unsure


def drillhole_role(df: pd.DataFrame) -> str | None:
    """What part of a drillhole set a table is, from its columns (usual names, any case): ``"collars"``, ``"surveys"``,
    ``"intervals"``, ``"points"`` (depth points along the holes), or None (not drillhole data, or free X/Y/Z points)."""
    kind = table_kind(df)
    return kind if kind in DRILLHOLE_KINDS else None


def hole_ids(df: pd.DataFrame, column: str | None = None) -> set[str]:
    """The holes a table names (as text, stripped), empty if it has no hole id column. ``column`` names it (else its usual names)."""
    hole = column if column is not None else find_column(df, None, HOLE_CANDIDATES, "", required=False)
    if hole is None:
        return set()
    return {str(h).strip() for h in df[hole].dropna()}


def _auto_spec(name: str, df: pd.DataFrame, role: str, known: set | None = None) -> TableSpec:
    """A table's spec guessed with nobody to review it: only the roles found by their usual names are kept (as before specs), not the
    weaker guesses from part of a name ("HOLE_LENGTH_M" contains LENGTH)."""
    g = guess_table(df, name, kind=role, known_holes=known)
    for c in g.columns:
        if c.role and c.role not in REQUIRED[role] and not any(r.startswith("name matches") for r in c.reasons):
            c.role, c.dtype = None, "number"
    cols = [ColumnSpec(c.name, c.dtype, role=c.role, unit=c.unit) for c in g.columns]
    return TableSpec(name, role, cols)


@dataclass(eq=False)
class DrillholeSet:
    """One collar table, at most one survey table (without one, each hole is oriented by its collar's azimuth and dip), and named
    interval and point tables. ``dip_positive_down``: how the survey (or collar) dips are stored (True, geomodeltools' default, as
    Leapfrog's survey export; False for -60 = 60 degrees down; None: read from the dips' signs). ``specs``: table name -> `TableSpec`
    (a table without one has its spec guessed from its names when first needed)."""
    name: str
    collars: pd.DataFrame
    surveys: pd.DataFrame | None = None
    intervals: dict = field(default_factory=dict)
    points: dict = field(default_factory=dict)
    dip_positive_down: bool | None = True
    #: the names the collar and survey tables were loaded under (e.g. their files), for lists and messages
    collars_name: str = ""
    surveys_name: str = ""
    specs: dict = field(default_factory=dict)
    #: table -> {column: (count, examples)}: values that could not be read as their column's type (see `TableSpec.apply`)
    unparsed: dict = field(default_factory=dict)
    #: how the dip convention was decided, in plain words
    dip_note: str = ""
    crs: str | None = None
    #: table -> columns found in its file that its saved spec did not know (guessed when the set was opened: worth a look)
    review: dict = field(default_factory=dict)

    def __post_init__(self):
        self.collars_name = self.collars_name or f"{self.name} collars"
        if self.surveys is not None and not self.surveys_name:
            self.surveys_name = f"{self.name} surveys"
        self._guessed: dict = {}
        if self.spec_of(self.collars_name).column("hole") is None:
            raise ValueError(f"set {self.name!r}: the collar table has no hole id column (one of {', '.join(HOLE_CANDIDATES)})")
        if self.dip_positive_down is None:
            self.dip_positive_down = self.detect_dip()

    # ---- specs --------------------------------------------------------------------------------------------------------------------

    def frame(self, name: str) -> pd.DataFrame:
        """A table of the set by its name."""
        if name == self.collars_name:
            return self.collars
        if self.surveys is not None and name == self.surveys_name:
            return self.surveys
        for group in (self.intervals, self.points):
            if name in group:
                return group[name]
        raise KeyError(f"set {self.name!r} has no table {name!r} (it has {', '.join(n for n, _ in self.tables())})")

    def role_of(self, name: str) -> str:
        return next(r for n, r in self.tables() if n == name)

    def spec_of(self, name: str) -> TableSpec:
        """The spec of a table: the one given, else one guessed from its column names (kept while the table is the same)."""
        if name in self.specs:
            return self.specs[name]
        df = self.frame(name)
        key = (name, id(df))
        if key not in self._guessed:
            role = "collars" if name == self.collars_name else self.role_of(name)
            self._guessed[key] = _auto_spec(name, df, role)
        return self._guessed[key]

    def canonical(self, name: str) -> pd.DataFrame:
        """A table with its role columns under the standard names and its kept value columns (`TableSpec.canonical`)."""
        spec = self.spec_of(name)
        return spec.canonical(self.frame(name), attributes=spec.kind in ("intervals", "points"))

    def detect_dip(self) -> bool:
        """The dip convention from the signs of the survey dips (else the collars'); True when they do not tell. Sets ``dip_note``."""
        for name in ([self.surveys_name] if self.surveys is not None else []) + [self.collars_name]:
            spec = self.spec_of(name)
            if spec.column("dip"):
                res = detect_dip_convention(self.frame(name), dip=spec.column("dip"), hole=spec.column("hole"))
                if res["positive_down"] is not None:
                    self.dip_note = f"Dips: {res['reason']}."
                    return bool(res["positive_down"])
        self.dip_note = "Dips: their signs do not tell; read as positive-down (geomodeltools' default)."
        return True

    # ---- what is in it ---------------------------------------------------------------------------------------------------------

    def holes(self) -> set[str]:
        """The holes of the set: those of its collar table."""
        return hole_ids(self.collars, self.spec_of(self.collars_name).column("hole"))

    def tables(self) -> list[tuple[str, str]]:
        """(name, role) of every table in the set, collars first."""
        out = [(self.collars_name, "collars")]
        if self.surveys is not None:
            out.append((self.surveys_name, "surveys"))
        out += [(n, "intervals") for n in self.intervals] + [(n, "points") for n in self.points]
        return out

    def share(self, df: pd.DataFrame, spec: TableSpec | None = None) -> float:
        """The share of a table's holes that are in this set's collars (0 to 1; 0 for a table with no holes)."""
        ids = hole_ids(df, spec.column("hole") if spec else None)
        return len(ids & self.holes()) / len(ids) if ids else 0.0

    def missing(self, df: pd.DataFrame, spec: TableSpec | None = None) -> set[str]:
        """Holes of a table that this set's collars do not have (their rows cannot be placed)."""
        return hole_ids(df, spec.column("hole") if spec else None) - self.holes()

    # ---- changing it ------------------------------------------------------------------------------------------------------------

    def add(self, name: str, df: pd.DataFrame, role: str | None = None, spec: TableSpec | None = None) -> str:
        """Put a table in the set, as ``role`` (its spec's kind, else found from its columns): a survey replaces the set's survey table,
        an interval or point table is added under ``name`` (replacing one of the same name). Returns the role. Collars cannot be added:
        a set has exactly one collar table (make a new set for another)."""
        role = role or (spec.kind if spec else drillhole_role(df))
        if role not in ("surveys", "intervals", "points"):
            raise ValueError(f"{name}: a set takes a survey, interval or point table, not {role or 'this table'}; a collar table starts its own set")
        hole = spec.column("hole") if spec else find_column(df, None, HOLE_CANDIDATES, "", required=False)
        if hole is None:
            raise ValueError(f"{name}: no hole id column (one of {', '.join(HOLE_CANDIDATES)})")
        self.remove(name)
        if role == "surveys":
            if self.surveys is not None:
                self.remove(self.surveys_name)
            self.surveys, self.surveys_name = df, name
        else:
            getattr(self, role)[name] = df
        if spec is not None:
            self.specs[name] = spec
        return role

    def remove(self, name: str) -> pd.DataFrame | None:
        """Take a survey, interval or point table out of the set (by the name it has in the set); returns it, or None if absent."""
        df = None
        if self.surveys is not None and name == self.surveys_name:
            df, self.surveys, self.surveys_name = self.surveys, None, ""
        else:
            for group in (self.intervals, self.points):
                if name in group:
                    df = group.pop(name)
                    break
        if df is not None:
            self.specs.pop(name, None)
            self.unparsed.pop(name, None)
        return df

    # ---- desurveyed ---------------------------------------------------------------------------------------------------------------

    def collars_and_surveys(self) -> tuple:
        """The collar and survey tables as `desurvey` takes them (`canonical`; surveys None when the set has none)."""
        return self.canonical(self.collars_name), (self.canonical(self.surveys_name) if self.surveys is not None else None)

    def interval_table(self, name: str, **kw) -> pd.DataFrame:
        """An interval table of the set in real coordinates (`geomodeltools.drillholes.interval_table`: TOP/MID/BOTTOM X/Y/Z)."""
        from .drillholes import interval_table

        c, s = self.collars_and_surveys()
        return interval_table(c, s, self.canonical(name), dip_positive_down=self.dip_positive_down, **kw)

    def point_table(self, name: str, **kw) -> pd.DataFrame:
        """A point table of the set in real coordinates (`geomodeltools.drillholes.point_table`: X/Y/Z at each depth)."""
        from .drillholes import point_table

        c, s = self.collars_and_surveys()
        p = self.canonical(name)
        return point_table(c, s, p, depth="DEPTH", attributes=[x for x in p.columns if x not in ("HOLEID", "DEPTH")],
                           dip_positive_down=self.dip_positive_down, **kw)

    def traces(self, **kw) -> list:
        """One `Polyline` per hole of the set (`geomodeltools.drillholes.traces`)."""
        from .drillholes import traces

        c, s = self.collars_and_surveys()
        return traces(c, s, dip_positive_down=self.dip_positive_down, **kw)

    def desurvey(self) -> dict[str, pd.DataFrame]:
        """Every interval and point table of the set in real coordinates, by name."""
        out = {n: self.interval_table(n) for n in self.intervals}
        out.update({n: self.point_table(n) for n in self.points})
        return out

    # ---- checking it ---------------------------------------------------------------------------------------------------------------

    def validate(self) -> list:
        """Every problem of the set, errors first (`validate.validate_set`): a list of `Issue`."""
        from .validate import validate_set

        return validate_set(self)

    def issues_frame(self) -> pd.DataFrame:
        """`validate` as a table."""
        from .validate import issues_frame

        return issues_frame(self.validate())

    # ---- from specs, and saved --------------------------------------------------------------------------------------------------

    @classmethod
    def from_specs(cls, name: str, *, collars: tuple, surveys: tuple | None = None, intervals: Mapping[str, tuple] | None = None,
                   points: Mapping[str, tuple] | None = None, dip_positive_down: bool | None = None, crs: str | None = None) -> DrillholeSet:
        """A set from confirmed specs: each table is a ``(TableSpec, DataFrame)`` pair (``intervals`` and ``points``: name -> pair). The
        tables are typed as their specs say (`TableSpec.apply`; the values that could not be read are kept in ``unparsed``).
        ``dip_positive_down`` None reads it from the dips' signs."""
        unparsed, specs = {}, {}

        def typed(pair, kind):
            spec, df = pair
            if spec.kind != kind:
                raise ValueError(f"{spec.name}: its spec says it is a {spec.kind} table, given as {kind}")
            missing = spec.missing_roles()
            if missing:
                raise ValueError(f"{spec.name}: no column for {', '.join(ROLE_LABELS[r] for r in missing)}")
            table, bad = spec.apply(df, report=True)
            specs[spec.name] = spec
            if bad:
                unparsed[spec.name] = bad
            return spec.name, table

        cname, ctable = typed(collars, "collars")
        sname, stable = typed(surveys, "surveys") if surveys else ("", None)
        ints = dict(typed(p, "intervals") for p in (intervals or {}).values())
        pts = dict(typed(p, "points") for p in (points or {}).values())
        return cls(name, ctable, stable, ints, pts, dip_positive_down=dip_positive_down, collars_name=cname, surveys_name=sname,
                   specs=specs, unparsed=unparsed, crs=crs)

    def to_spec(self, *, keep_dip: bool | None = None) -> SetSpec:
        """The set's `SetSpec` (what `save` writes). The dip convention is saved when it was given, not when it was read from the signs
        (``keep_dip`` True saves it either way)."""
        t = {n: self.spec_of(n) for n, _ in self.tables()}
        detected = self.dip_note.startswith("Dips:")
        dip = self.dip_positive_down if (keep_dip or (keep_dip is None and not detected)) else None
        return SetSpec(self.name, t[self.collars_name], t[self.surveys_name] if self.surveys is not None else None,
                       [t[n] for n in self.intervals], [t[n] for n in self.points], dip_positive_down=dip, crs=self.crs)

    def save(self, path) -> pathlib.Path:
        """Write the set's specs to a set file (``.dhset.yaml``, or ``.json``) beside the data: what each table is and what its columns
        are, never the data. Every table's spec needs its file (``path``): a file in the set file's folder (or below it) is saved relative
        to it, so the folder can move; one elsewhere keeps its full path."""
        from .setfile import write_setfile

        path = pathlib.Path(path)
        here = path.parent.resolve()
        spec = self.to_spec()
        for t in spec.tables():
            if not t.path:
                raise ValueError(f"{t.name}: its spec has no file (path=), so the set file could not find it again")
            p = pathlib.Path(t.path)
            p = (p if p.is_absolute() else pathlib.Path.cwd() / p).resolve()
            try:
                t.path = p.relative_to(here).as_posix()
            except ValueError:  # not under the set file's folder
                t.path = p.as_posix()
        out = write_setfile(path, spec)
        for t in spec.tables():  # the specs in the set keep full paths
            t.path = (here / t.path).as_posix()
        return out

    @classmethod
    def open(cls, path, *, accept_new: bool = True, table_names: str = "spec") -> DrillholeSet:
        """A set from a set file: its tables read as their specs say, from beside the file. A table with columns its spec does not know
        has them guessed (``accept_new``; they are listed in ``review``); a table that lost columns its spec names raises
        `SpecMismatch`, saying which. ``table_names="file"`` names the tables by their file names (collar.csv) instead of their specs'."""
        from .setfile import read_setfile
        from .tables import read_table

        path = pathlib.Path(path)
        spec = read_setfile(path)
        if table_names == "file":
            for t in spec.tables():
                t.name = pathlib.PurePath(t.path).name
        frames, changes = {}, {}
        for t in spec.tables():
            df, _ = read_table(path.parent / t.path, t.parse)
            change = t.check(df)
            if change["missing"] or (change["new"] and not accept_new):
                changes[t.name] = change
            frames[t.name] = df
        if changes:
            lines = [f"{n}: " + "; ".join(f"{k} {', '.join(v)}" for k, v in c.items() if v) for n, c in changes.items()]
            raise SpecMismatch(f"{path.name}: the tables changed since the set was saved ({' | '.join(lines)}); guess them again "
                               "(guess_table) and save the set", changes)
        review = {}
        for t in spec.tables():
            new = t.check(frames[t.name])["new"]
            for c in new:
                dtype = value_dtype(c, frames[t.name][c])
                t.columns.append(ColumnSpec(c, dtype, unit=unit_of(c) if dtype == "number" else None))
            if new:
                review[t.name] = new
        pair = lambda t: (t, frames[t.name])
        s = cls.from_specs(spec.name, collars=pair(spec.collars), surveys=pair(spec.surveys) if spec.surveys else None,
                           intervals={t.name: pair(t) for t in spec.intervals}, points={t.name: pair(t) for t in spec.points},
                           dip_positive_down=spec.dip_positive_down, crs=spec.crs)
        for t in spec.tables():  # paths as found, so a later save writes them again
            t.path = (path.parent / t.path).as_posix()
        s.review = review
        return s

    @classmethod
    def onboard(cls, folder, files: Sequence[str], name: str | None = None, *, edits: Mapping[str, dict] | None = None,
                threshold: float = REVIEW_BELOW, dip_positive_down: bool | None = None, templates: Sequence[TableSpec] = ()) -> DrillholeSet:
        """Read, guess and confirm a set's tables in one call: one collar table, at most one survey table, and interval and point tables
        (``files``, in ``folder``). When a guess is unsure (a column under ``threshold``, a missing role, a table that is not drillhole data)
        it raises `NeedsReview`, saying which columns, with the guesses; pass ``edits={file: {column: {...}}}`` to confirm them (``{}``
        accepts a guess). Tables are named by their file names without the extension."""
        from .tables import read_table

        folder = pathlib.Path(folder)
        edits = dict(edits or {})
        read = {f: read_table(folder / f) for f in files}
        kinds = {f: table_kind(df) for f, (df, _) in read.items()}
        collar_files = [f for f, k in kinds.items() if k == "collars"]
        if len(collar_files) != 1:
            raise ValueError(f"a set has one collar table; found {len(collar_files)} among {', '.join(files)}"
                             + (" (use group_tables for several sets)" if collar_files else ""))
        cfile = collar_files[0]
        known = hole_ids(read[cfile][0])
        guesses = {f: guess_table(df, pathlib.Path(f).stem, templates=templates, known_holes=None if f == cfile else known)
                   for f, (df, _) in read.items()}
        unsure = {}
        for f, g in guesses.items():
            asked = set(edits.get(f, {}))
            todo = [c for c in g.needs_review(threshold) if c not in asked]
            todo += [f"<no {ROLE_LABELS[r]} column>" for r in g.missing_roles() if r not in {e.get("role") for e in edits.get(f, {}).values()}]
            if g.kind not in DRILLHOLE_KINDS:
                todo = [f"<not drillhole data: {g.kind or 'no hole id'}>"]
            if todo:
                unsure[f] = todo
        if [f for f, k in kinds.items() if k == "surveys"][1:]:
            unsure["(set)"] = ["<more than one survey table>"]
        if unsure:
            lines = []
            for f, cols in unsure.items():
                g = guesses.get(f)
                for c in cols:
                    why = next((("; ".join(x.reasons)) for x in (g.columns if g else []) if x.name == c), "")
                    lines.append(f"  {f}: {c}" + (f" ({why})" if why else ""))
            example = next(((f, c) for f, cols in unsure.items() for c in cols if not c.startswith("<")), None)
            hint = f", e.g. edits={{{example[0]!r}: {{{example[1]!r}: {{}}}}}}" if example else ""
            raise NeedsReview("some columns need a look before the set is made:\n" + "\n".join(lines) +
                              "\nconfirm them with edits={file: {column: {...}}} ({} accepts the guess, {'role': ...} gives one)" + hint,
                              guesses, unsure)
        specs = {f: g.to_spec(path=str((folder / f).resolve()), parse=read[f][1], edits=edits.get(f)) for f, g in guesses.items()}
        pair = lambda f: (specs[f], read[f][0])
        by = lambda k: [f for f in files if specs[f].kind == k]
        return cls.from_specs(name or specs[cfile].name, collars=pair(cfile),
                              surveys=pair(by("surveys")[0]) if by("surveys") else None,
                              intervals={specs[f].name: pair(f) for f in by("intervals")},
                              points={specs[f].name: pair(f) for f in by("points")}, dip_positive_down=dip_positive_down)


# ---- sorting tables into sets ---------------------------------------------------------------------------------------------------


def best_set(df: pd.DataFrame, sets: Sequence[DrillholeSet], role: str, *, spec: TableSpec | None = None, min_share: float = MIN_SHARE,
             margin: float = MARGIN):
    """The set a survey, interval or point table belongs to: ``(set, "", scores)``, or ``(None, why, scores)`` when unclear (no set, too
    few of its holes in any set's collars, two sets about equally, or a second survey for the set). ``scores``: (set name, share of the
    table's holes in its collars), best first."""
    scores = sorted(((s, round(s.share(df, spec), 3)) for s in sets), key=lambda x: -x[1])
    named = [(s.name, v) for s, v in scores]
    if not scores:
        return None, "there is no collar table to join (load one, or say which set it belongs to)", named
    best, share = scores[0]
    second = scores[1][1] if len(scores) > 1 else 0.0
    if share < min_share:
        return None, f"only {share:.0%} of its holes are in the collars of {best.name!r}", named
    if share - second < margin:
        return None, f"its holes are in the collars of {best.name!r} and {scores[1][0].name!r} about equally", named
    if role == "surveys" and best.surveys is not None:
        return None, f"{best.name!r} already has a survey table ({best.surveys_name})", named
    return best, "", named


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


def unique_name(name: str, taken) -> str:
    """``name``, or ``name (2)``, ``name (3)``... if it is taken (any case)."""
    taken = {t.casefold() for t in taken}
    out, k = name, 2
    while out.casefold() in taken:
        out, k = f"{name} ({k})", k + 1
    return out


def group_tables(tables: Mapping[str, pd.DataFrame], sets: Sequence[DrillholeSet] = (), *, roles: Mapping[str, str] | None = None,
                 specs: Mapping[str, TableSpec] | None = None, min_share: float = MIN_SHARE, margin: float = MARGIN,
                 dip_positive_down: bool = True) -> Grouping:
    """Sort tables (name -> DataFrame, e.g. files just loaded) into drillhole sets, the existing ``sets`` first.

    Each collar table starts a new set, named after it. A survey, interval or point table joins the set `best_set` finds; otherwise it
    is returned in ``unsure`` (nothing guessed). ``roles`` (name -> role) overrides what `drillhole_role` finds, e.g. after the user said
    what a table is; ``specs`` (name -> `TableSpec`) gives tables their confirmed specs (and so their roles).
    ``dip_positive_down`` is given to the new sets."""
    roles, specs = dict(roles or {}), dict(specs or {})
    out = list(sets)
    unsure, other = [], {}
    found = {}
    for name, df in tables.items():
        role = roles.get(name) or (specs[name].kind if name in specs else drillhole_role(df))
        if role not in ROLES:
            other[name] = "not drillhole data (no hole id with from/to, depth, or a collar position)" if role is None else f"unknown role {role!r}"
            continue
        found[name] = role
    for name, role in found.items():          # collars first: they make the sets the other tables join
        if role == "collars":
            s = DrillholeSet(unique_name(name, {s.name for s in out}), tables[name], dip_positive_down=dip_positive_down, collars_name=name,
                             specs={name: specs[name]} if name in specs else {})
            out.append(s)
    order = sorted((n for n, r in found.items() if r != "collars"), key=lambda n: ("surveys", "intervals", "points").index(found[n]))
    for name in order:
        role, df = found[name], tables[name]
        target, why, scores = best_set(df, out, role, spec=specs.get(name), min_share=min_share, margin=margin)
        if target is None:
            unsure.append(Unsure(name, role, scores, why))
            continue
        target.add(name, df, role, spec=specs.get(name))
    return Grouping(out, unsure, other)
