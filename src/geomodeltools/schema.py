"""What a table's columns are: the declarative, saved form of "this column is the hole id, that one is copper in %, this one is
left out". `guess.guess_table` proposes one, a person (or an app's import window) corrects it, and a `DrillholeSet` is built from
it, so nothing downstream guesses column names again.

    ColumnSpec("SONDAJE", "text", role="hole")
    ColumnSpec("CU", "number", unit="%")
    ColumnSpec("OBS", "text", selected=False)          # left out, but still known to be free text

    TableSpec("assays", "intervals", columns=[...], path="intervals.csv", parse=ParseOptions(delimiter=";"))
    SetSpec("Centinela 2025", collars=..., surveys=..., intervals=[...])   # what a .dhset.yaml file holds (`setfile`)

A spec holds column names and settings only, never data. Column names are kept as written in the file (nothing is renamed).
"""
from __future__ import annotations

import hashlib
import re
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .tables import ParseOptions

#: what a table is. Drillhole tables: ``collars`` (one row per hole, with its position), ``surveys`` (direction at depths along each
#: hole), ``intervals`` (from/to along a hole: assays, lithology...), ``points`` (one depth along a hole: structures, samples).
#: Other tables: ``xyz`` (free points) and ``blocks`` (a block model: block centres and sizes).
KINDS = ("collars", "surveys", "intervals", "points", "xyz", "blocks")
DRILLHOLE_KINDS = ("collars", "surveys", "intervals", "points")
#: the part each column can play, per kind, in the order an import window lists them
ROLES = {
    "collars": ("hole", "x", "y", "z", "length", "azimuth", "dip"),
    "surveys": ("hole", "depth", "azimuth", "dip"),
    "intervals": ("hole", "from", "to"),
    "points": ("hole", "depth"),
    "xyz": ("x", "y", "z"),
    "blocks": ("x", "y", "z", "dx", "dy", "dz"),
}
#: the roles a table of each kind cannot do without
REQUIRED = {"collars": ("hole", "x", "y"), "surveys": ("hole", "depth", "azimuth", "dip"), "intervals": ("hole", "from", "to"),
            "points": ("hole", "depth"), "xyz": ("x", "y"), "blocks": ("x", "y", "z")}
ROLE_LABELS = {"hole": "Hole id", "x": "X", "y": "Y", "z": "Z", "length": "Hole length", "azimuth": "Azimuth", "dip": "Dip", "depth": "Depth",
               "from": "From", "to": "To", "dx": "DX (block size)", "dy": "DY", "dz": "DZ"}
#: what a column holds. ``category``: a few repeated values (lithology); ``text``: free text (comments); ``colour``: "#RRGGBB".
DTYPES = ("number", "category", "text", "date", "colour")
DTYPE_LABELS = {"number": "Number", "category": "Category", "text": "Text", "date": "Date", "colour": "Colour #RRGGBB"}
#: the name each role's column has in the tables `desurvey` is given (see `canonical`)
CANONICAL = {"hole": "HOLEID", "x": "X", "y": "Y", "z": "Z", "length": "LENGTH", "azimuth": "AZIMUTH", "dip": "DIP", "depth": "DEPTH",
             "from": "FROM", "to": "TO", "dx": "DX", "dy": "DY", "dz": "DZ"}
SOURCES = ("guess", "template", "user")


def role_dtype(role: str) -> str:
    """The dtype a role's column always has: text for the hole id, a number for everything else."""
    return "text" if role == "hole" else "number"


def signature(columns) -> str:
    """The key of a table's layout: its column names, whatever their order or case. The same key geoview's saved import recipes use."""
    names = sorted({str(c).strip().lower() for c in columns})
    return hashlib.sha1("\x1f".join(names).encode("utf-8")).hexdigest()[:16]


@dataclass
class ColumnSpec:
    """One column: its ``name`` as written in the file, what it holds (``dtype``), the part it plays (``role``, None for a value column),
    its ``unit`` ("m", "%", "ppm", "g/t", "deg"...), whether it is kept (``selected``), and who said so (``source``: "guess", "template"
    or "user")."""
    name: str
    dtype: str
    role: str | None = None
    unit: str | None = None
    selected: bool = True
    source: str = "guess"

    def __post_init__(self):
        self.name = str(self.name)
        if self.dtype not in DTYPES:
            raise ValueError(f"column {self.name!r}: dtype {self.dtype!r} is not one of {', '.join(DTYPES)}")
        if self.role is not None and self.role not in CANONICAL:
            raise ValueError(f"column {self.name!r}: role {self.role!r} is not one of {', '.join(CANONICAL)}")
        if self.source not in SOURCES:
            raise ValueError(f"column {self.name!r}: source {self.source!r} is not one of {', '.join(SOURCES)}")

    def to_dict(self) -> dict:
        d = {"name": self.name, "dtype": self.dtype}
        if self.role:
            d["role"] = self.role
        if self.unit:
            d["unit"] = self.unit
        if not self.selected:
            d["selected"] = False
        if self.source != "guess":
            d["source"] = self.source
        return d

    @classmethod
    def from_dict(cls, d: dict) -> ColumnSpec:
        return cls(name=d["name"], dtype=d.get("dtype", "number"), role=d.get("role"), unit=d.get("unit"), selected=bool(d.get("selected", True)),
                   source=d.get("source", "guess"))


@dataclass
class TableSpec:
    """A table: its ``name`` (in a set and in messages), ``kind`` (one of `KINDS`), its ``columns``, how its file is read (``parse``), the
    file (``path``, relative to the set file that lists it; None for a template), its column ``signature``, and ``options``: settings an
    app keeps with the table (geoview's point size, a block size mode...), carried untouched."""
    name: str
    kind: str
    columns: list = field(default_factory=list)
    parse: ParseOptions = field(default_factory=ParseOptions)
    path: str | None = None
    signature: str = ""
    options: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"table {self.name!r}: kind {self.kind!r} is not one of {', '.join(KINDS)}")
        if isinstance(self.parse, dict):
            self.parse = ParseOptions.from_dict(self.parse)
        self.columns = [c if isinstance(c, ColumnSpec) else ColumnSpec.from_dict(c) for c in self.columns]
        seen = {}
        for c in self.columns:
            if c.role is None:
                continue
            if c.role not in ROLES[self.kind]:
                raise ValueError(f"table {self.name!r}: a {self.kind} table has no role {c.role!r} (column {c.name!r}); its roles are "
                                 f"{', '.join(ROLES[self.kind])}")
            if c.role in seen:
                raise ValueError(f"table {self.name!r}: columns {seen[c.role]!r} and {c.name!r} are both the {ROLE_LABELS[c.role]}")
            seen[c.role] = c.name
        if not self.signature:
            self.signature = signature(c.name for c in self.columns)

    # ---- what it says ------------------------------------------------------------------------------------------------------------

    def column(self, role: str) -> str | None:
        """The column that plays ``role``, or None."""
        return next((c.name for c in self.columns if c.role == role), None)

    def roles(self) -> dict:
        """role -> column, for the roles this table has."""
        return {c.role: c.name for c in self.columns if c.role}

    def missing_roles(self) -> list:
        """The required roles (`REQUIRED`) no column plays."""
        have = self.roles()
        return [r for r in REQUIRED[self.kind] if r not in have]

    def attributes(self) -> list:
        """The kept columns that play no role (the values: assays, lithology...), in table order."""
        return [c.name for c in self.columns if c.role is None and c.selected]

    def spec(self, name: str) -> ColumnSpec | None:
        return next((c for c in self.columns if c.name == name), None)

    def of_dtype(self, *dtypes: str) -> list:
        """The kept value columns of these dtypes (e.g. ``"category"`` for the columns to colour by category)."""
        return [c.name for c in self.columns if c.role is None and c.selected and c.dtype in dtypes]

    # ---- changing it ------------------------------------------------------------------------------------------------------------

    def edit(self, edits: dict) -> TableSpec:
        """A copy with columns changed: ``{"CU": {"unit": "%"}, "OBS": {"selected": False}, "BHID": {"role": "hole"}}``. Giving a column a role
        takes it from the column that had it; an empty edit (``{}``) just confirms the column. Every column named is marked ``source="user"``."""
        cols = [ColumnSpec(**vars(c)) for c in self.columns]
        by_name = {c.name: c for c in cols}
        for name, change in (edits or {}).items():
            if name not in by_name:
                raise KeyError(f"table {self.name!r} has no column {name!r} (it has {', '.join(by_name)})")
            unknown = set(change) - {"dtype", "role", "unit", "selected"}
            if unknown:
                raise ValueError(f"column {name!r}: cannot change {', '.join(sorted(unknown))} (only dtype, role, unit, selected)")
            c = by_name[name]
            if "role" in change and change["role"] != c.role:
                role = change["role"]
                if role:
                    for other in cols:
                        if other.role == role:
                            other.role = None
                c.role = role or None
                if role:
                    c.dtype = role_dtype(role)
            for key in ("dtype", "unit", "selected"):
                if key in change:
                    setattr(c, key, change[key])
            c.source = "user"
        return TableSpec(self.name, self.kind, cols, ParseOptions(**vars(self.parse)), self.path, self.signature, dict(self.options))

    # ---- using it ---------------------------------------------------------------------------------------------------------------

    def check(self, df: pd.DataFrame) -> dict:
        """Does the spec still fit this table? ``{"missing": columns of the spec the table no longer has, "new": columns the spec does not
        know}``; both empty when it fits."""
        have = [str(c) for c in df.columns]
        known = {c.name for c in self.columns}
        return {"missing": [c.name for c in self.columns if c.name not in set(have)], "new": [c for c in have if c not in known]}

    def apply(self, df: pd.DataFrame, report: bool = False):
        """The table as the spec describes it: the columns left out dropped, the rest typed (numbers coerced, hole ids and text stripped,
        categories as text, dates parsed, colours as "#RRGGBB"), the names as written. With ``report``, also the values that could not be
        read as their type: ``{column: (count, a few examples)}``."""
        missing = self.check(df)["missing"]
        if missing:
            raise KeyError(f"table {self.name!r}: the table has no column {', '.join(map(repr, missing))} (it has {', '.join(map(str, df.columns[:12]))})")
        out, bad = {}, {}
        cols = {str(c): c for c in df.columns}
        for c in self.columns:
            if not c.selected and c.role is None:
                continue
            raw = df[cols[c.name]]
            typed = _typed(raw, c.dtype)
            out[c.name] = typed
            if c.dtype in ("number", "date"):
                lost = raw.notna() & (raw.astype(str).str.strip() != "") & typed.isna()
                if lost.any():
                    bad[c.name] = (int(lost.sum()), list(dict.fromkeys(raw[lost].astype(str)))[:3])
        table = pd.DataFrame(out, index=df.index)
        return (table, bad) if report else table

    def canonical(self, df: pd.DataFrame, attributes: bool = True) -> pd.DataFrame:
        """The table with its role columns under the standard names (`CANONICAL`: HOLEID, X, Y, Z, FROM, TO...), then its kept value
        columns (``attributes``) under their own names: what `desurvey` is given, so it never has to find a column by its name. A value
        column whose name is a standard one is given ``<name>_original``."""
        cols = {str(c): c for c in df.columns}
        roles = self.roles()
        out = {CANONICAL[r]: df[cols[c]] for r, c in roles.items()}
        if attributes:
            for name in self.attributes():
                key = f"{name}_original" if name.upper() in {v.upper() for v in out} else name
                out[key] = df[cols[name]]
        table = pd.DataFrame(out, index=df.index).reset_index(drop=True)
        if "HOLEID" in table:
            table["HOLEID"] = table["HOLEID"].astype(str).str.strip()
        for r in roles:
            if role_dtype(r) == "number":
                table[CANONICAL[r]] = pd.to_numeric(table[CANONICAL[r]], errors="coerce")
        return table

    # ---- saved form ---------------------------------------------------------------------------------------------------------------

    def to_dict(self) -> dict:
        d = {"name": self.name, "kind": self.kind}
        if self.path:
            d["path"] = self.path
        parse = {k: v for k, v in self.parse.to_dict().items() if v != getattr(ParseOptions(), k)}
        if parse:
            d["parse"] = parse
        d["signature"] = self.signature
        if self.options:
            d["options"] = dict(self.options)
        d["columns"] = [c.to_dict() for c in self.columns]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> TableSpec:
        return cls(name=d.get("name") or d.get("path") or "table", kind=d["kind"], columns=[ColumnSpec.from_dict(c) for c in d.get("columns", [])],
                   parse=ParseOptions.from_dict(d.get("parse")), path=d.get("path"), signature=d.get("signature", ""), options=dict(d.get("options") or {}))


@dataclass
class SetSpec:
    """A drillhole set as saved (`setfile`): its tables' specs and the set's own settings. ``dip_positive_down``: None to read the
    convention from the dips' signs each time."""
    name: str
    collars: TableSpec
    surveys: TableSpec | None = None
    intervals: list = field(default_factory=list)
    points: list = field(default_factory=list)
    dip_positive_down: bool | None = None
    crs: str | None = None
    version: int = 1

    def tables(self) -> list:
        """Every table spec, collars first."""
        return [self.collars, *([self.surveys] if self.surveys else []), *self.intervals, *self.points]

    def to_dict(self) -> dict:
        d = {"version": self.version, "name": self.name}
        if self.dip_positive_down is not None:
            d["dip_positive_down"] = self.dip_positive_down
        if self.crs:
            d["crs"] = self.crs
        d["collars"] = self.collars.to_dict()
        if self.surveys:
            d["surveys"] = self.surveys.to_dict()
        if self.intervals:
            d["intervals"] = [t.to_dict() for t in self.intervals]
        if self.points:
            d["points"] = [t.to_dict() for t in self.points]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> SetSpec:
        version = int(d.get("version", 1))
        if version > 1:
            raise ValueError(f"drillhole set file version {version} is newer than this geomodeltools reads (1): update geomodeltools")
        t = TableSpec.from_dict
        return cls(name=d["name"], collars=t(d["collars"]), surveys=t(d["surveys"]) if d.get("surveys") else None,
                   intervals=[t(x) for x in d.get("intervals") or []], points=[t(x) for x in d.get("points") or []],
                   dip_positive_down=d.get("dip_positive_down"), crs=d.get("crs"), version=version)


# ---- typing values ------------------------------------------------------------------------------------------------------------------

_HEX = re.compile(r"#?[0-9a-fA-F]{6}")


def _typed(s: pd.Series, dtype: str) -> pd.Series:
    if dtype == "number":
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            return s.astype(np.float64) if not pd.api.types.is_float_dtype(s) else s
        return pd.to_numeric(s.astype("string").str.strip().str.replace(",", ".", regex=False), errors="coerce").astype(np.float64)
    if dtype == "date":
        return parse_dates(s)
    if dtype == "colour":
        t = s.astype("string").str.strip()
        return t.where(t.isna() | t.str.startswith("#"), "#" + t).astype(object)
    t = s.astype("string").str.strip()
    return t.astype(object).where(t.notna(), None) if dtype == "text" else t.fillna("").astype(object)


_EXCEL_DAYS = (20_000, 80_000)  # 1954..2119 as Excel day numbers


def parse_dates(s: pd.Series) -> pd.Series:
    """Dates from text (any common format; day first when the values say so) and from Excel day numbers (40089 = 2009-10-03), which
    database exports often mix in one column. What cannot be read becomes NaT."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    text = s.astype("string").str.strip()
    num = pd.to_numeric(text, errors="coerce")
    serial = num.between(*_EXCEL_DAYS).fillna(False).astype(bool)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = pd.to_datetime(text.where(num.isna()), errors="coerce", format="mixed", dayfirst=_dayfirst(text))
    days = pd.to_datetime(num.where(serial), unit="D", origin="1899-12-30", errors="coerce")
    return out.where(~serial, days)


def _dayfirst(s: pd.Series) -> bool:
    """Do dates written with slashes put the day first (31/12/2025, as Chilean tables do)? True when any first number is over 12."""
    v = s.dropna().astype(str).head(200).str.extract(r"^(\d{1,2})[/.-](\d{1,2})[/.-]\d{2,4}")
    if v.empty or v[0].isna().all():
        return False
    return bool((pd.to_numeric(v[0], errors="coerce") > 12).any())
