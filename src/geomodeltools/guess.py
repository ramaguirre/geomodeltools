"""Guessing what a table is and what its columns are, with how sure the guess is and why: the first step of guess -> review -> confirm.

    g = guess_table(assays, "intervals.csv", known_holes=hole_ids(collars))
    print(g)                       # kind, then per column: role, dtype, unit, confidence and the reasons
    g.needs_review()               # the columns the guess is not sure about (confidence under 0.8)
    spec = g.to_spec(edits={"CU": {"unit": "%"}, "OBS": {"selected": False}})    # a `TableSpec`: what the table is, as confirmed

Names are matched first (the lists in `columns`, any case), then the values are checked: azimuths within 0-360, dips within
+-90, FROM before TO, one row per hole in a collar table, hole ids that a collar table knows. A saved spec (a *template*) whose
columns are those of the table is used as it is, with full confidence.

The table's kind follows the rules geoview and `drillholesets` have always used: hole id + from + to: **intervals**; hole id +
depth + azimuth + dip and no X/Y/Z: **surveys**; hole id + depth and no X/Y/Z: **points** (depths along the holes); hole id + X/Y/Z
with one row per hole: **collars**; X/Y/Z and block sizes: **blocks**; X/Y/Z: **xyz** (free points, also those carrying a hole id).
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .columns import (DIP_CANDIDATES, GENERIC_NAMES, HOLE_CANDIDATES, ROLE_CANDIDATES, SIZE_CANDIDATES, XYZ_CANDIDATES, find_column,
                      find_columns)
from .schema import KINDS, REQUIRED, ROLE_LABELS, ROLES, ColumnSpec, TableSpec, parse_dates, role_dtype, signature
from .tables import ParseOptions

#: a guess under this confidence asks to be reviewed
REVIEW_BELOW = 0.8
_SAMPLE = 20_000
_HEX = re.compile(r"#[0-9a-fA-F]{6}")
#: how assay tables write a value below (or above) the detection limit, or a missing sample
_LIMIT = re.compile(r"[<>]\s*-?\d+([.,]\d+)?|ND|N/D|NA|N/A|LD|BDL|IS|NS|SM|S/M|-+|\*+")
_DATE_NAME = re.compile(r"DATE|FECHA", re.I)
_DATE_VALUE = re.compile(r"\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}([ T]\d{1,2}:\d{2}(:\d{2})?)?")
_UNITS = [(re.compile(r"(^|[_ .\-(])(PCT|PERC|PERCENT|%)\)?$", re.I), "%"), (re.compile(r"(^|[_ .\-(])PPM\)?$", re.I), "ppm"),
          (re.compile(r"(^|[_ .\-(])PPB\)?$", re.I), "ppb"), (re.compile(r"(^|[_ .\-(])(GPT|G_T|G/T|GT)\)?$", re.I), "g/t")]
_ROLE_UNITS = {"x": "m", "y": "m", "z": "m", "length": "m", "depth": "m", "from": "m", "to": "m", "dx": "m", "dy": "m", "dz": "m",
               "azimuth": "deg", "dip": "deg"}


@dataclass
class ColumnGuess:
    """What one column looks like: its ``role`` (None for a value column), ``dtype``, ``unit``, how sure (``confidence``, 0 to 1),
    ``reasons`` (in plain words) and the ``alternatives`` considered ((role or dtype, confidence))."""
    name: str
    role: str | None
    dtype: str
    unit: str | None = None
    confidence: float = 1.0
    reasons: list = field(default_factory=list)
    alternatives: list = field(default_factory=list)
    source: str = "guess"
    selected: bool = True


@dataclass
class TableGuess:
    """What a table looks like: its ``kind`` (None when it is no table geomodeltools knows), the score of each kind (``kind_scores``),
    the ``columns`` (a `ColumnGuess` each, in table order), and the table's column ``signature``."""
    name: str
    kind: str | None
    kind_scores: dict
    columns: list
    signature: str
    template: str | None = None

    def column(self, role: str) -> str | None:
        return next((c.name for c in self.columns if c.role == role), None)

    def roles(self) -> dict:
        return {c.role: c.name for c in self.columns if c.role}

    def missing_roles(self) -> list:
        """The required roles of the kind that no column plays."""
        if self.kind is None:
            return []
        have = self.roles()
        return [r for r in REQUIRED[self.kind] if r not in have]

    def needs_review(self, threshold: float = REVIEW_BELOW) -> list:
        """The columns whose guess is under ``threshold`` (the ones to look at before confirming)."""
        return [c.name for c in self.columns if c.confidence < threshold]

    def is_sure(self, threshold: float = REVIEW_BELOW) -> bool:
        """Is there nothing to review: a kind, every required role, and every column at or above ``threshold``?"""
        return self.kind is not None and not self.missing_roles() and not self.needs_review(threshold)

    def to_spec(self, name: str | None = None, *, path: str | None = None, parse: ParseOptions | dict | None = None,
                edits: dict | None = None, options: dict | None = None) -> TableSpec:
        """The confirmed `TableSpec`: the guess, with ``edits`` applied (see `TableSpec.edit`). Raises if the table has no kind, or lacks a
        required role after the edits."""
        if self.kind is None:
            raise ValueError(f"{self.name or 'table'}: not a table geomodeltools recognises; guess it again with kind= to say what it is")
        cols = [ColumnSpec(c.name, c.dtype, role=c.role, unit=c.unit, selected=c.selected, source=c.source) for c in self.columns]
        spec = TableSpec(name or self.name or "table", self.kind, cols,
                         parse if isinstance(parse, ParseOptions) else ParseOptions.from_dict(parse), path, self.signature, dict(options or {}))
        if edits:
            spec = spec.edit(edits)
        missing = spec.missing_roles()
        if missing:
            raise ValueError(f"{spec.name}: a {spec.kind} table needs a column for {', '.join(ROLE_LABELS[r] for r in missing)}; "
                             f"give it with edits={{'<column>': {{'role': '{missing[0]}'}}}}")
        return spec

    def to_dict(self) -> dict:
        """The guess as plain data (an import window's JSON)."""
        return {"name": self.name, "kind": self.kind, "kind_scores": self.kind_scores, "signature": self.signature, "template": self.template,
                "columns": [vars(c) | {"alternatives": [list(a) for a in c.alternatives]} for c in self.columns]}

    def __str__(self) -> str:
        head = f"{self.name or 'table'}: {self.kind or 'not recognised'}" + (f" ({self.kind_scores.get(self.kind, 0):.2f})" if self.kind else "")
        if self.template:
            head += f", from the saved settings {self.template!r}"
        width = max([len(c.name) for c in self.columns] + [4])
        lines = [head]
        for c in self.columns:
            flag = "  <- review" if c.confidence < REVIEW_BELOW else ""
            what = f"role={c.role}" if c.role else ""
            lines.append(f"  {c.name:<{width}}  {what:<13} {c.dtype:<8} {c.unit or '':<5} {c.confidence:.2f}  {'; '.join(c.reasons)}{flag}")
        missing = self.missing_roles()
        if missing:
            lines.append(f"  missing: {', '.join(ROLE_LABELS[r] for r in missing)}")
        return "\n".join(lines)

    __repr__ = __str__


# ---- recognising the kind -------------------------------------------------------------------------------------------------------


def _exact(df: pd.DataFrame) -> dict:
    """The columns found by their usual names: role -> column, plus "xyz" and "size" triples."""
    found = {r: find_column(df, None, cands, "", required=False) for r, cands in ROLE_CANDIDATES.items()}
    found = {r: c for r, c in found.items() if c is not None}
    xyz, size = find_columns(df, XYZ_CANDIDATES), find_columns(df, SIZE_CANDIDATES)
    if xyz:
        found["xyz"] = xyz
    if size:
        found["size"] = size
    return found


def _repeats(df: pd.DataFrame, hole) -> bool:
    """Several rows per hole, each with a position: free points carrying a hole id, not a collar table."""
    n = len(df)
    return n >= 20 and df[hole].nunique() < 0.9 * n


def table_kind(df: pd.DataFrame, found: dict | None = None) -> str | None:
    """The kind of a table from its column names (see the module's rules), or None (no hole id, no X/Y/Z)."""
    f = _exact(df) if found is None else found
    hole, xyz = f.get("hole"), f.get("xyz")
    if hole is not None:
        if "from" in f and "to" in f:
            return "intervals"
        if xyz is None and {"depth", "azimuth", "dip"} <= f.keys():
            return "surveys"
        if xyz is None and "depth" in f:
            return "points"
        if xyz is not None:
            return "xyz" if _repeats(df, hole) else "collars"
    if xyz is not None:
        return "blocks" if "size" in f else "xyz"
    return None


# ---- values -----------------------------------------------------------------------------------------------------------------------


def _numbers(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        return s.astype(float)
    return pd.to_numeric(s.astype("string").str.strip().str.replace(",", ".", regex=False), errors="coerce")


def _filled(s: pd.Series) -> pd.Series:
    v = s.dropna()
    if v.dtype == object or pd.api.types.is_string_dtype(v):
        v = v.astype(str).str.strip()
        v = v[v != ""]
    return v


def _dtype(name: str, s: pd.Series) -> tuple:
    """(dtype, confidence, reasons, alternatives) of a value column."""
    v = _filled(s)
    if v.empty:
        return "number", 0.9, ["empty column: left out"], [("category", 0.5)]
    if _DATE_NAME.search(name) and not pd.api.types.is_datetime64_any_dtype(v):
        read = parse_dates(v.head(2000)).notna().mean()
        if read >= 0.9:
            mixed = pd.to_numeric(v.head(2000).astype(str), errors="coerce").between(20_000, 80_000).any()
            return "date", 0.85 if mixed else 0.9, ["dates (name says so)" + ("; some written as Excel day numbers" if mixed else "")], [("text", 0.1)]
    if pd.api.types.is_bool_dtype(v):
        return "category", 0.9, ["true/false values"], []
    if pd.api.types.is_datetime64_any_dtype(v):
        return "date", 0.99, ["dates"], []
    if pd.api.types.is_numeric_dtype(v):
        return "number", 0.97, ["numbers"], []
    text = v.astype(str)
    if text.str.fullmatch(_HEX.pattern).all():
        return "colour", 0.99, ["every value is #RRGGBB"], []
    numeric = _numbers(text).notna()
    share = numeric.mean()
    if share == 1:
        return "number", 0.95, ["numbers (written as text)"], []
    bad = text[~numeric]
    limits = bad.str.fullmatch(_LIMIT.pattern, case=False)
    if share >= 0.95 or (share >= 0.5 and limits.all()):          # assays: a few "<0.01", "ND", "-" among the numbers
        examples = ", ".join(repr(x) for x in list(dict.fromkeys(bad))[:2])
        what = "detection-limit codes" if limits.all() else "not numbers"
        return "number", 0.7, [f"{share:.0%} numbers; {len(bad):,} values are {what} (e.g. {examples}) and would become empty"], [("category", 0.3)]
    if text.head(200).str.fullmatch(_DATE_VALUE.pattern).mean() >= 0.9 and parse_dates(text.head(500)).notna().mean() >= 0.9:
        return "date", 0.9, ["dates"], [("text", 0.1)]
    n, distinct = len(text), text.nunique()
    if share >= 0.5:
        return "category", 0.6, [f"mostly numbers ({share:.0%}) but not all"], [("number", 0.4)]
    if n >= 5 and distinct > 0.5 * n and text.str.len().mean() > 12:
        return "text", 0.85, [f"free text, {distinct / n:.0%} of the values different"], [("category", 0.15)]
    return "category", 0.9, [f"{distinct:,} distinct values"], [("text", 0.1)]


def unit_of(name: str) -> str | None:
    """The unit a value column's name gives (CU_PCT: %, AU_PPM: ppm, AU_GPT: g/t), or None."""
    for pattern, unit in _UNITS:
        if pattern.search(name):
            return unit
    return None


def value_dtype(name: str, s: pd.Series) -> str:
    """What a value column holds: "number", "category", "text", "date" or "colour" (the guess, without its confidence)."""
    return _dtype(str(name), s)[0]


def _tokens(name: str) -> set:
    return {t for t in re.split(r"[^A-Z0-9ÁÉÍÓÚÑ]+", str(name).upper()) if t}


# ---- the guess --------------------------------------------------------------------------------------------------------------------


def _role_guesses(df: pd.DataFrame, kind: str, found: dict, known_holes: set | None) -> dict:
    """role -> (column, confidence, reasons) for the roles of ``kind``."""
    out = {}
    roles = ROLES[kind]
    if "x" in roles and "xyz" in found:
        for r, c in zip("xyz", found["xyz"]):
            out[r] = [c, 0.9, [f"name matches {r.upper()} of {'/'.join(found['xyz'])}"]]
    if "dx" in roles and "size" in found:
        for r, c in zip(("dx", "dy", "dz"), found["size"]):
            out[r] = [c, 0.9, ["name matches a block size"]]
    used = {c for c, *_ in out.values()}
    for r in roles:
        if r in out or r not in ROLE_CANDIDATES:
            continue
        c = found.get(r)
        if c is not None and c not in used:
            generic = str(c).strip().upper() in GENERIC_NAMES
            out[r] = [c, 0.6 if generic else 0.9, [f"name matches {str(c).upper()}" + (" (a name used for many things)" if generic else "")]]
            used.add(c)
    for r in roles:  # a usual name as part of a longer one (HOLE_NAME, AZIMUTH_GRID): weaker
        if r in out or r not in ROLE_CANDIDATES:
            continue
        for c in df.columns:
            if c in used:
                continue
            hit = next((cand for cand in ROLE_CANDIDATES[r] if len(cand) >= 3 and cand not in GENERIC_NAMES and cand in _tokens(c)), None)
            if hit:
                out[r] = [c, 0.55, [f"name contains {hit}"]]
                used.add(c)
                break
    sample = df.head(_SAMPLE)
    for r, g in out.items():
        c = g[0]
        if r == "hole":
            ids = sample[c].dropna().astype(str).str.strip()
            if kind == "collars":
                if ids.is_unique:
                    g[1] += 0.05
                    g[2].append("one row per hole")
                else:
                    g[1] -= 0.2
                    g[2].append(f"{int(ids.duplicated().sum()):,} hole ids repeat")
            if known_holes:
                uniq = set(ids)
                share = len(uniq & known_holes) / len(uniq) if uniq else 0.0
                g[2].append(f"{share:.0%} of its holes are in the collars")
                g[1] += 0.08 if share >= 0.5 else -0.3
            continue
        x = _numbers(sample[c])
        ok = x.notna().mean() if len(x) else 0.0
        if ok < 0.95:
            g[1] -= 0.5
            g[2].append(f"only {ok:.0%} numbers")
            continue
        x = x.dropna()
        if r == "azimuth":
            inside = x.between(-360, 360).mean()           # -27 is 333: both are written
            g[1] += 0.05 if inside == 1 else -0.4
            g[2].append(("values within 0-360" if (x >= 0).all() else "values within -360..360") if inside == 1
                        else f"{1 - inside:.0%} of the values outside -360..360")
        elif r == "dip":
            inside = x.between(-90, 90).mean()
            g[1] += 0.05 if inside == 1 else -0.4
            g[2].append("values within -90..90" if inside == 1 else f"{1 - inside:.0%} of the values outside -90..90")
        elif r in ("length", "depth"):
            low = (x < 0).mean()
            g[1] += 0.05 if low == 0 else -0.3
            if low:
                g[2].append(f"{low:.0%} negative values")
    if "from" in out and "to" in out:
        f, t = _numbers(sample[out["from"][0]]), _numbers(sample[out["to"][0]])
        both = f.notna() & t.notna()
        share = (f[both] < t[both]).mean() if both.any() else 0.0
        for r in ("from", "to"):
            out[r][1] += 0.05 if share >= 0.95 else -0.3
            out[r][2].append(f"{out['from'][0]} < {out['to'][0]} in {share:.0%} of rows")
    return {r: (c, float(np.clip(conf, 0.05, 0.99)), reasons) for r, (c, conf, reasons) in out.items()}


def _from_template(df: pd.DataFrame, name: str, t: TableSpec) -> TableGuess:
    by = {c.name: c for c in t.columns}
    cols = [ColumnGuess(str(c), by[str(c)].role, by[str(c)].dtype, by[str(c)].unit, 1.0, [f"saved settings {t.name!r}"], source="template",
                        selected=by[str(c)].selected) for c in df.columns]
    return TableGuess(name, t.kind, {t.kind: 1.0}, cols, signature(df.columns), template=t.name)


def guess_table(df: pd.DataFrame, name: str = "", *, kind: str | None = None, templates: Iterable[TableSpec] = (),
                known_holes: Iterable[str] | None = None) -> TableGuess:
    """What ``df`` is and what each of its columns is, with a confidence and the reasons. ``kind`` says what the table is instead of
    guessing it (its roles are then guessed for that kind). ``templates``: saved specs; one whose columns are the table's is used as it
    is. ``known_holes``: the hole ids of the collar table the table should belong to (hole id guesses are checked against them)."""
    if kind is not None and kind not in KINDS:
        raise ValueError(f"kind {kind!r} is not one of {', '.join(KINDS)}")
    sig = signature(df.columns)
    for t in templates:
        if t.signature == sig and (kind is None or t.kind == kind) and {c.name for c in t.columns} == {str(c) for c in df.columns}:
            return _from_template(df, name, t)
    found = _exact(df)
    detected = table_kind(df, found)
    chosen = kind or detected
    known = {str(h).strip() for h in known_holes} if known_holes is not None else None
    scores = {}
    for k in KINDS:
        rg = _role_guesses(df, k, found, known if k != "collars" else None)
        req = REQUIRED[k]
        scores[k] = round(sum(rg[r][1] for r in req if r in rg) / len(req), 2)
    if chosen is None:
        cols = [ColumnGuess(str(c), None, _dtype(str(c), df[c])[0], None, 0.5, ["no hole id or X/Y/Z found"]) for c in df.columns]
        return TableGuess(name, None, scores, cols, sig)
    roles = _role_guesses(df, chosen, found, known if chosen != "collars" else None)
    role_of = {c: (r, conf, reasons) for r, (c, conf, reasons) in roles.items()}
    cols = []
    for c in df.columns:
        if c in role_of:
            r, conf, reasons = role_of[c]
            cols.append(ColumnGuess(str(c), r, role_dtype(r), _ROLE_UNITS.get(r), round(conf, 2), reasons))
        else:
            dtype, conf, reasons, alts = _dtype(str(c), df[c])
            cols.append(ColumnGuess(str(c), None, dtype, unit_of(str(c)) if dtype == "number" else None, round(conf, 2), reasons, alts,
                                    selected=not _filled(df[c]).empty))
    if kind is not None and kind != detected:
        scores[kind] = max(scores.get(kind, 0.0), 0.5)
    return TableGuess(name, chosen, scores, cols, sig)


def confirm(guess: TableGuess, edits: dict | None = None, **kw) -> TableSpec:
    """The `TableSpec` of a guess with the user's ``edits`` (`TableGuess.to_spec`)."""
    return guess.to_spec(edits=edits, **kw)


# ---- dip convention -------------------------------------------------------------------------------------------------------------


def detect_dip_convention(df: pd.DataFrame, dip: str | None = None, hole: str | None = None) -> dict:
    """Which way a table's dips point, from their signs: ``positive_down`` is True (dips are positive downwards), False (negative
    downwards, as Leapfrog's interval exports write them) or None (cannot tell: all zero, no dip column, or both signs about equally);
    ``reason`` says why, in plain words. Holes count once each (their mean dip), so a few upward holes do not change the answer: most
    exploration holes go down. ``dip`` and ``hole`` name the columns (found by their usual names otherwise)."""
    dip = dip if dip is not None and dip in df.columns else find_column(df, None, DIP_CANDIDATES, "", required=False)
    if dip is None:
        return {"positive_down": None, "reason": "no dip column"}
    values = pd.to_numeric(df[dip], errors="coerce")
    ok = values.notna()
    hole = hole if hole is not None and hole in df.columns else find_column(df, None, HOLE_CANDIDATES, "", required=False)
    per_hole = values[ok].groupby(df.loc[ok, hole]).mean() if hole else values[ok]
    pos, neg = int((per_hole > 1e-9).sum()), int((per_hole < -1e-9).sum())
    total = pos + neg
    if total == 0:
        return {"positive_down": None, "reason": "all dips are zero: cannot tell which way they point"}
    if neg == 0:
        return {"positive_down": True, "reason": "all dips are positive: read as positive-down"}
    if pos == 0:
        return {"positive_down": False, "reason": "all dips are negative: read as negative-down (as Leapfrog exports them)"}
    if pos / total >= 0.8:
        return {"positive_down": True, "reason": f"{pos} of {total} holes have positive dips: read as positive-down (the others are probably upward holes)"}
    if pos / total <= 0.2:
        return {"positive_down": False, "reason": f"{neg} of {total} holes have negative dips: read as negative-down (the others are probably upward holes)"}
    return {"positive_down": None, "reason": f"dips have both signs ({pos} holes positive, {neg} negative): cannot tell, read as negative-down (as Leapfrog); "
                                              "choose the convention if the holes point the wrong way"}
