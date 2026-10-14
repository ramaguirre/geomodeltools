"""Finding a table's columns by their usual names (case-insensitive). Shared by `desurvey` and geoview."""
from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

XYZ_CANDIDATES = [
    ("X", "Y", "Z"), ("XC", "YC", "ZC"), ("XCENTRE", "YCENTRE", "ZCENTRE"), ("XCENTER", "YCENTER", "ZCENTER"),
    ("CENTROID_X", "CENTROID_Y", "CENTROID_Z"), ("EAST", "NORTH", "RL"), ("EASTING", "NORTHING", "ELEVATION"),
]


def find_columns(df: pd.DataFrame, candidates: Sequence[tuple[str, str, str]]) -> tuple[str, str, str] | None:
    """The first triple of `candidates` whose three names are all columns of `df`, as spelled in `df`; else None."""
    upper = {str(c).upper(): c for c in df.columns}
    for names in candidates:
        if all(n in upper for n in names):
            return tuple(upper[n] for n in names)  # type: ignore[return-value]
    return None


def find_column(df: pd.DataFrame, given: str | None, candidates: Sequence[str], what: str, required: bool = True) -> str | None:
    """`given` if passed, else the first candidate present (case-insensitive)."""
    if given is not None:
        if given not in df.columns:
            raise KeyError(f"{what} column {given!r} not in {list(df.columns)}")
        return given
    upper = {str(c).upper(): c for c in df.columns}
    for c in candidates:
        if c in upper:
            return upper[c]
    if required:
        raise ValueError(f"could not find the {what} column in {list(df.columns)}; pass it explicitly")
    return None
