"""The usual names of a table's columns, and finding them (case-insensitive). The one home of every candidate list: `desurvey`,
`guess`, `drillholesets` and geoview all read them from here (``geomodeltools.desurvey`` re-exports the drillhole ones)."""
from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

XYZ_CANDIDATES = [
    ("X", "Y", "Z"), ("XC", "YC", "ZC"), ("XCENTRE", "YCENTRE", "ZCENTRE"), ("XCENTER", "YCENTER", "ZCENTER"),
    ("CENTROID_X", "CENTROID_Y", "CENTROID_Z"), ("EAST", "NORTH", "RL"), ("EASTING", "NORTHING", "ELEVATION"),
    # Spanish (as Chilean and other Latin American tables write them; matched in any case, accents as written)
    ("ESTE", "NORTE", "COTA"), ("ESTE", "NORTE", "ELEVACION"), ("ESTE", "NORTE", "ELEVACIÓN"), ("ESTE", "NORTE", "ELEV"),
    ("ESTE", "NORTE", "Z"), ("ESTE_COLLAR", "NORTE_COLLAR", "COTA_COLLAR"),
]
#: block sizes of a block model
SIZE_CANDIDATES = [("DX", "DY", "DZ"), ("XINC", "YINC", "ZINC"), ("SIZE_X", "SIZE_Y", "SIZE_Z"), ("XSIZE", "YSIZE", "ZSIZE")]

HOLE_CANDIDATES = ["HOLEID", "HOLE_ID", "BHID", "DHID", "HOLE", "SONDAJE", "ID"]
DEPTH_CANDIDATES = ["DEPTH", "AT", "MD", "DISTANCE", "PROF", "PROFUNDIDAD"]
LENGTH_CANDIDATES = ["MAX_DEPTH", "MAXDEPTH", "LENGTH", "DEPTH", "EOH", "TD", "LARGO", "LONGITUD", "PROFUNDIDAD_TOTAL"]
AZIMUTH_CANDIDATES = ["AZIMUTH", "AZI", "AZ", "BRG", "BEARING", "AZIMUT", "RUMBO"]
DIP_CANDIDATES = ["DIP", "INCLINACION", "INCLINACIÓN", "INCL", "BUZAMIENTO", "BUZ"]
FROM_CANDIDATES = ["FROM", "MFROM", "DESDE"]
TO_CANDIDATES = ["TO", "MTO", "HASTA"]

#: the single-column roles of drillhole tables and their usual names, in the order they are tried
ROLE_CANDIDATES = {"hole": HOLE_CANDIDATES, "depth": DEPTH_CANDIDATES, "length": LENGTH_CANDIDATES, "azimuth": AZIMUTH_CANDIDATES,
                   "dip": DIP_CANDIDATES, "from": FROM_CANDIDATES, "to": TO_CANDIDATES}
#: names that match too much to be trusted on their own (a guess from one of them asks to be reviewed)
GENERIC_NAMES = {"ID", "AT", "TD", "MD", "AZ", "PROF", "DISTANCE"}


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
