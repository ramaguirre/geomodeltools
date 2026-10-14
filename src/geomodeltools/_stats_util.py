"""Small helpers shared by eda.py and contact_plots.py. Private: not part of the public API."""

from __future__ import annotations

import numpy as np
import pandas as pd


def weights_or_ones(df: pd.DataFrame, weight: str | None) -> pd.Series:
    """Return `df[weight]` as float, or a column of ones the same length as `df` if `weight` is None."""
    if weight is None:
        return pd.Series(1.0, index=df.index)
    return df[weight].astype(float)


def nice_step(raw: float) -> float:
    """Round `raw` up to a "nice" axis-tick-like step: 1, 2, or 5 times a power of ten."""
    if not np.isfinite(raw) or raw <= 0:
        return 1.0
    exponent = np.floor(np.log10(raw))
    base = raw / 10**exponent
    if base <= 1:
        nice = 1
    elif base <= 2:
        nice = 2
    elif base <= 5:
        nice = 5
    else:
        nice = 10
    return float(nice * 10**exponent)


def as_list(value) -> list:
    """Normalise a single code or a collection of codes into a plain list."""
    if isinstance(value, (list, tuple, set, np.ndarray, pd.Index)):
        return list(value)
    return [value]
