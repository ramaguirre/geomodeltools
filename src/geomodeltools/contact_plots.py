"""Distance-to-contact calculations and contact plots between two geological units.

A "contact plot" shows how a variable changes as samples approach, cross, and move away
from the boundary between two units (e.g. grade dying off into waste, or lithology mix
changing across an intrusive margin). `contact_plot` is the one-call entry point; it works
for both a numeric variable (a mean/median profile) and a categorical one (a composition
breakdown), which is the split the rest of this module is built around.

This replaces `Funciones_RAguirre.Funciones_RAguirre.distance_function` (which stays as-is,
for reference) with a `scipy.spatial.cKDTree` nearest-neighbour query: one vectorised query
per side instead of a hand-rolled batched `cdist` loop, and no more growing an empty-string
column to hold floats.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy.spatial import cKDTree

from ._stats_util import as_list, nice_step, weights_or_ones


def distance_to_contact(
    df: pd.DataFrame,
    coords: list[str],
    unit: str,
    inside,
    outside,
    contact=None,
) -> pd.Series:
    """Signed distance from each sample to the nearest point on the other side of a contact.

    Rows whose `unit` value is in `inside` get the distance to the nearest `outside` (or
    `contact`) point, as a positive number. Rows in `outside` get the distance to the
    nearest `inside` (or `contact`) point, as a negative number. Rows in `contact`, or in
    neither list, get NaN. If `inside` (or `outside`) has no rows at all, the other side's
    distances are all NaN too, since there is nothing to measure them against.

    Parameters:
        df: input table.
        coords: coordinate column names, e.g. ``["x", "y", "z"]``.
        unit: column holding the geological unit code for each row.
        inside: unit code, or list of codes, on the inside of the contact.
        outside: unit code, or list of codes, on the outside of the contact.
        contact: optional unit code(s) that sit exactly on the contact (distance 0 for
            the other side's nearest-point search; the contact rows themselves get NaN).

    Returns:
        A float Series aligned to `df.index`.
    """
    inside = as_list(inside)
    outside = as_list(outside)
    contact = as_list(contact) if contact is not None else []

    xyz = df[list(coords)].to_numpy(dtype=float)
    unit_values = df[unit]

    is_inside = unit_values.isin(inside).to_numpy()
    is_outside = unit_values.isin(outside).to_numpy()
    is_contact = unit_values.isin(contact).to_numpy()

    inside_pts = xyz[is_inside]
    outside_pts = xyz[is_outside]
    contact_pts = xyz[is_contact]

    distance = pd.Series(np.nan, index=df.index, dtype=float)

    outside_targets = np.concatenate([a for a in (inside_pts, contact_pts) if len(a)]) if (len(inside_pts) or len(contact_pts)) else None
    if len(outside_pts) and outside_targets is not None and len(outside_targets):
        dist, _ = cKDTree(outside_targets).query(outside_pts)
        distance.loc[df.index[is_outside]] = -dist

    inside_targets = np.concatenate([a for a in (outside_pts, contact_pts) if len(a)]) if (len(outside_pts) or len(contact_pts)) else None
    if len(inside_pts) and inside_targets is not None and len(inside_targets):
        dist, _ = cKDTree(inside_targets).query(inside_pts)
        distance.loc[df.index[is_inside]] = dist

    return distance


def _bin_edges(distance: pd.Series, bin_width: float | None, n_bins: int) -> np.ndarray:
    if bin_width is None:
        span = 2 * np.nanpercentile(np.abs(distance), 90)
        bin_width = nice_step(span / n_bins) if span > 0 else 1.0
    half = n_bins / 2
    return bin_width * (np.arange(n_bins + 1) - half)


def _distance_series(df: pd.DataFrame, distance) -> pd.Series:
    if isinstance(distance, str):
        return df[distance]
    return pd.Series(distance, index=df.index)


def contact_profile(
    df: pd.DataFrame,
    distance,
    value: str,
    bin_width: float | None = None,
    n_bins: int = 10,
    weight: str | None = None,
    agg: str = "mean",
) -> pd.DataFrame:
    """Bin `value` by distance-to-contact and aggregate each bin.

    Bins are centred on the contact (distance 0), `n_bins` wide in total, each `bin_width`
    across (auto-picked from the data if not given).

    Parameters:
        df: input table.
        distance: a column name in `df`, or an array/Series of the same length, typically
            from `distance_to_contact`.
        value: numeric column to aggregate.
        bin_width: width of each distance bin. Auto-picked to span the central 90% of
            `|distance|` if None.
        n_bins: number of bins, split evenly either side of the contact.
        weight: optional column of per-row weights; only used for `agg="mean"`.
        agg: ``"mean"`` (weighted, if `weight` is given) or ``"median"`` (unweighted).

    Returns:
        One row per bin, with columns `mid` (bin midpoint), `agg`, and `n` (sample count).
    """
    if agg not in ("mean", "median"):
        raise ValueError(f"agg must be 'mean' or 'median', got {agg!r}")

    d = _distance_series(df, distance)
    v = df[value]
    w = weights_or_ones(df, weight)

    valid = d.notna() & v.notna()
    d, v, w = d[valid], v[valid], w[valid]
    if d.empty:
        raise ValueError("No rows with both a distance and a value to profile.")

    bins = pd.cut(d, _bin_edges(d, bin_width, n_bins))
    work = pd.DataFrame({"value": v.to_numpy(), "weight": w.to_numpy()}, index=d.index)
    grouped = work.groupby(bins, observed=False)

    n = grouped.size()
    if agg == "median":
        stat = grouped["value"].median()
    else:
        stat = grouped.apply(lambda g: np.average(g["value"], weights=g["weight"]) if len(g) else np.nan)

    mids = n.index.map(lambda interval: interval.mid).to_numpy(dtype=float)
    return pd.DataFrame({"mid": mids, agg: stat.to_numpy(dtype=float), "n": n.to_numpy()})


def contact_boxplot_bins(
    df: pd.DataFrame,
    distance,
    value: str,
    bin_width: float | None = None,
    n_bins: int = 10,
) -> dict:
    """Raw `value` samples grouped by distance-to-contact bin, for a literal box-per-bin
    contact diagram (`Funciones_RAguirre.contact_boxplots`' style) rather than
    `contact_profile`'s single mean/median line.

    Same binning as `contact_profile`; see its docstring for `bin_width`/`n_bins`.

    Returns:
        `{"mid": [...], "values": [[...], ...]}`: one inner list of raw values per bin, in
        `mid` order (empty bins included as empty lists).
    """
    d = _distance_series(df, distance)
    v = df[value]
    valid = d.notna() & v.notna()
    d, v = d[valid], v[valid]
    if d.empty:
        raise ValueError("No rows with both a distance and a value to bin.")

    bins = pd.cut(d, _bin_edges(d, bin_width, n_bins))
    grouped = v.groupby(bins, observed=False)
    values = []
    for interval in bins.cat.categories:
        try:
            values.append(grouped.get_group(interval).tolist())
        except KeyError:
            values.append([])  # no samples in this bin
    return {"mid": bins.cat.categories.mid.tolist(), "values": values}


def plot_contact_boxplot(
    df: pd.DataFrame,
    distance,
    value: str,
    bin_width: float | None = None,
    n_bins: int = 10,
    ax=None,
) -> Figure:
    """Box-and-whisker plot of `value` per distance-to-contact bin: a literal box per bin
    (`Funciones_RAguirre.contact_boxplots`' style), as opposed to `plot_contact_profile`'s
    single mean/median line.
    """
    data = contact_boxplot_bins(df, distance, value, bin_width=bin_width, n_bins=n_bins)
    mids = [mid for mid, vals in zip(data["mid"], data["values"]) if vals]
    values = [vals for vals in data["values"] if vals]
    if not values:
        raise ValueError("Every distance bin is empty.")

    fig, ax = _ensure_axes(ax)
    width = float(np.diff(data["mid"]).mean()) if len(data["mid"]) > 1 else 1.0
    ax.boxplot(values, positions=mids, widths=width * 0.7, showmeans=True)
    ax.axvline(0, color="k", ls=":", lw=1.5)
    ax.set_xlabel("Distance to contact")
    ax.set_ylabel(value)
    ax.set_title(f"{value}: contact box plot (n={sum(len(v) for v in values)})")
    fig.tight_layout()
    return fig


def contact_composition(
    df: pd.DataFrame,
    distance,
    category: str,
    bin_width: float | None = None,
    n_bins: int = 10,
    weight: str | None = None,
) -> pd.DataFrame:
    """Bin a categorical column by distance-to-contact and compute each bin's composition.

    Same binning as `contact_profile`; see its docstring for `bin_width`/`n_bins`.

    Returns:
        One row per bin, with columns `mid`, `n`, and one column per category holding its
        percentage of that bin's (weighted) total.
    """
    d = _distance_series(df, distance)
    c = df[category]
    w = weights_or_ones(df, weight)

    valid = d.notna() & c.notna()
    d, c, w = d[valid], c[valid], w[valid]
    if d.empty:
        raise ValueError("No rows with both a distance and a category to profile.")

    bins = pd.cut(d, _bin_edges(d, bin_width, n_bins))
    table = pd.crosstab(bins, c, values=w.to_numpy(), aggfunc="sum", dropna=False).fillna(0.0)

    n = table.sum(axis=1)
    pct = table.div(n.replace(0, np.nan), axis=0) * 100
    mids = pct.index.map(lambda interval: interval.mid).to_numpy(dtype=float)
    pct.insert(0, "mid", mids)
    pct.insert(1, "n", n.to_numpy())
    return pct.reset_index(drop=True)


def _ensure_axes(ax):
    if ax is None:
        fig, ax = _new_figure()
        return fig, ax
    return ax.figure, ax


def _new_figure():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    return fig, ax


def plot_contact_profile(
    df: pd.DataFrame,
    distance,
    value: str,
    bin_width: float | None = None,
    n_bins: int = 10,
    weight: str | None = None,
    agg: str = "mean",
    ax=None,
) -> Figure:
    """Plot the `contact_profile` of `value`: its binned statistic vs. distance-to-contact."""
    table = contact_profile(df, distance, value, bin_width=bin_width, n_bins=n_bins, weight=weight, agg=agg)
    fig, ax = _ensure_axes(ax)
    ax.plot(table["mid"], table[agg], marker="o", color="steelblue")
    ax.axvline(0, color="firebrick", ls=":", lw=1.5)
    ax.set_xlabel("Distance to contact")
    ax.set_ylabel(f"{agg} {value}")
    ax.set_title(f"{value}: contact profile (n={int(table['n'].sum())})")
    fig.tight_layout()
    return fig


def plot_contact_composition(
    df: pd.DataFrame,
    distance,
    category: str,
    bin_width: float | None = None,
    n_bins: int = 10,
    weight: str | None = None,
    ax=None,
) -> Figure:
    """Plot the `contact_composition` of `category`: stacked percentage bars vs. distance-to-contact."""
    table = contact_composition(df, distance, category, bin_width=bin_width, n_bins=n_bins, weight=weight)
    categories = [c for c in table.columns if c not in ("mid", "n")]
    fig, ax = _ensure_axes(ax)
    width = float(np.diff(table["mid"]).mean()) if len(table) > 1 else 1.0
    bottom = np.zeros(len(table))
    for cat in categories:
        values = table[cat].fillna(0).to_numpy()
        ax.bar(table["mid"], values, bottom=bottom, width=width * 0.95, label=str(cat))
        bottom += values
    ax.axvline(0, color="k", ls=":", lw=1.5)
    ax.set_xlabel("Distance to contact")
    ax.set_ylabel(f"{category} composition [%]")
    ax.set_title(f"{category}: contact composition (n={int(table['n'].sum())})")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout()
    return fig


def contact_plot(
    df: pd.DataFrame,
    coords: list[str],
    unit: str,
    inside,
    outside,
    *,
    value: str | None = None,
    category: str | None = None,
    kind: str = "profile",
    contact=None,
    bin_width: float | None = None,
    n_bins: int = 10,
    weight: str | None = None,
    agg: str = "mean",
    ax=None,
) -> Figure:
    """Contact plot: how a numeric or categorical variable changes across a unit boundary.

    One call that does `distance_to_contact` then one of: `plot_contact_profile` (pass
    `value`, `kind="profile"` (default): a mean/median line), `plot_contact_boxplot`
    (pass `value`, `kind="boxplot"`: a literal box per distance bin), or
    `plot_contact_composition` (pass `category`: a stacked percentage breakdown). Exactly
    one of `value`/`category` must be given; `kind` only applies to `value`.

    `coords`, `unit`, `inside`, `outside`, `contact` are passed straight through to
    `distance_to_contact`; `bin_width`, `n_bins`, `weight` (and `agg`, for `kind="profile"`)
    to the profile/boxplot/composition function.
    """
    if (value is None) == (category is None):
        raise ValueError("Pass exactly one of `value` (numeric) or `category` (categorical).")
    if kind not in ("profile", "boxplot"):
        raise ValueError(f"kind must be 'profile' or 'boxplot', got {kind!r}")

    distance = distance_to_contact(df, coords, unit, inside, outside, contact=contact)

    if value is not None:
        if kind == "boxplot":
            return plot_contact_boxplot(df, distance, value, bin_width=bin_width, n_bins=n_bins, ax=ax)
        return plot_contact_profile(df, distance, value, bin_width=bin_width, n_bins=n_bins, weight=weight, agg=agg, ax=ax)
    return plot_contact_composition(df, distance, category, bin_width=bin_width, n_bins=n_bins, weight=weight, ax=ax)
