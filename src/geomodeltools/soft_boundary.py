"""Soft-boundary (gradational contact) sample pooling and QC box plots.

A rewrite of `Funciones_RAguirre.Geostats_RA.soft_boundary_samples`, which stays as-is for
reference. That function pools, per unit, its own samples plus any neighbour's samples
that fall within a per-pair distance threshold of the contact (a "soft"/gradational
boundary, as opposed to a hard one) into one merged column per unit — useful when a
contact is known to be gradational and you want the neighbouring domain's near-boundary
samples folded into statistics for the unit they are geologically part of.

`soft_boundary_values` reuses `contacts.distance_to_contact` for the actual distance
calculation (no second implementation of it) and, unlike the original, keeps the "hard"
(strict within-unit) and "soft" (boundary-pooled) populations separate, along with each
contributing neighbour's own pooled-in slice, so they can be compared. No box-plot
companion to `soft_boundary_samples` exists in the reference code, so the panel layout in
`plot_soft_boundary_boxplots` (one subplot per target unit, in the soft-boundary matrix's
own column order) is a judgment call, not a port.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from .contacts import distance_to_contact


def soft_boundary_values(
    df: pd.DataFrame,
    coords: list[str],
    value: str,
    unit: str,
    soft_boundary_matrix: pd.DataFrame,
) -> dict:
    """The hard/soft populations behind the soft-boundary box plots, as JSON-ready data.

    `soft_boundary_matrix.loc[neighbor, target]` is the distance threshold (same units as
    `coords`) within which `neighbor`'s samples are pooled into `target`'s "soft"
    population — the same row=neighbor/column=target convention as
    `Geostats_RA.soft_boundary_samples`. Thresholds that are missing or <= 0 are ignored.

    Parameters:
        df: input table.
        coords: coordinate column names, e.g. ``["x", "y", "z"]``.
        value: numeric column to collect per population.
        unit: column holding the geological unit code for each row.
        soft_boundary_matrix: a unit x unit DataFrame of distance thresholds (see above).

    Returns:
        `{target: {"hard": [...], "soft": [...], "neighbors": {neighbor: [...]}}}`, one
        entry per column of `soft_boundary_matrix`. "soft" is "hard" plus every
        neighbour's pooled-in slice combined (matching `soft_boundary_samples`'s single
        merged column); "neighbors" keeps each neighbour's own slice separate.
    """
    result = {}
    for target in soft_boundary_matrix.columns:
        thresholds = soft_boundary_matrix[target]
        thresholds = thresholds[thresholds > 0]

        hard = df.loc[df[unit] == target, value].dropna().tolist()
        soft = list(hard)
        neighbors = {}
        for neighbor, threshold in thresholds.items():
            if neighbor == target:
                continue
            distance = distance_to_contact(df, coords, unit, inside=target, outside=neighbor)
            mask = (df[unit] == neighbor) & distance.notna() & (distance.abs() <= threshold)
            pooled = df.loc[mask, value].dropna().tolist()
            neighbors[str(neighbor)] = pooled
            soft.extend(pooled)

        result[str(target)] = {"hard": hard, "soft": soft, "neighbors": neighbors}
    return result


def plot_soft_boundary_boxplots(
    df: pd.DataFrame,
    coords: list[str],
    value: str,
    unit: str,
    soft_boundary_matrix: pd.DataFrame,
    log: bool = False,
    ncols: int | None = None,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Matrix of box plots: per target unit, "hard" vs. "soft" (boundary-pooled) `value`.

    One panel per column of `soft_boundary_matrix` (its own order), each a box plot of
    that unit's `hard` and `soft` populations (see `soft_boundary_values`) plus one box
    per contributing neighbour's own pooled-in slice, so you can see both the overall
    effect of pooling and which neighbour is driving it.

    Parameters:
        log: log10-scale the value axis in every panel.
        ncols: panels per row. Defaults to `ceil(sqrt(n_targets))`.
        figsize: passed to `plt.subplots`; a sensible default is picked otherwise.
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    data = soft_boundary_values(df, coords, value, unit, soft_boundary_matrix)
    targets = list(data.keys())
    if not targets:
        raise ValueError("`soft_boundary_matrix` has no columns.")

    ncols = ncols or int(np.ceil(np.sqrt(len(targets))))
    nrows = int(np.ceil(len(targets) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize or (4.5 * ncols, 4 * nrows), squeeze=False)

    for ax, target in zip(axes.flat, targets):
        record = data[target]
        labels = ["hard", "soft"] + list(record["neighbors"])
        series = [record["hard"], record["soft"]] + list(record["neighbors"].values())
        labels = [label for label, s in zip(labels, series) if len(s)]
        series = [s for s in series if len(s)]
        if log:
            series = [np.log10(s) for s in series]
        ax.boxplot(series, showmeans=True)  # positions boxes at 1..len(series) regardless of matplotlib version
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels)
        if log:
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _pos: f"{10**x:g}"))
        ax.set_title(target)
        ax.set_ylabel(value)
        for label in ax.get_xticklabels():
            label.set_rotation(30)
            label.set_ha("right")

    for ax in axes.flat[len(targets) :]:
        ax.set_visible(False)

    fig.suptitle(f"{value}: soft-boundary pooling by {unit}")
    fig.tight_layout()
    return fig
