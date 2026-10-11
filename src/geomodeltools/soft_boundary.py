"""Soft-boundary (gradational contact) sample pooling and a matrix of contact box plots.

`soft_boundary_values` is a rewrite of `Funciones_RAguirre.Geostats_RA.soft_boundary_samples`,
which stays as-is for reference. That function pools, per unit, its own samples plus any
neighbour's samples that fall within a per-pair distance threshold of the contact (a
"soft"/gradational boundary, as opposed to a hard one) into one merged column per unit.
This keeps the "hard" (strict within-unit) and "soft" (boundary-pooled) populations
separate, along with each contributing neighbour's own pooled-in slice, and reuses
`contacts.distance_to_contact` rather than a second implementation of the distance
calculation.

`plot_soft_boundary_boxplots` is the "matrix of box plots": one panel per positive-threshold
(neighbor, target) pair in the same matrix, each a literal box-per-distance-bin contact
diagram (`Funciones_RAguirre.contact_boxplots`'s style, via `contacts.plot_contact_boxplot`)
out to that pair's own threshold distance. An earlier version of this panel compared the
hard/soft/neighbour *populations* as three plain boxes per unit; `contact_boxplots` is the
box-plot style actually referenced for "matrix of box plots", so this now matches it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from .contacts import distance_to_contact, plot_contact_boxplot


def soft_boundary_values(
    df: pd.DataFrame,
    coords: list[str],
    value: str,
    unit: str,
    soft_boundary_matrix: pd.DataFrame,
) -> dict:
    """The hard/soft pooled populations, as JSON-ready data (`soft_boundary_samples`'s
    merged-column result, split apart).

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


def soft_boundary_pairs(soft_boundary_matrix: pd.DataFrame) -> list[tuple[str, str, float]]:
    """`(neighbor, target, threshold)` for every positive, off-diagonal entry of
    `soft_boundary_matrix` (row=neighbor, column=target — see `soft_boundary_values`)."""
    pairs = []
    for target in soft_boundary_matrix.columns:
        thresholds = soft_boundary_matrix[target]
        for neighbor, threshold in thresholds.items():
            if neighbor != target and pd.notna(threshold) and threshold > 0:
                pairs.append((neighbor, target, float(threshold)))
    return pairs


def plot_soft_boundary_boxplots(
    df: pd.DataFrame,
    coords: list[str],
    value: str,
    unit: str,
    soft_boundary_matrix: pd.DataFrame,
    n_bins: int = 10,
    ncols: int | None = None,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Matrix of contact box plots: one panel per positive-threshold (neighbor, target)
    pair in `soft_boundary_matrix`.

    Each panel is a literal box-and-whisker-per-distance-bin diagram of `value` across
    that pair's contact (`contacts.plot_contact_boxplot`, i.e. `Funciones_RAguirre.
    contact_boxplots`'s style), spanning the pair's own soft-boundary distance threshold
    (`bin_width = 2 * threshold / n_bins`, so the bins cover exactly +/- threshold).

    Parameters:
        n_bins: bins per panel, split evenly either side of the contact.
        ncols: panels per row. Defaults to `ceil(sqrt(n_pairs))`.
        figsize: passed to `plt.subplots`; a sensible default is picked otherwise.
    """
    import matplotlib.pyplot as plt

    pairs = soft_boundary_pairs(soft_boundary_matrix)
    if not pairs:
        raise ValueError("No positive threshold in `soft_boundary_matrix`.")

    ncols = ncols or int(np.ceil(np.sqrt(len(pairs))))
    nrows = int(np.ceil(len(pairs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize or (4.5 * ncols, 4 * nrows), squeeze=False)

    for ax, (neighbor, target, threshold) in zip(axes.flat, pairs):
        distance = distance_to_contact(df, coords, unit, inside=target, outside=neighbor)
        plot_contact_boxplot(df, distance, value, bin_width=2 * threshold / n_bins, n_bins=n_bins, ax=ax)
        ax.set_title(f"{neighbor} / {target}")

    for ax in axes.flat[len(pairs) :]:
        ax.set_visible(False)

    fig.suptitle(f"{value}: soft-boundary contact box plots by {unit}")
    fig.tight_layout()
    return fig
