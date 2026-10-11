"""Quick exploratory-data-analysis plots for numeric and categorical unit attributes.

A cleaned-up, non-interactive rewrite of the EDA plots in
`Funciones_RAguirre.Plotting_Functions_RAguirre` (`quick_eda`, `violin_by_cat`, `var_coeff`,
`bar_cat_vs_cat`...), which stays as-is for reference. The main differences: no
seaborn/ipywidgets/Jupyter dependency, weights are optional everywhere instead of being
silently dropped, and every function returns its `Figure` rather than saving a PNG to the
current directory as a side effect.

Each plotting function (`quick_eda`, `categorical_eda`, `cat_vs_cat`) is a thin matplotlib
renderer over a `*_data()` counterpart that does the actual grouping/aggregation and returns
plain, JSON-ready Python types. `report.py` calls the same `*_data()` functions to build the
interactive HTML report, so the numbers behind a static figure and the HTML report never
drift apart from having two separate implementations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from pandas.api.types import is_numeric_dtype
from scipy import stats

from ._stats_util import weights_or_ones


def _probability_scores(values: np.ndarray, weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sorted values and their normal-score (z-score) plotting positions, Hazen convention."""
    order = np.argsort(values)
    v_sorted = values[order]
    w_sorted = weight[order]
    cum = np.cumsum(w_sorted) - 0.5 * w_sorted
    p = cum / w_sorted.sum()
    return v_sorted, stats.norm.ppf(p)


def category_summary(df: pd.DataFrame, value: str, category: str, weight: str | None = None) -> pd.DataFrame:
    """Per-category count, mean, median, std and coefficient of variation (std/mean) of `value`."""
    w = weights_or_ones(df, weight)
    work = pd.DataFrame({"value": df[value], "weight": w, "category": df[category]}).dropna(subset=["value", "category"])

    rows = {}
    for cat, g in work.groupby("category"):
        mean = float(np.average(g["value"], weights=g["weight"]))
        variance = float(np.average((g["value"] - mean) ** 2, weights=g["weight"]))
        std = variance**0.5
        rows[cat] = {
            "count": len(g),
            "mean": mean,
            "median": float(g["value"].median()),
            "std": std,
            "cov": std / mean if mean else np.nan,
        }
    return pd.DataFrame.from_dict(rows, orient="index")[["count", "mean", "median", "std", "cov"]]


def quick_eda_data(df: pd.DataFrame, value: str, category: str | None = None, weight: str | None = None) -> dict:
    """The numbers behind `quick_eda`, as plain JSON-ready Python types.

    Without `category`: `{"mode": "overall", "values": [...], "weights": [...],
    "probability": {"v": [...], "z": [...]}, "summary": {count, mean, median, std, cov}}`.

    With `category`: `{"mode": "by_category", "order": [...cats sorted by median...],
    "series": {cat: {"values": [...], "probability": {"v": [...], "z": [...]}}},
    "summary": {cat: {count, mean, median, std, cov}}}`.
    """
    if not is_numeric_dtype(df[value]):
        raise ValueError(f"quick_eda needs a numeric `value` column; {value!r} is {df[value].dtype}.")

    w = weights_or_ones(df, weight)
    valid = df[value].notna()
    data = df.loc[valid, value]
    weights = w.loc[valid]

    if category is None:
        v_sorted, z = _probability_scores(data.to_numpy(dtype=float), weights.to_numpy(dtype=float))
        summary = category_summary(df.loc[valid].assign(_all="all"), value, "_all", weight=weight).iloc[0]
        return {
            "mode": "overall",
            "value": value,
            "values": data.tolist(),
            "weights": weights.tolist(),
            "probability": {"v": v_sorted.tolist(), "z": z.tolist()},
            "summary": summary.to_dict(),
        }

    cats = df.loc[valid, category]
    summary = category_summary(df.loc[valid], value, category, weight=weight)
    order_raw = summary["median"].sort_values().index.tolist()
    if not order_raw:
        raise ValueError(f"No rows with both `{value}` and `{category}` set.")
    order = [str(c) for c in order_raw]

    series = {}
    for raw_cat, cat in zip(order_raw, order):
        mask = (cats == raw_cat).to_numpy()
        v_sorted, z = _probability_scores(data.to_numpy(dtype=float)[mask], weights.to_numpy(dtype=float)[mask])
        series[cat] = {"values": data.to_numpy()[mask].tolist(), "probability": {"v": v_sorted.tolist(), "z": z.tolist()}}

    return {
        "mode": "by_category",
        "value": value,
        "category": category,
        "order": order,
        "series": series,
        "summary": {cat: summary.loc[raw_cat].to_dict() for raw_cat, cat in zip(order_raw, order)},
    }


def quick_eda(
    df: pd.DataFrame,
    value: str,
    category: str | None = None,
    weight: str | None = None,
    log: bool = False,
    bins: int = 40,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Quick look at one numeric column: distribution, and variability by category if given.

    Without `category`: a histogram and a normal-probability plot of `value`, annotated
    with its weighted count/mean/median/std/CoV.

    With `category`: the probability plot splits into one curve per category (ordered by
    median), next to a violin plot and a coefficient-of-variation bar chart by category, so
    you can see at a glance which units are tight/variable or stand out from the rest.

    Parameters:
        df: input table.
        value: numeric column to plot. Must be numeric.
        category: optional column to split/compare by.
        weight: optional column of per-row weights.
        log: log10-scale the value axis (histogram x, probability-plot x, violin y).
        bins: histogram bin count (only used without `category`).
        figsize: passed to `plt.subplots`; a sensible default is picked otherwise.
    """
    import matplotlib.pyplot as plt

    data = quick_eda_data(df, value, category=category, weight=weight)

    if data["mode"] == "overall":
        fig, (ax_hist, ax_prob) = plt.subplots(1, 2, figsize=figsize or (10, 4))
        values = np.asarray(data["values"])
        weights = np.asarray(data["weights"])
        plot_values = np.log10(values) if log else values
        ax_hist.hist(plot_values, bins=bins, weights=weights, color="steelblue", edgecolor="white")
        if log:
            ax_hist.xaxis.set_major_formatter(FuncFormatter(lambda x, _pos: f"{10**x:g}"))
        ax_hist.set_xlabel(value)
        ax_hist.set_ylabel("Frequency")

        v = np.asarray(data["probability"]["v"])
        z = np.asarray(data["probability"]["z"])
        ax_prob.scatter(v, z, s=8, alpha=0.6, color="steelblue")
        if log:
            ax_prob.set_xscale("log")
        ax_prob.yaxis.set_major_formatter(FuncFormatter(lambda z, _pos: f"{stats.norm.cdf(z):.3f}"))
        ax_prob.set_xlabel(value)
        ax_prob.set_ylabel("Cumulative probability")

        text = "\n".join(f"{k}: {v:,.3g}" for k, v in data["summary"].items())
        ax_prob.text(
            0.02, 0.98, text, transform=ax_prob.transAxes, va="top", fontsize=8,
            bbox={"boxstyle": "round", "fc": "white", "alpha": 0.8},
        )
        fig.suptitle(value)
        fig.tight_layout()
        return fig

    ordered = data["order"]
    fig, (ax_prob, ax_violin, ax_cov) = plt.subplots(1, 3, figsize=figsize or (15, 4.5))

    for cat in ordered:
        v = np.asarray(data["series"][cat]["probability"]["v"])
        z = np.asarray(data["series"][cat]["probability"]["z"])
        ax_prob.plot(v, z, marker=".", ms=3, lw=0, alpha=0.7, label=cat)
    if log:
        ax_prob.set_xscale("log")
    ax_prob.yaxis.set_major_formatter(FuncFormatter(lambda z, _pos: f"{stats.norm.cdf(z):.3f}"))
    ax_prob.set_xlabel(value)
    ax_prob.set_ylabel("Cumulative probability")
    ax_prob.legend(fontsize=7, loc="lower right")

    series_by_cat = [np.asarray(data["series"][cat]["values"]) for cat in ordered]
    if log:
        series_by_cat = [np.log10(s) for s in series_by_cat]
    ax_violin.violinplot(series_by_cat, showmeans=True, showextrema=False)
    ax_violin.set_xticks(range(1, len(ordered) + 1))
    ax_violin.set_xticklabels(ordered, rotation=45, ha="right")
    if log:
        ax_violin.yaxis.set_major_formatter(FuncFormatter(lambda x, _pos: f"{10**x:g}"))
    ax_violin.set_ylabel(value)
    ax_violin.set_title(f"{value} by {category}")

    cov = [data["summary"][cat]["cov"] for cat in ordered]
    ax_cov.bar(range(len(ordered)), cov, color="steelblue")
    ax_cov.set_xticks(range(len(ordered)))
    ax_cov.set_xticklabels(ordered, rotation=45, ha="right")
    ax_cov.set_ylabel("Coefficient of variation (std / mean)")
    ax_cov.set_title("Variability by category")

    fig.suptitle(f"{value} — quick EDA by {category}")
    fig.tight_layout()
    return fig


def categorical_eda_data(df: pd.DataFrame, category: str, weight: str | None = None, top_n: int | None = None) -> dict:
    """The numbers behind `categorical_eda`: `{"labels": [...], "counts": [...], "pct": [...]}`."""
    w = weights_or_ones(df, weight)
    valid = df[category].notna()
    counts = (
        pd.DataFrame({"category": df.loc[valid, category], "weight": w.loc[valid]})
        .groupby("category")["weight"]
        .sum()
        .sort_values(ascending=False)
    )
    if top_n:
        counts = counts.iloc[:top_n]
    pct = counts / counts.sum() * 100
    return {
        "category": category,
        "labels": [str(c) for c in counts.index],
        "counts": counts.tolist(),
        "pct": pct.tolist(),
        "weighted": weight is not None,
    }


def categorical_eda(
    df: pd.DataFrame,
    category: str,
    weight: str | None = None,
    top_n: int | None = None,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Frequency bar chart for a single categorical column: (weighted) count and % of total."""
    import matplotlib.pyplot as plt

    data = categorical_eda_data(df, category, weight=weight, top_n=top_n)

    fig, ax = plt.subplots(figsize=figsize or (max(4.0, 0.5 * len(data["labels"])), 4))
    bars = ax.bar(data["labels"], data["counts"], color="steelblue")
    for bar, p in zip(bars, data["pct"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{p:.1f}%", ha="center", va="bottom", fontsize=8)
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha("right")
    ax.set_ylabel("Weight" if data["weighted"] else "Count")
    ax.set_title(category)
    fig.tight_layout()
    return fig


def cat_vs_cat_data(
    df: pd.DataFrame, category_x: str, category_y: str, weight: str | None = None, normalize: bool = True
) -> dict:
    """The numbers behind `cat_vs_cat`: a matrix with `x_labels` rows and `y_labels` columns."""
    w = weights_or_ones(df, weight)
    valid = df[category_x].notna() & df[category_y].notna()
    work = pd.DataFrame({"x": df.loc[valid, category_x], "y": df.loc[valid, category_y], "w": w.loc[valid]})
    pivot = work.pivot_table(values="w", index="x", columns="y", aggfunc="sum", fill_value=0.0)
    if normalize:
        pivot = pivot.div(pivot.sum(axis=1), axis=0) * 100
    return {
        "category_x": category_x,
        "category_y": category_y,
        "x_labels": [str(i) for i in pivot.index],
        "y_labels": [str(c) for c in pivot.columns],
        "matrix": pivot.to_numpy().tolist(),  # rows = x_labels, columns = y_labels
        "normalize": normalize,
        "weighted": weight is not None,
    }


def cat_vs_cat(
    df: pd.DataFrame,
    category_x: str,
    category_y: str,
    weight: str | None = None,
    normalize: bool = True,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Stacked bar chart: `category_y` composition within each `category_x` value."""
    import matplotlib.pyplot as plt

    data = cat_vs_cat_data(df, category_x, category_y, weight=weight, normalize=normalize)
    matrix = np.asarray(data["matrix"])

    fig, ax = plt.subplots(figsize=figsize or (max(4.0, 0.5 * len(data["x_labels"])), 4.5))
    bottom = np.zeros(len(data["x_labels"]))
    for j, y_label in enumerate(data["y_labels"]):
        values = matrix[:, j]
        ax.bar(data["x_labels"], values, bottom=bottom, label=y_label)
        bottom += values
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha("right")
    unit = "%" if normalize else ("weight" if data["weighted"] else "count")
    ax.set_ylabel(f"{category_y} [{unit}]")
    ax.set_title(f"{category_y} composition by {category_x}")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout()
    return fig
