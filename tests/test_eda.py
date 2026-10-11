"""Tests for the quick-EDA plotting functions: synthetic data only, figure structure and
the numbers behind the plots, not pixel output.
"""

import pandas as pd
import pytest

from geomodeltools import cat_vs_cat, categorical_eda, category_summary, quick_eda


def test_category_summary_matches_hand_computed_stats():
    df = pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0], "unit": ["A", "A", "B", "B"]})
    summary = category_summary(df, "value", "unit")

    assert summary.loc["A", "count"] == 2
    assert summary.loc["A", "mean"] == pytest.approx(1.5)
    assert summary.loc["A", "std"] == pytest.approx(0.5)
    assert summary.loc["A", "cov"] == pytest.approx(0.5 / 1.5)

    assert summary.loc["B", "mean"] == pytest.approx(3.5)
    assert summary.loc["B", "cov"] == pytest.approx(0.5 / 3.5)


def test_category_summary_ignores_missing_rows():
    df = pd.DataFrame({"value": [1.0, None, 3.0], "unit": ["A", "A", None]})
    summary = category_summary(df, "value", "unit")
    assert list(summary.index) == ["A"]
    assert summary.loc["A", "count"] == 1


def test_quick_eda_without_category_has_two_panels_and_rejects_non_numeric():
    df = pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0, 5.0], "unit": list("ABABA")})
    fig = quick_eda(df, "value")
    assert len(fig.axes) == 2

    with pytest.raises(ValueError):
        quick_eda(df, "unit")


def test_quick_eda_with_category_orders_panels_by_median():
    df = pd.DataFrame({"value": [1.0, 2.0, 10.0, 11.0], "unit": ["lo", "lo", "hi", "hi"]})
    fig = quick_eda(df, "value", category="unit")
    assert len(fig.axes) == 3

    violin_ax = fig.axes[1]
    labels = [t.get_text() for t in violin_ax.get_xticklabels()]
    assert labels == ["lo", "hi"]  # median(lo)=1.5 < median(hi)=10.5


def test_categorical_eda_orders_bars_by_count():
    df = pd.DataFrame({"unit": ["x", "x", "y"]})
    fig = categorical_eda(df, "unit")
    ax = fig.axes[0]
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert labels == ["x", "y"]  # x has 2 rows, y has 1


def test_cat_vs_cat_produces_one_stacked_bar_segment_per_cell():
    df = pd.DataFrame({"x": ["A", "A", "B"], "y": ["p", "q", "p"]})
    fig = cat_vs_cat(df, "x", "y")
    ax = fig.axes[0]
    # 2 distinct y-values x 2 distinct x-categories = 4 stacked bar segments
    assert len(ax.patches) == 4
