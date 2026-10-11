"""Tests for distance-to-contact and the contact profile/composition tables.

A 1D synthetic layout along x is enough to hand-verify the nearest-neighbour distances:
unit 'B' (outside) at x in {-3, -2, -1}, unit 'A' (inside) at x in {1, 2, 3}, nothing at
x=0 so the contact sits at the midpoint with no sample on it.
"""

import numpy as np
import pandas as pd
import pytest

from geomodeltools import contact_composition, contact_plot, contact_profile, distance_to_contact
from geomodeltools.contacts import plot_contact_composition, plot_contact_profile


def _line_df():
    x = [-3, -2, -1, 1, 2, 3]
    return pd.DataFrame({"x": x, "y": 0.0, "z": 0.0, "unit": ["B", "B", "B", "A", "A", "A"], "value": x})


def test_distance_to_contact_signed_nearest_neighbour():
    df = _line_df()
    dist = distance_to_contact(df, ["x"], "unit", inside="A", outside="B")
    expected = pd.Series([-4.0, -3.0, -2.0, 2.0, 3.0, 4.0], index=df.index)
    pd.testing.assert_series_equal(dist, expected, check_names=False)


def test_distance_to_contact_contact_code_shortens_distances_and_is_itself_nan():
    df = pd.concat([_line_df(), pd.DataFrame({"x": [0], "y": [0.0], "z": [0.0], "unit": ["C"], "value": [0]})], ignore_index=True)
    dist = distance_to_contact(df, ["x"], "unit", inside="A", outside="B", contact="C")
    # every inside/outside point's nearest "other side" is now the contact point at x=0
    assert dist.iloc[:6].abs().tolist() == pytest.approx([3, 2, 1, 1, 2, 3])
    assert np.isnan(dist.iloc[6])  # the contact row itself


def test_distance_to_contact_nan_when_one_side_is_empty():
    df = _line_df()
    dist = distance_to_contact(df, ["x"], "unit", inside="A", outside="nonexistent")
    assert dist.isna().all()


def test_contact_profile_bins_and_averages_correctly():
    df = _line_df()
    distance = distance_to_contact(df, ["x"], "unit", inside="A", outside="B")
    table = contact_profile(df, distance, "value", bin_width=3, n_bins=4)

    assert table["mid"].tolist() == pytest.approx([-4.5, -1.5, 1.5, 4.5])
    assert table["n"].tolist() == [2, 1, 2, 1]
    assert table["mean"].tolist() == pytest.approx([-2.5, -1.0, 1.5, 3.0])


def test_contact_composition_is_pure_on_each_side():
    df = _line_df()
    distance = distance_to_contact(df, ["x"], "unit", inside="A", outside="B")
    table = contact_composition(df, distance, "unit", bin_width=3, n_bins=4)

    assert table["A"].tolist() == pytest.approx([0.0, 0.0, 100.0, 100.0])
    assert table["B"].tolist() == pytest.approx([100.0, 100.0, 0.0, 0.0])


def test_contact_plot_requires_exactly_one_of_value_or_category():
    df = _line_df()
    with pytest.raises(ValueError):
        contact_plot(df, ["x"], "unit", "A", "B")
    with pytest.raises(ValueError):
        contact_plot(df, ["x"], "unit", "A", "B", value="value", category="unit")


def test_contact_plot_dispatches_to_profile_and_composition():
    df = _line_df()
    fig_numeric = contact_plot(df, ["x"], "unit", "A", "B", value="value", bin_width=3, n_bins=4)
    assert fig_numeric.axes[0].lines  # a profile line was drawn

    fig_cat = contact_plot(df, ["x"], "unit", "A", "B", category="unit", bin_width=3, n_bins=4)
    assert fig_cat.axes[0].patches  # stacked bars were drawn


def test_plot_functions_accept_an_existing_axes():
    import matplotlib.pyplot as plt

    df = _line_df()
    distance = distance_to_contact(df, ["x"], "unit", inside="A", outside="B")
    fig, ax = plt.subplots()
    returned = plot_contact_profile(df, distance, "value", bin_width=3, n_bins=4, ax=ax)
    assert returned is fig

    fig2, ax2 = plt.subplots()
    returned2 = plot_contact_composition(df, distance, "unit", bin_width=3, n_bins=4, ax=ax2)
    assert returned2 is fig2
