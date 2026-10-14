"""Tests for soft-boundary sample pooling: a 1D synthetic layout (same style as
test_contacts.py) so the pooled-in neighbour samples can be hand-verified.

Layout along x: unit B at x in {-3, -2, -1}, unit A at x in {1, 2, 3}, unit C at
x in {10, 11, 12} (far from both, never pooled). distance_to_contact(A, B) gives B
points distances -4, -3, -2 (see test_contacts.py), so a threshold of 2 pools only the
closest B point (x=-1, distance -2) into A.
"""

import pandas as pd
import pytest

from geomodeltools import plot_soft_boundary_boxplots, soft_boundary_pairs, soft_boundary_values


def _df():
    x = [-3, -2, -1, 1, 2, 3, 10, 11, 12]
    unit = ["B", "B", "B", "A", "A", "A", "C", "C", "C"]
    return pd.DataFrame({"x": x, "y": 0.0, "z": 0.0, "unit": unit, "value": x})


def _matrix():
    matrix = pd.DataFrame(0.0, index=["A", "B", "C"], columns=["A", "B", "C"])
    matrix.loc["B", "A"] = 2.0  # B samples within 2 m of the A/B contact pool into A
    return matrix


def test_soft_boundary_values_pools_only_neighbors_within_threshold():
    data = soft_boundary_values(_df(), ["x"], "value", "unit", _matrix())

    assert data["A"]["hard"] == [1.0, 2.0, 3.0]
    assert data["A"]["neighbors"] == {"B": [-1.0]}  # only x=-1 (distance -2) is within threshold 2
    assert sorted(data["A"]["soft"]) == [-1.0, 1.0, 2.0, 3.0]

    # B's column has no positive threshold: soft falls back to hard, no neighbors
    assert data["B"]["hard"] == [-3.0, -2.0, -1.0]
    assert data["B"]["soft"] == data["B"]["hard"]
    assert data["B"]["neighbors"] == {}

    assert data["C"]["hard"] == [10.0, 11.0, 12.0]
    assert data["C"]["soft"] == data["C"]["hard"]
    assert data["C"]["neighbors"] == {}


def test_soft_boundary_values_ignores_non_positive_thresholds():
    matrix = _matrix()
    matrix.loc["B", "A"] = 0.0  # back to no pooling
    data = soft_boundary_values(_df(), ["x"], "value", "unit", matrix)
    assert data["A"]["soft"] == data["A"]["hard"]
    assert data["A"]["neighbors"] == {}


def test_soft_boundary_pairs_lists_positive_off_diagonal_entries():
    matrix = _matrix()
    matrix.loc["A", "B"] = 3.0  # also pool A's samples into B
    matrix.loc["A", "A"] = 5.0  # diagonal: must be ignored even though positive

    pairs = soft_boundary_pairs(matrix)
    assert sorted(pairs) == sorted([("B", "A", 2.0), ("A", "B", 3.0)])


def test_plot_soft_boundary_boxplots_one_panel_per_positive_pair():
    # Funciones_RAguirre.contact_boxplots style: a literal box per distance bin, not a
    # hard/soft population comparison.
    matrix = _matrix()
    matrix.loc["A", "B"] = 3.0
    fig = plot_soft_boundary_boxplots(_df(), ["x", "y", "z"], "value", "unit", matrix, n_bins=4)

    visible = [ax for ax in fig.axes if ax.get_visible()]
    assert len(visible) == 2  # the two positive pairs: (B, A) and (A, B)

    titles = {ax.get_title() for ax in visible}
    assert titles == {"B / A", "A / B"}
    for ax in visible:
        assert len(ax.get_lines()) > 0  # boxplot artists (whiskers/medians) were actually drawn


def test_plot_soft_boundary_boxplots_rejects_matrix_with_no_positive_threshold():
    with pytest.raises(ValueError):
        plot_soft_boundary_boxplots(_df(), ["x"], "value", "unit", pd.DataFrame())

    zero_matrix = pd.DataFrame(0.0, index=["A", "B"], columns=["A", "B"])
    with pytest.raises(ValueError):
        plot_soft_boundary_boxplots(_df(), ["x"], "value", "unit", zero_matrix)
