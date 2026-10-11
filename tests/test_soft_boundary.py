"""Tests for soft-boundary sample pooling: a 1D synthetic layout (same style as
test_contacts.py) so the pooled-in neighbour samples can be hand-verified.

Layout along x: unit B at x in {-3, -2, -1}, unit A at x in {1, 2, 3}, unit C at
x in {10, 11, 12} (far from both, never pooled). distance_to_contact(A, B) gives B
points distances -4, -3, -2 (see test_contacts.py), so a threshold of 2 pools only the
closest B point (x=-1, distance -2) into A.
"""

import pandas as pd
import pytest

from geomodeltools import plot_soft_boundary_boxplots, soft_boundary_values


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


def test_plot_soft_boundary_boxplots_one_panel_per_target_with_matching_boxes():
    fig = plot_soft_boundary_boxplots(_df(), ["x"], "value", "unit", _matrix())
    visible = [ax for ax in fig.axes if ax.get_visible()]
    assert len(visible) == 3  # A, B, C

    by_title = {ax.get_title(): ax for ax in visible}
    assert set(by_title) == {"A", "B", "C"}

    a_labels = [t.get_text() for t in by_title["A"].get_xticklabels()]
    assert a_labels == ["hard", "soft", "B"]  # A has a pooled-in neighbour

    b_labels = [t.get_text() for t in by_title["B"].get_xticklabels()]
    assert b_labels == ["hard", "soft"]  # B has no neighbours: no extra box


def test_plot_soft_boundary_boxplots_rejects_empty_matrix():
    with pytest.raises(ValueError):
        plot_soft_boundary_boxplots(_df(), ["x"], "value", "unit", pd.DataFrame())
