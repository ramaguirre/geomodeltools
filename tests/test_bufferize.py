"""Tests for bufferize_2d_polygons: map_scale defaults and adaptive mode."""

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from geomodeltools import bufferize_2d_polygons

CRS = "EPSG:24879"


def _polys():
    # A wide unit (2 x 1.5 km) and a narrow one (1 km x 40 m, half-width ~19 m),
    # which the 1:100,000 buffers (first step 50 m) shrink away at once.
    return gpd.GeoDataFrame(
        {"unit": ["wide", "narrow"]},
        geometry=[box(0, 0, 2000, 1500), box(3000, 0, 4000, 40)],
        crs=CRS,
    )


def _run(gdf, **kw):
    return bufferize_2d_polygons(gdf, "unit", return_polydata=False, verbose=False, **kw)


def test_default_call_has_no_point_source_column():
    out = _run(_polys())
    assert "point_source" not in out.columns
    assert set(out["unit"]) == {"wide", "narrow"}  # default steps start at 0.1 m


def test_map_scale_equals_its_explicit_parameters():
    via_scale = _run(_polys(), map_scale=100_000)
    explicit = _run(
        _polys(),
        simplify_tolerance=25,
        segment_max_length=100,
        increments=(50, 200),
        repeats=(2, 5),
    )
    pd.testing.assert_frame_equal(
        via_scale.drop(columns="geometry").reset_index(drop=True),
        explicit.drop(columns="geometry").reset_index(drop=True),
    )


def test_explicit_value_overrides_map_scale():
    coarse = _run(_polys(), map_scale=100_000)
    fine = _run(_polys(), map_scale=100_000, segment_max_length=10)
    assert len(fine) > len(coarse)


def test_narrow_polygon_needs_adaptive_at_map_scale():
    fixed = _run(_polys(), map_scale=100_000)
    assert "narrow" not in set(fixed["unit"])

    adaptive = _run(_polys(), map_scale=100_000, adaptive=True)
    narrow = adaptive[adaptive["unit"] == "narrow"]
    assert len(narrow) > 0
    assert set(adaptive["point_source"]) <= {"ring", "adaptive_ring", "centreline", "fallback_point"}
    assert "centreline" in set(narrow["point_source"])
    # Points stay inside their polygon.
    assert narrow.geometry.within(box(3000, 0, 4000, 40).buffer(1e-6)).all()


def test_adaptive_leaves_wide_polygon_rings_alone():
    fixed = _run(_polys(), map_scale=100_000)
    adaptive = _run(_polys(), map_scale=100_000, adaptive=True)
    wide_fixed = fixed[fixed["unit"] == "wide"]
    wide_ring = adaptive[(adaptive["unit"] == "wide") & (adaptive["point_source"] == "ring")]
    assert len(wide_ring) == len(wide_fixed)


def test_every_polygon_gets_points_in_adaptive_mode():
    tiny = gpd.GeoDataFrame(
        {"unit": ["speck"]}, geometry=[box(0, 0, 30, 30)], crs=CRS
    )
    out = _run(tiny, map_scale=100_000, simplify_tolerance=0, adaptive=True)
    assert len(out) >= 1


@pytest.mark.parametrize("bad", [0, -100_000])
def test_bad_map_scale_raises(bad):
    with pytest.raises(ValueError, match="map_scale"):
        _run(_polys(), map_scale=bad)


def test_increments_without_repeats_raises():
    with pytest.raises(ValueError, match="together"):
        _run(_polys(), increments=(10, 50))
