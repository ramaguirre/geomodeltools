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


def test_ring_points_carry_nominal_distance_and_ring_index():
    out = _run(_polys(), map_scale=100_000, adaptive=True)
    rings = out[(out["unit"] == "wide") & (out["point_source"] == "ring")]
    # 1:100,000 -> buffers 50, 100, then every 200 m
    assert sorted(rings["buffer_distance"].unique())[:4] == [50.0, 100.0, 300.0, 500.0]
    first = rings[rings["ring_index"] == 1]
    assert (first["buffer_distance"] == 50.0).all()
    assert rings.groupby("ring_index")["buffer_distance"].nunique().eq(1).all()
    assert out.loc[out["point_source"] != "ring", "ring_index"].isna().all()


def test_default_call_also_has_distance_and_ring_index():
    out = _run(_polys(), map_scale=100_000)
    assert {"buffer_distance", "ring_index"} <= set(out.columns)
    assert (out["buffer_distance"] > 0).all()


def test_adaptive_ring_distance_is_its_fraction_of_half_width():
    poly = box(0, 0, 3000, 120)                       # half-width 3000*120 / 6240 ~ 57.7 m: one 50 m ring only
    gdf = gpd.GeoDataFrame({"unit": ["u"]}, geometry=[poly], crs=CRS)
    out = _run(gdf, map_scale=100_000, adaptive=True)
    ad = out[out["point_source"] == "adaptive_ring"]
    assert len(ad) > 0
    assert ad["buffer_distance"].round(6).eq(round(0.4 * poly.area / poly.length, 6)).all()


def test_centreline_distance_measured_and_points_stay_inside_unsimplified_polygon():
    # a thin wavy band: the simplification tolerance (25 m at 1:100,000) is close to its half-width,
    # so a centreline built on the simplified outline would cross the real edge
    import numpy as np
    from shapely.geometry import LineString
    xs = np.linspace(0, 4000, 400)
    band = LineString(np.column_stack([xs, 60 * np.sin(xs / 150)])).buffer(30, cap_style="flat")
    gdf = gpd.GeoDataFrame({"unit": ["band"]}, geometry=[band], crs=CRS)
    out = _run(gdf, map_scale=100_000, adaptive=True)
    pts = gpd.points_from_xy(out["x"], out["y"])
    assert band.buffer(1e-6).contains(pts).all()
    cl = out[out["point_source"] == "centreline"]
    assert len(cl) > 0
    measured = band.boundary.distance(gpd.points_from_xy(cl["x"], cl["y"]))
    assert np.allclose(cl["buffer_distance"], measured)
