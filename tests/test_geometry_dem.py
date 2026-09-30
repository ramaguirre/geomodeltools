"""Tests for densify_geometries, geometries_to_points and add_z_from_opentopography.

The DEM tests use a small synthetic GeoTIFF: an existing `out_tiff_path` is reused as-is,
so nothing is downloaded and no API key is needed.
"""

import numpy as np
import geopandas as gpd
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString, MultiLineString, Point, Polygon, box

from geomodeltools import add_z_from_opentopography, densify_geometries, geometries_to_points

CRS = "EPSG:24879"
X0, Y0, CELL, N = 400_000.0, 7_500_000.0, 10.0, 60  # DEM covers X0..X0+600, Y0-600..Y0


def _plane(x, y):
    return 1000.0 + 0.01 * (x - X0) + 0.02 * (Y0 - y)


@pytest.fixture
def dem(tmp_path):
    cols = X0 + CELL * (np.arange(N) + 0.5)
    rows = Y0 - CELL * (np.arange(N) + 0.5)
    z = _plane(cols[None, :], rows[:, None]).astype("float32")
    path = tmp_path / "dem.tif"
    with rasterio.open(path, "w", driver="GTiff", height=N, width=N, count=1, dtype="float32",
                       crs=CRS, transform=from_origin(X0, Y0, CELL, CELL), nodata=-9999) as dst:
        dst.write(z, 1)
    return path


def _max_step(coords):
    c = np.asarray(coords)[:, :2]
    return np.linalg.norm(np.diff(c, axis=0), axis=1).max()


# ---- densify_geometries ---------------------------------------------------------------------


def test_densify_line_keeps_ends_and_spacing():
    line = LineString([(0, 0), (100, 0), (100, 35)])
    out = densify_geometries(gpd.GeoDataFrame(geometry=[line], crs=CRS), spacing=10).geometry[0]
    coords = list(out.coords)
    assert coords[0] == (0, 0) and coords[-1] == (100, 35)
    assert (100, 0) in coords  # original vertices survive
    assert _max_step(coords) <= 10 + 1e-9
    assert out.length == pytest.approx(line.length)


def test_densify_polygon_keeps_holes_and_area():
    poly = Polygon(box(0, 0, 100, 100).exterior.coords, [box(40, 40, 60, 60).exterior.coords])
    out = densify_geometries(gpd.GeoDataFrame(geometry=[poly], crs=CRS), spacing=5).geometry[0]
    assert len(out.interiors) == 1
    assert out.area == pytest.approx(poly.area)
    assert _max_step(out.exterior.coords) <= 5 + 1e-9


def test_densify_multiline_densifies_every_part():
    ml = MultiLineString([[(0, 0), (50, 0)], [(0, 10), (0, 90)]])
    out = densify_geometries(gpd.GeoDataFrame(geometry=[ml], crs=CRS), spacing=10).geometry[0]
    assert len(out.geoms) == 2
    assert all(_max_step(g.coords) <= 10 + 1e-9 for g in out.geoms)


# ---- geometries_to_points -------------------------------------------------------------------


def test_points_carry_attributes_and_ids():
    poly = Polygon(box(0, 0, 10, 10).exterior.coords, [box(4, 4, 6, 6).exterior.coords])
    ml = MultiLineString([[(0, 0), (1, 0)], [(5, 5), (6, 6), (7, 7)]])
    gdf = gpd.GeoDataFrame({"unit": ["a", "b"]}, geometry=[poly, ml], crs=CRS)
    pts = geometries_to_points(gdf)
    assert set(pts["unit"]) == {"a", "b"}
    a, b = pts[pts.unit == "a"], pts[pts.unit == "b"]
    assert set(a["ring_id"]) == {0, 1}                 # exterior and one hole
    assert set(b["part_id"]) == {0, 1}                 # two parts
    assert list(b[b.part_id == 1]["vertex_id"]) == [0, 1, 2]
    assert pts["z"].isna().all()                       # 2D input: no z
    assert (pts["geometry_x"] == pts.geometry.x).all()


# ---- add_z_from_opentopography with a local DEM ------------------------------------------------


def test_points_get_z_from_existing_dem(dem):
    pts = gpd.GeoDataFrame(geometry=[Point(X0 + 105, Y0 - 205), Point(X0 + 455, Y0 - 55)], crs=CRS)
    out, path = add_z_from_opentopography(pts, out_tiff_path=dem, verbose=False)
    assert path == str(dem)
    z = np.array([g.z for g in out.geometry])
    assert z == pytest.approx(_plane(np.array([X0 + 105, X0 + 455]), np.array([Y0 - 205, Y0 - 55])), abs=0.5)


def test_line_drapes_on_dem_vertex_by_vertex(dem):
    line = gpd.GeoDataFrame({"name": ["trace"]},
                            geometry=[LineString([(X0 + 50, Y0 - 50), (X0 + 550, Y0 - 450)])], crs=CRS)
    dense = densify_geometries(line, spacing=25)
    draped, _ = add_z_from_opentopography(dense, out_tiff_path=dem, verbose=False)
    pts = geometries_to_points(draped)
    assert len(pts) == len(dense.geometry[0].coords)
    assert pts["z"].notna().all()
    assert pts["z"].to_numpy() == pytest.approx(_plane(pts.geometry.x.to_numpy(), pts.geometry.y.to_numpy()), abs=0.5)
    assert (pts["name"] == "trace").all()


def test_missing_dem_without_key_explains_how_to_get_one(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENTOPOGRAPHY_API_KEY", raising=False)
    monkeypatch.setattr("geomodeltools._config.OPENTOPOGRAPHY_API_KEY", None, raising=False)
    pts = gpd.GeoDataFrame(geometry=[Point(X0, Y0)], crs=CRS)
    with pytest.raises(ValueError, match="OPENTOPOGRAPHY_API_KEY"):
        add_z_from_opentopography(pts, out_tiff_path=tmp_path / "none.tif", verbose=False)
