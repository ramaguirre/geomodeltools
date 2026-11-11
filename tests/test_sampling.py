"""points_in_polygons (spaced points inside polygons) and z_on_surface (elevation on a triangulated mesh)."""
import time

import numpy as np
import pytest
import shapely
from scipy.spatial import cKDTree

from geomodeltools import points_in_polygons, z_on_surface


def _rect(x0, y0, w, h):
    return np.array([[x0, y0], [x0 + w, y0], [x0 + w, y0 + h], [x0, y0 + h]], float)


def _closest_pair(df):
    xy = df[["x", "y"]].to_numpy()
    return cKDTree(xy).query(xy, k=2)[0][:, 1].min()


# --- points_in_polygons ----------------------------------------------------------------------------------------

def test_narrow_strip_gets_one_point_flagged_margin_not_ok():
    out = points_in_polygons([_rect(0, 0, 200, 6)], d_min=50, margin=5)
    assert len(out) == 1
    assert not out.margin_ok.iloc[0]
    assert out.y.iloc[0] == pytest.approx(3, abs=1.0)            # on the strip's centre line
    assert out.attrs["missing"] == []


def test_wide_strip_with_margin_is_filled_and_margin_ok():
    out = points_in_polygons([_rect(0, 0, 300, 40)], d_min=50, margin=5)
    assert len(out) >= 4
    assert out.margin_ok.all() and (out.edge_dist >= 5).all()


def test_no_pair_closer_than_d_min_and_each_point_in_its_polygon():
    polys = [_rect(0, 0, 400, 300), _rect(410, 0, 6, 300), _rect(0, 310, 250, 80),
             np.array([[500, 0], [800, 0], [650, 250]], float)]
    out = points_in_polygons(polys, d_min=30, margin=2, seed=3)
    assert len(out) > 50
    assert _closest_pair(out) >= 30 - 1e-9
    for pid, g in out.groupby("poly_id"):
        assert shapely.contains_xy(shapely.Polygon(polys[pid]), g.x.to_numpy(), g.y.to_numpy()).all()
    assert sorted(out.poly_id.unique()) == [0, 1, 2, 3]


def test_cluster_closer_than_d_min_reports_missing_polygons():
    polys = [_rect(10 * i, 0, 5, 5) for i in range(5)]          # five 5 m squares, 5 m apart
    out = points_in_polygons(polys, d_min=100)
    assert len(out) == 1
    assert sorted(out.attrs["missing"] + out.poly_id.tolist()) == [0, 1, 2, 3, 4]
    assert len(out.attrs["missing"]) == 4


def test_zero_area_polygon_is_missing():
    out = points_in_polygons([_rect(0, 0, 50, 50), np.array([[100, 0], [110, 0], [120, 0]], float)], d_min=10)
    assert out.attrs["missing"] == [1]


def test_same_seed_same_points_other_seed_differs():
    polys = [_rect(0, 0, 300, 200), _rect(320, 0, 8, 150)]
    a = points_in_polygons(polys, d_min=25, margin=3, seed=7)
    b = points_in_polygons(polys, d_min=25, margin=3, seed=7)
    c = points_in_polygons(polys, d_min=25, margin=3, seed=8)
    assert a.equals(b)
    assert not (len(a) == len(c) and np.allclose(a[["x", "y"]], c[["x", "y"]]))


def test_dict_fields_copied_and_z_from_3d_vertices():
    p3 = np.column_stack([_rect(0, 0, 100, 100), np.full(4, 1500.0)])
    out = points_in_polygons([{"xy": _rect(200, 0, 100, 100), "z": 2100.0, "layer": "PAD_A"}, p3], d_min=40)
    assert list(out.columns[:6]) == ["x", "y", "z", "poly_id", "edge_dist", "margin_ok"]
    a, b = out[out.poly_id == 0], out[out.poly_id == 1]
    assert (a.z == 2100.0).all() and (a.layer == "PAD_A").all()
    assert (b.z == 1500.0).all() and b.layer.isna().all()


# --- z_on_surface ----------------------------------------------------------------------------------------------

def _grid_mesh(n, x0=0.0, y0=0.0, size=100.0, zfun=lambda x, y: 0 * x):
    g = np.linspace(0, size, n + 1)
    gx, gy = np.meshgrid(g + x0, g + y0)
    v = np.column_stack([gx.ravel(), gy.ravel(), zfun(gx.ravel(), gy.ravel())])
    idx = np.arange((n + 1) ** 2).reshape(n + 1, n + 1)
    a, b, c, d = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, 1:].ravel(), idx[1:, :-1].ravel()
    return v, np.vstack([np.column_stack([a, b, c]), np.column_stack([a, c, d])])


def plane(x, y):
    return 1000.0 + 0.3 * x - 0.2 * y


def test_exact_on_tilted_plane_and_nan_outside():
    v, t = _grid_mesh(20, zfun=plane)
    rng = np.random.default_rng(0)
    x, y = rng.uniform(0, 100, 5000), rng.uniform(0, 100, 5000)
    np.testing.assert_allclose(z_on_surface(x, y, v, t), plane(x, y), rtol=0, atol=1e-9)
    np.testing.assert_allclose(z_on_surface(v[:, 0], v[:, 1], v, t), v[:, 2], atol=1e-9)   # on vertices/edges
    out = z_on_surface([-1, 50, 101, 50], [50, -0.5, 50, 1e6], v, t)
    assert np.isnan(out).all()


def test_folded_surface_returns_highest_sheet():
    lo_v, lo_t = _grid_mesh(10, zfun=lambda x, y: 0 * x)
    hi_v, hi_t = _grid_mesh(10, x0=50, zfun=lambda x, y: 10 + 0.1 * x)   # overhang over x in 50..100
    v = np.vstack([lo_v, hi_v])
    t = np.vstack([lo_t, hi_t + len(lo_v)])
    x, y = np.array([25.0, 75.0, 125.0]), np.array([50.0, 50.0, 50.0])
    np.testing.assert_allclose(z_on_surface(x, y, v, t), [0.0, 17.5, 22.5])


def test_vertical_triangles_ignored_and_scalar_input():
    v = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0], [0, 0, 50], [10, 0, 50]], float)
    t = np.array([[0, 1, 2], [0, 1, 4], [0, 4, 3]])              # one flat, two vertical
    assert z_on_surface(2, 2, v, t)[0] == pytest.approx(0.0)


def test_many_points_many_triangles_is_fast():
    v, t = _grid_mesh(500, size=5000.0, zfun=plane)                     # 500k triangles
    rng = np.random.default_rng(1)
    x, y = rng.uniform(0, 5000, 100_000), rng.uniform(0, 5000, 100_000)
    t0 = time.perf_counter()
    z = z_on_surface(x, y, v, t)
    assert time.perf_counter() - t0 < 20
    np.testing.assert_allclose(z, plane(x, y), atol=1e-6)
