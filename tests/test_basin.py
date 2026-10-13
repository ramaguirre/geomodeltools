"""Synthetic data only: a planar sloping DEM and circular / thin polygons with a known answer."""
import numpy as np
import pytest
import rasterio
import shapely
from rasterio.transform import from_origin
from scipy import integrate

from geomodeltools import basin as bs

CRS = "EPSG:32719"
X0, Y0 = 350000.0, 6460000.0


@pytest.fixture
def dem(tmp_path):
    """6 km x 6 km, 30 m cells, z = 2000 + 0.05 * (x - X0) + 0.02 * (y - Y0)."""
    n = 200
    xs = X0 + 30 * (np.arange(n) + 0.5)
    ys = Y0 + 6000 - 30 * (np.arange(n) + 0.5)
    z = (2000 + 0.05 * (xs[None, :] - X0) + 0.02 * (ys[:, None] - Y0)).astype("float32")
    path = tmp_path / "dem.tif"
    with rasterio.open(path, "w", driver="GTiff", height=n, width=n, count=1, dtype="float32", crs=CRS,
                       transform=from_origin(X0, Y0 + 6000, 30, 30)) as dst:
        dst.write(z, 1)
    return path


BBOX = (X0 + 500, X0 + 5500, Y0 + 500, Y0 + 5500)
PARAMS = bs.BasinParams(z_max=60.0, L=120.0, edge_spacing=50.0, grid_cell=25.0, max_edge=800.0, refine_passes=2)


def test_circular_basin_qa_and_volume(dem, tmp_path):
    R = 900.0
    circle = shapely.Point(X0 + 3000, Y0 + 3000).buffer(R, 64)
    res = bs.build_basin_surface(circle, CRS, dem, BBOX, PARAMS, out_dir=tmp_path, name="c", verbose=False)
    q = res.qa
    assert q["degenerate_triangles"] == 0 and q["flipped_triangles"] == 0
    assert q["min_angle_deg"]["min"] > 15.0
    assert q["ring_max_abs_dz"] < 1e-6 and q["ring_vertices"] > 100
    assert q["inside_base_below_top"] and q["outside_base_above_top"]
    # independent volume: integral over r of z_max (1 - exp(-(R - r)/L)) 2 pi r dr
    expected, _ = integrate.quad(lambda r: PARAMS.z_max * (1 - np.exp(-(R - r) / PARAMS.L)) * 2 * np.pi * r, 0, R)
    assert q["basin_volume_m3"] == pytest.approx(expected, rel=0.06)
    for name in ("c_top_local.ply", "c_base_local.ply", "c_top.obj", "c_base.obj", "c_frame.json", "c_transects.png"):
        assert (tmp_path / name).exists()
    # top and base share the triangulation: same vertex count and the ring vertices coincide
    top, base = res.surface("top"), res.surface("base")
    assert top.shape == base.shape
    ring = np.abs(res.d) < 1e-6
    assert np.allclose(top[ring], base[ring], atol=1e-6)
    # real coordinates restored
    real = res.surface("top", real_coords=True)
    assert real[:, 0].min() > X0 and real[:, 2].min() > 1900


def test_thin_river_like_polygon_is_meshed(dem):
    line = shapely.LineString([(X0 + 1000, Y0 + 1000), (X0 + 2500, Y0 + 2200), (X0 + 4200, Y0 + 2400), (X0 + 4900, Y0 + 4300)])
    river = line.buffer(35.0)                       # about 70 m wide: thinner than two edge spacings
    p = bs.BasinParams(z_max=10.0, L=30.0, edge_spacing=25.0, grid_cell=12.5, max_edge=600.0, refine_passes=2)
    res = bs.build_basin_surface(river, CRS, dem, BBOX, p, verbose=False)
    q = res.qa
    assert q["flipped_triangles"] == 0 and q["ring_max_abs_dz"] < 1e-6 and q["outside_base_above_top"]
    assert 0 < q["mean_depth_m"] < p.z_max


def test_z_base_formula():
    p = bs.BasinParams(z_max=50, L=100, h_min=4, h_slope=0.1, ramp_width=50)
    assert bs.z_base(1000.0, 0.0, p) == pytest.approx(1000.0)
    assert bs.z_base(1000.0, 1e9, p) == pytest.approx(950.0)
    assert bs.z_base(1000.0, 100.0, p) == pytest.approx(1000 - 50 * (1 - np.exp(-1)))
    assert bs.z_base(1000.0, -100.0, p) == pytest.approx(1000 + 4 * 1.0 + 0.1 * 100)       # ramp saturated at w = 50
    assert bs.z_base(1000.0, -1e-9, p) == pytest.approx(1000.0, abs=1e-6)                  # meets the ring


def test_polygon_touching_the_box_is_cut_back(dem, capsys):
    big = shapely.box(BBOX[0] - 500, BBOX[2] + 1000, BBOX[0] + 1500, BBOX[2] + 2500)
    res = bs.build_basin_surface(big, CRS, dem, BBOX, PARAMS, verbose=False)
    assert "cut off" in capsys.readouterr().out
    assert res.qa["flipped_triangles"] == 0


def test_older_basin_continues_under_a_younger_cover_and_never_touches_it(dem):
    circle = shapely.Point(X0 + 3000, Y0 + 3000).buffer(900.0, 64)
    channel = shapely.box(X0 + 2940, Y0 + 1800, X0 + 3060, Y0 + 4200)          # crosses the whole basin, 120 m wide
    p_young = bs.BasinParams(z_max=10.0, L=30.0, edge_spacing=25.0, grid_cell=12.5, max_edge=800.0, refine_passes=2)
    young = bs.build_basin_surface(channel.intersection(circle), CRS, dem, BBOX, p_young, verbose=False)
    old_poly = circle.difference(channel)                                           # the map shows the older basin in two halves
    p_old = bs.BasinParams(z_max=60.0, L=120.0, edge_spacing=37.0, grid_cell=18.5, max_edge=800.0, refine_passes=2)
    bbox_old = (BBOX[0] + 10, BBOX[1] - 10, BBOX[2] + 10, BBOX[3] - 10)             # different box: no shared box vertices
    res = bs.build_basin_surface(old_poly, CRS, dem, bbox_old, p_old, verbose=False, cover=channel.intersection(circle),
                                 younger=young, gap=1.0)
    q = res.qa
    assert q["flipped_triangles"] == 0 and q["degenerate_triangles"] == 0
    # the basin does not close on the channel: depth under the channel centre is a large fraction of the depth beside it
    x0, y0 = (BBOX[0] + 10 + BBOX[1] - 10) / 2, (BBOX[2] + 10 + BBOX[3] - 10) / 2
    xy = res.vertices
    under = (np.abs(xy[:, 0] - (X0 + 3000 - x0)) < 150) & (np.abs(xy[:, 1] - (Y0 + 3000 - y0)) < 300)   # along the channel centre
    assert under.sum() >= 3 and (res.z_top - res.z_base)[under].mean() > 25.0      # the basin continues under the channel
    assert q["mean_depth_m"] > 15.0
    # no vertex of the older surface coincides with one of the younger surface
    ya = young.surface("base", real_coords=True)[:, :2].round(4)
    oa = res.surface("base", real_coords=True)[:, :2].round(4)
    assert len({tuple(r) for r in ya} & {tuple(r) for r in oa}) == 0


def _sharp_vertices(geom, deg=60.0):
    """Number of ring vertices where the outline turns by more than `deg` degrees."""
    n = 0
    for g in getattr(geom, "geoms", [geom]):
        for ring in [g.exterior, *g.interiors]:
            xy = np.asarray(ring.coords)[:-1]
            a, b = xy - np.roll(xy, 1, axis=0), np.roll(xy, -1, axis=0) - xy
            turn = np.degrees(np.abs(np.arctan2(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0], (a * b).sum(1))))
            n += int((turn > deg).sum())
    return n


def test_filled_outline_is_smooth_and_has_the_right_area(dem):
    circle = shapely.Point(X0 + 3000, Y0 + 3000).buffer(900.0, 64)
    channel = shapely.box(X0 + 2940, Y0 + 1800, X0 + 3060, Y0 + 4200).intersection(circle)
    old_poly = circle.difference(channel)
    p = bs.BasinParams(z_max=60.0, L=120.0, edge_spacing=37.0, grid_cell=18.5, max_edge=800.0, refine_passes=2)
    res = bs.build_basin_surface(old_poly, CRS, dem, BBOX, p, verbose=False, cover=channel)
    out = bs.filled_outline(res)
    assert out.is_valid
    assert out.area == pytest.approx(circle.area, rel=0.03)                         # original + the covered part
    assert out.area > old_poly.area
    assert out.symmetric_difference(circle).area < 0.03 * circle.area
    # no raster staircase: the old grid vectorisation had hundreds of right-angle corners, this one has only the real ones
    assert _sharp_vertices(out) <= 12
