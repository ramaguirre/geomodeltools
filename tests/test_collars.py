"""Collars from lat/lon: PSAD56 conversion, block-model footprint, DEM Z. Offline: synthetic DEM, patched download."""
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_bounds, from_origin

import geomodeltools.dem as dem_mod
from geomodeltools import (block_model_bounds, collars_from_latlon, dem_for_block_model, latlon_to_xy,
                           sample_dem_z)

CRS = "EPSG:24879"
X0, Y0, CELL, N = 424_000.0, 7_460_000.0, 30.0, 100      # covers 424000..427000 E, 7457000..7460000 N


def _plane(x, y):
    return 2000.0 + 0.1 * (x - X0) + 0.05 * (Y0 - y)


def _write_dem(path):
    cols = X0 + CELL * (np.arange(N) + 0.5)
    rows = Y0 - CELL * (np.arange(N) + 0.5)
    z = _plane(cols[None, :], rows[:, None]).astype("float32")
    with rasterio.open(path, "w", driver="GTiff", height=N, width=N, count=1, dtype="float32", crs=CRS,
                       transform=from_origin(X0, Y0, CELL, CELL), nodata=-9999) as dst:
        dst.write(z, 1)
    return path


def test_latlon_to_psad56_utm19s_lands_where_cachorro_is():
    x, y = latlon_to_xy(-22.98, -69.73)
    assert abs(float(x) - 425_360) < 100 and abs(float(y) - 7_458_882) < 100   # PSAD56 / UTM 19S
    xs, ys = latlon_to_xy([-22.98, -23.0], [-69.73, -69.7])
    assert xs.shape == ys.shape == (2,) and ys[0] > ys[1]


def test_psad56_differs_from_wgs84_utm_by_the_datum_shift():
    x56, y56 = latlon_to_xy(-22.98, -69.73)
    x84, y84 = latlon_to_xy(-22.98, -69.73, "EPSG:32719")
    shift = float(np.hypot(x56 - x84, y56 - y84))
    assert 100 < shift < 600


def test_block_model_bounds_include_half_blocks():
    bm = pd.DataFrame({"xcentre": [10.0, 30.0], "ycentre": [100.0, 140.0],
                       "xlength": [20.0, 20.0], "ylength": [10.0, 10.0]})
    assert block_model_bounds(bm) == [0.0, 95.0, 40.0, 145.0]
    assert block_model_bounds(bm, size_cols=None) == [10.0, 100.0, 30.0, 140.0]


def test_block_model_bounds_from_csv(tmp_path):
    bm = pd.DataFrame({"xcentre": [10.0], "ycentre": [100.0], "xlength": [20.0], "ylength": [10.0], "extra": [1]})
    bm.to_csv(tmp_path / "bm.csv", index=False)
    assert block_model_bounds(tmp_path / "bm.csv") == [0.0, 95.0, 20.0, 105.0]


def test_sample_dem_z_inside_outside(tmp_path):
    dem = _write_dem(tmp_path / "dem.tif")
    x, y = X0 + 1500.0 + 15.0, Y0 - 1500.0 - 15.0          # a cell centre
    z = sample_dem_z([x, X0 - 10, x], [y, y, Y0 + 10], dem)
    assert z[0] == pytest.approx(_plane(x, y), abs=0.01)
    assert np.isnan(z[1]) and np.isnan(z[2])               # outside: NaN, not extrapolated


def test_collars_from_latlon_adds_xyz(tmp_path):
    dem = _write_dem(tmp_path / "dem.tif")
    df = pd.DataFrame({"hole": ["PCA-521", "FAR"], "lat": [-22.98, -22.5], "lon": [-69.73, -69.73]})
    out = collars_from_latlon(df, dem)
    assert list(df.columns) == ["hole", "lat", "lon"]       # input untouched
    a = out.iloc[0]
    assert a["z"] == pytest.approx(_plane(a["x"], a["y"]), abs=5.0)   # bilinear-free nearest cell on a plane
    assert np.isnan(out.iloc[1]["z"])                       # outside the DEM


def test_dem_for_block_model_requests_footprint_plus_margin(tmp_path, monkeypatch):
    bm = pd.DataFrame({"xcentre": [425_000.0, 426_000.0], "ycentre": [7_458_000.0, 7_459_000.0],
                       "xlength": [10.0, 10.0], "ylength": [10.0, 10.0]})
    seen = {}

    def fake_get(url, params, timeout):
        seen.update(params)
        # a wide tile (+-1 deg) so the reprojection to UTM leaves visible fill at the corners
        w, s, e, n = params["west"] - 1, params["south"] - 1, params["east"] + 1, params["north"] + 1
        with rasterio.MemoryFile() as mf:
            with mf.open(driver="GTiff", height=400, width=400, count=1, dtype="float32", crs="EPSG:4326",
                         transform=from_bounds(w, s, e, n, 400, 400), nodata=-9999) as dst:
                dst.write(np.full((1, 400, 400), 2500.0, dtype="float32"))
            data = mf.read()

        class R:
            status_code, content, text = 200, data, ""
        return R()

    monkeypatch.setattr(dem_mod.requests, "get", fake_get)
    out = tmp_path / "dem.tif"
    path = dem_for_block_model(bm, out, margin_m=300, api_key="x")
    assert seen["demtype"] == "AW3D30"
    # the request must contain footprint + 300 m (checked in the grid CRS, with a few metres of tolerance)
    wx, wy = latlon_to_xy([seen["south"], seen["north"]], [seen["west"], seen["east"]])
    assert min(wx) <= 425_000 - 5 - 300 + 2 and max(wx) >= 426_000 + 5 + 300 - 2
    assert min(wy) <= 7_458_000 - 5 - 300 + 2 and max(wy) >= 7_459_000 + 5 + 300 - 2
    with rasterio.open(path) as src:
        assert src.crs.to_epsg() == 24879
        assert src.nodata is not None
        z = src.read(1, masked=True)
        assert z.mask.any()                                  # reprojection fill is nodata, not elevation 0
        assert (z.compressed() == 2500).all()
    assert dem_for_block_model(bm, out) == str(out)          # existing file: reused, no new request
