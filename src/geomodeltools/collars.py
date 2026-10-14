"""Collar coordinates from a gyro file's lat/lon: to PSAD56 / UTM 19S, then Z from an ALOS (AW3D30) DEM.

Contractor gyro files carry the survey device's position in geographic coordinates (WGS84 lat/lon) and no
Z. These helpers take that to the project grid and drape it on a DEM downloaded for the block model's
footprint (``download_opentopography_dem`` does the download; this module only picks the footprint).

Check the precision of the lat/lon you are given: Cachorro's gyro files carry two decimals (-22.98, -69.73),
about 1.1 km. The X/Y you get is then good to ~1 km and so is the DEM Z on a slope.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import _config as _cfg
from .dem import download_opentopography_dem


def latlon_to_xy(lat, lon, crs=_cfg.DEFAULT_CRS):
    """WGS84 latitude / longitude (degrees) -> (x, y) in ``crs`` (default PSAD56 / UTM 19S, EPSG:24879).

    PROJ picks the datum shift for the location (for northern Chile, "PSAD56 to WGS 84 (15)", ~17 m).
    Returns two float arrays shaped like the input.
    """
    import pyproj

    lat, lon = np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)
    t = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x, y = t.transform(lon, lat)
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


def block_model_bounds(block_model, xy_cols=("xcentre", "ycentre"), size_cols=("xlength", "ylength")):
    """Footprint of a block model: [minx, miny, maxx, maxy] including the half-block edges.

    ``block_model`` is a DataFrame or a path to a .parquet / .csv (parquet needs pyarrow). With
    ``size_cols=None`` the bounds are those of the block centres.
    """
    cols = list(xy_cols) + (list(size_cols) if size_cols else [])
    if isinstance(block_model, (str, Path)):
        p = Path(block_model)
        bm = pd.read_parquet(p, columns=cols) if p.suffix.lower() == ".parquet" else pd.read_csv(p, usecols=cols)
    else:
        bm = block_model[cols]
    half = [bm[c] / 2 for c in size_cols] if size_cols else [0.0, 0.0]
    xc, yc = bm[xy_cols[0]], bm[xy_cols[1]]
    return [float((xc - half[0]).min()), float((yc - half[1]).min()),
            float((xc + half[0]).max()), float((yc + half[1]).max())]


def dem_for_block_model(block_model, out_tiff_path, margin_m=300.0, crs=_cfg.DEFAULT_CRS,
                        xy_cols=("xcentre", "ycentre"), size_cols=("xlength", "ylength"), **kwargs):
    """Download an ALOS World 3D 30 m DEM (AW3D30) covering the block model footprint plus ``margin_m``.

    The GeoTIFF is written in ``crs`` (default PSAD56 / UTM 19S). Block-model coordinates are taken to be in
    ``crs``. An existing ``out_tiff_path`` is reused, no new download. Needs an OpenTopography API key
    (env var OPENTOPOGRAPHY_API_KEY or ``api_key=``); other ``kwargs`` go to ``download_opentopography_dem``.
    Returns the path as a string.
    """
    out = Path(out_tiff_path)
    if out.exists():
        return str(out)
    bounds = block_model_bounds(block_model, xy_cols, size_cols)
    kwargs.setdefault("demtype", "AW3D30")
    return download_opentopography_dem(bounds, out, crs=crs, margin_m=margin_m, out_crs=crs, **kwargs)


def sample_dem_z(x, y, dem_path):
    """Elevation at (x, y) read from a DEM GeoTIFF (x, y in the DEM's CRS). NaN outside the DEM or on nodata."""
    import rasterio

    x, y = np.atleast_1d(np.asarray(x, dtype=float)), np.atleast_1d(np.asarray(y, dtype=float))
    z = np.full(x.shape, np.nan)
    with rasterio.open(dem_path) as src:
        b = src.bounds
        inside = (x >= b.left) & (x <= b.right) & (y >= b.bottom) & (y <= b.top)
        if inside.any():
            vals = np.array([v[0] for v in src.sample(list(zip(x[inside], y[inside])))], dtype=float)
            if src.nodata is not None:
                vals[np.isclose(vals, src.nodata)] = np.nan
            z[inside] = vals
    return z


def collars_from_latlon(df, dem_path, lat_col="lat", lon_col="lon", crs=_cfg.DEFAULT_CRS):
    """Copy of ``df`` with ``x``, ``y`` (lat/lon -> ``crs``) and ``z`` (DEM elevation at x, y) columns.

    Holes whose point falls outside the DEM get z = NaN: the DEM is never extrapolated.
    """
    out = df.copy()
    out["x"], out["y"] = latlon_to_xy(out[lat_col], out[lon_col], crs)
    out["z"] = sample_dem_z(out["x"], out["y"], dem_path)
    return out
