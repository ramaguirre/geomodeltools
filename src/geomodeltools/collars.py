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
    from .dem import download_opentopography_dem

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


def _triangle_grid(vx, vy, tri):
    """Uniform 2D grid over the surface; each cell lists the triangles whose bounding box touches it (CSR)."""
    tx, ty = vx[tri], vy[tri]                                          # (k, 3)
    bx0, bx1, by0, by1 = tx.min(1), tx.max(1), ty.min(1), ty.max(1)
    x0, y0, x1, y1 = bx0.min(), by0.min(), bx1.max(), by1.max()
    w, h = max(x1 - x0, 1e-9), max(y1 - y0, 1e-9)
    cell = float(np.median(np.maximum(bx1 - bx0, by1 - by0)))
    cell = max(cell, np.sqrt(w * h / (4 * len(tri))), 1e-9)          # at most ~4 cells per triangle
    nx, ny = int(w // cell) + 1, int(h // cell) + 1
    i0 = np.clip(((bx0 - x0) // cell).astype(np.int64), 0, nx - 1)
    i1 = np.clip(((bx1 - x0) // cell).astype(np.int64), 0, nx - 1)
    j0 = np.clip(((by0 - y0) // cell).astype(np.int64), 0, ny - 1)
    j1 = np.clip(((by1 - y0) // cell).astype(np.int64), 0, ny - 1)
    span = i1 - i0 + 1
    n = span * (j1 - j0 + 1)
    rep = np.repeat(np.arange(len(tri)), n)
    k = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    cells = (j0[rep] + k // span[rep]) * nx + i0[rep] + k % span[rep]
    order = np.argsort(cells, kind="stable")
    counts = np.bincount(cells, minlength=nx * ny)
    return (x0, y0, cell, nx, ny), rep[order], counts, np.cumsum(counts) - counts


def z_on_surface(x, y, vertices, triangles, chunk_pairs=4_000_000):
    """Elevation at (x, y) on a triangulated surface: linear inside its own triangles, NaN outside.

    ``vertices``: (m, 3) array; ``triangles``: (k, 3) vertex indices. The surface is never re-triangulated.
    Where several triangles lie over the same (x, y) (an overhang, a fold) the highest z is returned.
    Triangles are indexed on a 2D grid, so 1e5 points against 1e6 triangles takes seconds. The mesh
    counterpart of ``sample_dem_z``.
    """
    x, y = np.atleast_1d(np.asarray(x, dtype=float)), np.atleast_1d(np.asarray(y, dtype=float))
    v = np.asarray(vertices, dtype=float)
    tri = np.asarray(triangles, dtype=np.int64)
    z = np.full(x.shape, -np.inf)
    if len(tri) == 0 or x.size == 0:
        return np.full(x.shape, np.nan)
    vx, vy, vz = v[:, 0], v[:, 1], v[:, 2]

    # barycentric set-up per triangle; vertical (zero-area in plan) triangles carry no elevation of their own
    ax, ay = vx[tri[:, 0]], vy[tri[:, 0]]
    e1x, e1y = vx[tri[:, 1]] - ax, vy[tri[:, 1]] - ay
    e2x, e2y = vx[tri[:, 2]] - ax, vy[tri[:, 2]] - ay
    det = e1x * e2y - e2x * e1y
    good = np.abs(det) > 1e-12 * np.hypot(e1x, e1y) * np.hypot(e2x, e2y)
    if not good.any():
        return np.full(x.shape, np.nan)
    tri, ax, ay, e1x, e1y, e2x, e2y, det = (a[good] for a in (tri, ax, ay, e1x, e1y, e2x, e2y, det))
    az, d1z, d2z = vz[tri[:, 0]], vz[tri[:, 1]] - vz[tri[:, 0]], vz[tri[:, 2]] - vz[tri[:, 0]]

    (gx0, gy0, cell, nx, ny), cell_tris, counts, starts = _triangle_grid(vx, vy, tri)
    ci = np.floor((x - gx0) / cell)
    cj = np.floor((y - gy0) / cell)
    ingrid = (ci >= 0) & (ci < nx) & (cj >= 0) & (cj < ny)
    pidx = np.flatnonzero(ingrid)
    pcell = (cj[pidx] * nx + ci[pidx]).astype(np.int64)
    pcount = counts[pcell]
    pidx, pcell, pcount = pidx[pcount > 0], pcell[pcount > 0], pcount[pcount > 0]

    # chunks of points so the (point, candidate triangle) pairs stay within chunk_pairs
    if len(pidx):
        cum = np.cumsum(pcount)
        edges = np.unique(np.r_[0, np.searchsorted(cum, np.arange(chunk_pairs, cum[-1], chunk_pairs)), len(pidx)])
    else:
        edges = np.array([0])
    for lo, hi in zip(edges[:-1], edges[1:]):
        p, c, n = pidx[lo:hi], pcell[lo:hi], pcount[lo:hi]
        pt = np.repeat(p, n)
        t = cell_tris[np.repeat(starts[c], n) + np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)]
        dx, dy = x[pt] - ax[t], y[pt] - ay[t]
        l1 = (dx * e2y[t] - e2x[t] * dy) / det[t]
        l2 = (e1x[t] * dy - dx * e1y[t]) / det[t]
        inside = (l1 >= -1e-9) & (l2 >= -1e-9) & (l1 + l2 <= 1 + 1e-9)
        if inside.any():
            np.maximum.at(z, pt[inside], (az[t] + l1 * d1z[t] + l2 * d2z[t])[inside])
    z[np.isneginf(z)] = np.nan
    return z


def collars_from_latlon(df, dem_path, lat_col="lat", lon_col="lon", crs=_cfg.DEFAULT_CRS):
    """Copy of ``df`` with ``x``, ``y`` (lat/lon -> ``crs``) and ``z`` (DEM elevation at x, y) columns.

    Holes whose point falls outside the DEM get z = NaN: the DEM is never extrapolated.
    """
    out = df.copy()
    out["x"], out["y"] = latlon_to_xy(out[lat_col], out[lon_col], crs)
    out["z"] = sample_dem_z(out["x"], out["y"], dem_path)
    return out
