from __future__ import annotations

import os
import tempfile
from pathlib import Path

from . import _config as _cfg
from .geometry import _densify_coords

import geopandas as gpd
import numpy as np
import requests
from shapely.geometry import (
    GeometryCollection,
    LineString,
    LinearRing,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
)


def _first_xyz(geom):
    """
    Extract the first (x, y, z) coordinate from a geometry.

    Parameters
    ----------
    geom : shapely.geometry.base.BaseGeometry
        Input geometry (Point, LineString, Polygon, etc.).

    Returns
    -------
    tuple
        (x, y, z) coordinates as floats. z is np.nan if not present.
    """
    if geom is None or geom.is_empty:
        return np.nan, np.nan, np.nan

    if isinstance(geom, Point):
        z = geom.z if getattr(geom, "has_z", False) else np.nan
        return geom.x, geom.y, z

    if isinstance(geom, (LineString, LinearRing)):
        c = np.asarray(geom.coords, dtype=float)
        if c.shape[1] == 2:
            return c[0, 0], c[0, 1], np.nan
        return c[0, 0], c[0, 1], c[0, 2]

    if isinstance(geom, Polygon):
        return _first_xyz(geom.exterior)

    if isinstance(geom, (MultiPoint, MultiLineString, MultiPolygon, GeometryCollection)):
        if len(geom.geoms) == 0:
            return np.nan, np.nan, np.nan
        return _first_xyz(geom.geoms[0])

    return np.nan, np.nan, np.nan


def _densify_bounds_to_wgs84(bounds, crs, margin_m=0.0, max_segment_length=None):
    """
    Compute a WGS84 bounding box that fully covers the reprojected footprint
    of a bounding rectangle.

    OpenTopography's API only accepts an axis-aligned WGS84 bounding box.
    Reprojecting only the 4 corners of a projected (e.g. UTM) rectangle to
    WGS84 and taking their envelope can under-cover the true extent: a UTM
    grid rectangle is rotated relative to WGS84 meridians away from its
    central meridian, and its edges become curves under reprojection. This
    function guards against that by densifying the rectangle's edges with
    many intermediate vertices *before* reprojecting, then taking the
    envelope of every reprojected vertex.

    Parameters
    ----------
    bounds : sequence of float
        [minx, miny, maxx, maxy] in `crs`.
    crs : any
        CRS of `bounds` (anything geopandas/pyproj accepts).
    margin_m : float, optional
        Margin to pad the bounding rectangle, in `crs`'s linear units
        (meters for a UTM/PSAD56 CRS), applied before densifying and
        reprojecting. Only meaningful for projected CRS's; ignored if `crs`
        is geographic (default 0.0).
    max_segment_length : float, optional
        Maximum spacing (in `crs`'s linear units) between consecutive
        densified vertices along each edge of the bounding rectangle. If
        None (default), computed automatically as
        ``max(width, height) / 100`` so the longest edge gets ~100 segments.

    Returns
    -------
    list
        [west, south, east, north] in WGS84, guaranteed to contain every
        densified vertex of the (margin-padded) source rectangle.
    """
    import pyproj

    minx, miny, maxx, maxy = [float(v) for v in bounds]

    crs = pyproj.CRS.from_user_input(crs)
    if crs.is_geographic:
        return [minx, miny, maxx, maxy]

    margin_m = float(margin_m)
    minx -= margin_m
    miny -= margin_m
    maxx += margin_m
    maxy += margin_m

    width = maxx - minx
    height = maxy - miny
    if max_segment_length is None:
        max_segment_length = max(width, height, 1e-6) / 100.0

    ring_xyz = np.array(
        [
            [minx, miny, np.nan],
            [maxx, miny, np.nan],
            [maxx, maxy, np.nan],
            [minx, maxy, np.nan],
            [minx, miny, np.nan],
        ],
        dtype=float,
    )
    densified = _densify_coords(ring_xyz, spacing=max_segment_length, closed=True)

    pts = gpd.GeoSeries(
        gpd.points_from_xy(densified[:, 0], densified[:, 1]), crs=crs
    ).to_crs(epsg=4326)

    return [pts.x.min(), pts.y.min(), pts.x.max(), pts.y.max()]


def download_opentopography_dem(
    bounds,
    out_tiff_path,
    crs=_cfg.DEFAULT_CRS,
    margin_m=0.0,
    max_segment_length=None,
    demtype="AW3D30",
    api_key=None,
    timeout=180,
    out_crs=_cfg.DEFAULT_CRS,
    resampling="bilinear",
):
    """
    Download a DEM from OpenTopography covering the given bounds.

    Parameters
    ----------
    bounds : sequence of float
        [minx, miny, maxx, maxy] in `crs`. Defaults to PSAD56 / UTM zone 19S
        (EPSG:24879) meters; pass `crs=...` to use a different CRS (e.g. pass
        WGS84 bounds directly with `crs="EPSG:4326"`).
    out_tiff_path : str or Path
        Output path for the DEM GeoTIFF.
    crs : any, optional
        CRS of `bounds` (default `EPSG:24879`, PSAD56 / UTM zone 19S).
        Accepts anything geopandas/pyproj can parse (EPSG int, "EPSG:xxxx",
        WKT, proj4, etc.).
    margin_m : float, optional
        Margin to pad `bounds` by, in `crs`'s linear units (meters for a
        UTM/PSAD56 CRS), applied before converting to WGS84. Ignored if
        `crs` is geographic. Default 0.0.
    max_segment_length : float, optional
        Maximum spacing (in `crs`'s linear units) between densified vertices
        used for the rotation-safe WGS84 bounds conversion (see
        `_densify_bounds_to_wgs84`). If None (default), computed
        automatically from the extent of `bounds`.
    demtype : str, optional
        DEM type (default "AW3D30").
    api_key : str, optional
        OpenTopography API key.
    timeout : int, optional
        Request timeout in seconds.
    out_crs : any, optional
        CRS to reproject the downloaded DEM into before writing to
        `out_tiff_path` (default `EPSG:24879`, PSAD56 / UTM zone 19S). Pass
        `None` or `"EPSG:4326"` to keep the raster in WGS84 as returned by
        OpenTopography (no reprojection).
    resampling : str, optional
        Resampling method used when reprojecting to `out_crs` (default
        "bilinear", appropriate for continuous elevation data). Any name
        from `rasterio.enums.Resampling` (e.g. "nearest", "cubic").

    Returns
    -------
    str
        Path to the downloaded (and, unless `out_crs` is WGS84/None,
        reprojected) DEM file.

    Raises
    ------
    ValueError
        If API key is missing.
    RuntimeError
        If the request fails.

    Notes
    -----
    OpenTopography's API only ever returns rasters in WGS84 (EPSG:4326) —
    this is a constraint of the external service, not configurable. `bounds`
    (in `crs`) is converted to a WGS84 bounding box internally, using a
    rotation-safe method that densifies the bounding rectangle's edges
    before reprojecting (see `_densify_bounds_to_wgs84`) so that grid
    rotation/curvature away from `crs`'s central meridian cannot cause the
    WGS84 request to under-cover `bounds`. The downloaded WGS84 raster is
    then reprojected to `out_crs` (default `EPSG:24879`) before being
    written to `out_tiff_path`.

    Available DEM types:

    - SRTMGL3      : SRTM GL3 90m
    - SRTMGL1      : SRTM GL1 30m
    - SRTMGL1_E    : SRTM GL1 Ellipsoidal 30m
    - AW3D30       : ALOS World 3D 30m
    - AW3D30_E     : ALOS World 3D Ellipsoidal 30m
    - SRTM15Plus   : Global Bathymetry SRTM15+ V2.1 500m
    - NASADEM      : NASADEM Global DEM
    - COP30        : Copernicus Global DSM 30m
    - COP90        : Copernicus Global DSM 90m
    - EU_DTM       : DTM 30m
    - GEDI_L3      : DTM 1000m
    - GEBCOIceTopo    : Global Bathymetry 500m
    - GEBCOSubIceTopo : Global Bathymetry 500m
    - CA_MRDEM_DSM : DSM 30m
    - CA_MRDEM_DTM : DTM 30m
    """
    api_key = api_key or os.getenv("OPENTOPOGRAPHY_API_KEY") or _cfg.OPENTOPOGRAPHY_API_KEY
    if not api_key:
        raise ValueError(
            "OpenTopography API key is required. Pass api_key=... or set OPENTOPOGRAPHY_API_KEY."
        )

    west, south, east, north = _densify_bounds_to_wgs84(
        bounds, crs=crs, margin_m=margin_m, max_segment_length=max_segment_length
    )
    url = "https://portal.opentopography.org/API/globaldem"
    params = {
        "demtype": demtype,
        "south": south,
        "north": north,
        "west": west,
        "east": east,
        "outputFormat": "GTiff",
        "API_Key": api_key,
    }

    response = requests.get(url, params=params, timeout=timeout)
    if response.status_code != 200:
        raise RuntimeError(
            f"OpenTopography request failed ({response.status_code}): {response.text[:400]}"
        )

    out_tiff_path = Path(out_tiff_path)
    out_tiff_path.parent.mkdir(parents=True, exist_ok=True)

    import rasterio
    from rasterio.crs import CRS as RasterioCRS
    from rasterio.enums import Resampling
    from rasterio.warp import calculate_default_transform, reproject

    target_crs = RasterioCRS.from_user_input(out_crs) if out_crs is not None else None

    if target_crs is None or target_crs == RasterioCRS.from_epsg(4326):
        out_tiff_path.write_bytes(response.content)
        return str(out_tiff_path)

    tmp_file = tempfile.NamedTemporaryFile(
        dir=out_tiff_path.parent, suffix=".tif", delete=False
    )
    tmp_path = Path(tmp_file.name)
    tmp_file.close()
    try:
        tmp_path.write_bytes(response.content)
        with rasterio.open(tmp_path) as src:
            transform, width, height = calculate_default_transform(
                src.crs, target_crs, src.width, src.height, *src.bounds
            )
            profile = src.profile.copy()
            profile.update(crs=target_crs, transform=transform, width=width, height=height)

            with rasterio.open(out_tiff_path, "w", **profile) as dst:
                for band in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, band),
                        destination=rasterio.band(dst, band),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=target_crs,
                        resampling=Resampling[resampling],
                    )
    finally:
        tmp_path.unlink(missing_ok=True)

    return str(out_tiff_path)


def add_z_from_opentopography(
    gdf_or_path,
    out_tiff_path,
    crs=None,
    margin_m=0.0,
    max_segment_length=None,
    keep_geometry_old=True,
    verbose=True,
    api_key=None,
    demtype="AW3D30",
    resampling="bilinear",
):
    """
    Add Z (elevation) values to geometries from an OpenTopography DEM.

    Parameters
    ----------
    gdf_or_path : geopandas.GeoDataFrame or str or Path
        Input GeoDataFrame or path to a file.
    out_tiff_path : str or Path
        Output path for the DEM GeoTIFF.
    crs : any, optional
        CRS to assume when the input has none set, and the CRS the DEM
        GeoTIFF is written in. Defaults to `EPSG:24879` (PSAD56 / UTM zone
        19S) if not given. Accepts anything geopandas/pyproj can parse. Does
        NOT reproject an input that already has its own CRS set — see Notes.
    margin_m : float, optional
        Margin to pad the input's bounds by, in the input's linear units
        (meters for a UTM/PSAD56 CRS), applied before the rotation-safe
        conversion to WGS84 for the OpenTopography request. Default 0.0.
    max_segment_length : float, optional
        Maximum spacing (in the input's linear units) between densified
        vertices used for the rotation-safe WGS84 bounds conversion (see
        `_densify_bounds_to_wgs84`). If None (default), computed
        automatically from the extent of the input's bounds.
    keep_geometry_old : bool, optional
        If True, store original geometry in 'geometry_old' if Z exists.
    verbose : bool, optional
        If True, print progress messages.
    api_key : str, optional
        OpenTopography API key.
    demtype : str, optional
        DEM type (default "AW3D30").
    resampling : str, optional
        Resampling method used when reprojecting the downloaded DEM to `crs`
        (default "bilinear"). Any name from `rasterio.enums.Resampling`.

    Returns
    -------
    out_gdf : geopandas.GeoDataFrame
        GeoDataFrame with Z values added to geometry and columns 'x', 'y', 'z'.
        Returned in the input's CRS (or the resolved `crs` default/override
        if the input had none) — see Notes.
    dem_path : str
        Path to the DEM file used.

    Raises
    ------
    ValueError
        If input is empty.

    Notes
    -----
    If the input has no CRS set, `crs` (default `EPSG:24879`) is assumed and
    assigned to it; an input that already has its own CRS set is never
    silently reprojected. The output GeoDataFrame is returned in the input's
    CRS (so it defaults to `EPSG:24879` only when the input was CRS-less);
    the DEM GeoTIFF written to `out_tiff_path`, however, is always in `crs`
    (default `EPSG:24879`) regardless of the input's CRS.

    If `out_tiff_path` already exists, it is reused as-is (no re-download or
    re-reprojection) — this means reusing the same path across calls with a
    different `crs` will silently serve the DEM in whatever CRS it was first
    written in; delete the file to force a fresh download in a new CRS.

    Available DEM types:

    - SRTMGL3      : SRTM GL3 90m
    - SRTMGL1      : SRTM GL1 30m
    - SRTMGL1_E    : SRTM GL1 Ellipsoidal 30m
    - AW3D30       : ALOS World 3D 30m
    - AW3D30_E     : ALOS World 3D Ellipsoidal 30m
    - SRTM15Plus   : Global Bathymetry SRTM15+ V2.1 500m
    - NASADEM      : NASADEM Global DEM
    - COP30        : Copernicus Global DSM 30m
    - COP90        : Copernicus Global DSM 90m
    - EU_DTM       : DTM 30m
    - GEDI_L3      : DTM 1000m
    - GEBCOIceTopo    : Global Bathymetry 500m
    - GEBCOSubIceTopo : Global Bathymetry 500m
    - CA_MRDEM_DSM : DSM 30m
    - CA_MRDEM_DTM : DTM 30m
    """
    if isinstance(gdf_or_path, (str, Path)):
        gdf = gpd.read_file(gdf_or_path)
    else:
        gdf = gdf_or_path.copy()

    if gdf.empty:
        raise ValueError("Input GeoDataFrame is empty.")

    effective_crs = crs if crs is not None else _cfg.DEFAULT_CRS
    if gdf.crs is None:
        if verbose:
            print(f"Input has no CRS set; assuming {effective_crs}.")
        gdf = gdf.set_crs(effective_crs)

    def _geom_has_z(geom):
        if geom is None or geom.is_empty:
            return False
        if hasattr(geom, "has_z") and geom.has_z:
            return True
        if hasattr(geom, "geoms"):
            return any(_geom_has_z(g) for g in geom.geoms)
        return False

    had_z = gdf.geometry.apply(_geom_has_z).any()
    if had_z and keep_geometry_old:
        gdf["geometry_old"] = gdf.geometry.copy()
        if verbose:
            print("Input already has elevation. Original geometry stored in 'geometry_old'.")
    elif had_z and verbose:
        print("Input already has elevation. Existing Z values will be replaced.")

    out_tiff_path = Path(out_tiff_path)
    out_tiff_path.parent.mkdir(parents=True, exist_ok=True)

    if out_tiff_path.exists():
        dem_path = str(out_tiff_path)
        if verbose:
            print(f"Using existing DEM: {dem_path}")
    else:
        if verbose:
            print("Computing rotation-safe WGS84 bounds for OpenTopography request.")

        dem_path = download_opentopography_dem(
            bounds=gdf.total_bounds,
            out_tiff_path=out_tiff_path,
            crs=gdf.crs,
            margin_m=margin_m,
            max_segment_length=max_segment_length,
            demtype=demtype,
            api_key=api_key,
            out_crs=effective_crs,
            resampling=resampling,
        )
        if verbose:
            print(f"DEM downloaded and reprojected to {effective_crs}: {dem_path}")

    import rasterio

    with rasterio.open(str(dem_path)) as src:
        raster_crs = src.crs
        raster_nodata = src.nodata

        gdf_sample = gdf.to_crs(raster_crs) if gdf.crs != raster_crs else gdf.copy()

        def _sample_z(xy_coords):
            if len(xy_coords) == 0:
                return np.array([], dtype=float)
            vals = np.array([v[0] for v in src.sample(xy_coords)], dtype=float)
            if raster_nodata is not None:
                vals[np.isclose(vals, raster_nodata)] = np.nan
            return vals

        def _add_z_to_geom(geom):
            # Recursively add Z values sampled from the DEM to all geometry types
            if geom is None or geom.is_empty:
                return geom

            if isinstance(geom, Point):
                z = _sample_z([(geom.x, geom.y)])[0]
                return Point(geom.x, geom.y, z)

            if isinstance(geom, LineString):
                coords = np.asarray(geom.coords, dtype=float)
                xy = [tuple(c[:2]) for c in coords]
                z = _sample_z(xy)
                xyz = [(x, y, zv) for (x, y), zv in zip(xy, z)]
                return LineString(xyz)

            if isinstance(geom, LinearRing):
                coords = np.asarray(geom.coords, dtype=float)
                xy = [tuple(c[:2]) for c in coords]
                z = _sample_z(xy)
                xyz = [(x, y, zv) for (x, y), zv in zip(xy, z)]
                return LinearRing(xyz)

            if isinstance(geom, Polygon):
                ext = _add_z_to_geom(geom.exterior)
                holes = [_add_z_to_geom(r) for r in geom.interiors]
                return Polygon(ext, holes)

            if isinstance(geom, MultiPoint):
                return MultiPoint([_add_z_to_geom(g) for g in geom.geoms])

            if isinstance(geom, MultiLineString):
                return MultiLineString([_add_z_to_geom(g) for g in geom.geoms])

            if isinstance(geom, MultiPolygon):
                return MultiPolygon([_add_z_to_geom(g) for g in geom.geoms])

            if isinstance(geom, GeometryCollection):
                return GeometryCollection([_add_z_to_geom(g) for g in geom.geoms])

            return geom

        gdf_sample["geometry"] = gdf_sample.geometry.apply(_add_z_to_geom)

    out_gdf = gdf_sample.to_crs(gdf.crs) if gdf_sample.crs != gdf.crs else gdf_sample

    xyz = out_gdf.geometry.apply(_first_xyz)
    out_gdf["x"] = xyz.apply(lambda t: t[0])
    out_gdf["y"] = xyz.apply(lambda t: t[1])
    out_gdf["z"] = xyz.apply(lambda t: t[2])

    if verbose:
        print("Z values added from DEM to geometries.")

    return out_gdf, str(dem_path)
