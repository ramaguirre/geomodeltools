from __future__ import annotations

import os
from pathlib import Path

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


def _bounds_to_wgs84(gdf, margin=0.0):
    gdf_wgs84 = gdf.to_crs(epsg=4326) if gdf.crs != "EPSG:4326" else gdf
    west, south, east, north = gdf_wgs84.total_bounds.tolist()
    margin = float(margin)
    return [west - margin, south - margin, east + margin, north + margin]


def _download_opentopography_dem(
    bounds_wgs84,
    out_tiff_path,
    demtype="COP30",
    api_key=None,
    timeout=180,
):
    api_key = api_key or os.getenv("OPENTOPOGRAPHY_API_KEY")
    if not api_key:
        raise ValueError(
            "OpenTopography API key is required. Pass api_key=... or set OPENTOPOGRAPHY_API_KEY."
        )

    west, south, east, north = [float(v) for v in bounds_wgs84]
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
    out_tiff_path.write_bytes(response.content)
    return str(out_tiff_path)


def add_z_from_opentopography(
    gdf_or_path,
    out_tiff_path,
    bounds_utm=True,
    margin=0,
    keep_geometry_old=True,
    verbose=True,
    api_key=None,
    demtype="COP30",
):
    if isinstance(gdf_or_path, (str, Path)):
        gdf = gpd.read_file(gdf_or_path)
    else:
        gdf = gdf_or_path.copy()

    if gdf.empty:
        raise ValueError("Input GeoDataFrame is empty.")
    if gdf.crs is None:
        raise ValueError("Input GeoDataFrame must have a valid CRS.")

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
        bounds_wgs84 = _bounds_to_wgs84(gdf, margin=margin)
        if verbose and bounds_utm:
            print("Converting input bounds to WGS84 for OpenTopography request.")

        dem_path = _download_opentopography_dem(
            bounds_wgs84=bounds_wgs84,
            out_tiff_path=out_tiff_path,
            demtype=demtype,
            api_key=api_key,
        )
        if verbose:
            print(f"DEM downloaded: {dem_path}")

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
