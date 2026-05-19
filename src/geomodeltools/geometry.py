from __future__ import annotations

import unicodedata
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from shapely.geometry import (
    GeometryCollection,
    LineString,
    LinearRing,
    MultiLineString,
    MultiPolygon,
    Point,
    Polygon,
)


def _coords_array_with_optional_z(linear_geom):
    coords = np.array(linear_geom.coords, dtype=float)
    if coords.shape[1] == 2:
        coords = np.column_stack([coords, np.full(coords.shape[0], np.nan)])
    return coords


def _densify_coords(coords_xyz, spacing, closed=False):
    if spacing <= 0:
        raise ValueError("spacing must be > 0")
    if coords_xyz.shape[0] < 2:
        return coords_xyz

    xy = coords_xyz[:, :2]
    z = coords_xyz[:, 2]
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    cumdist = np.insert(np.cumsum(seg), 0, 0.0)
    total = cumdist[-1]

    if total == 0:
        return coords_xyz[:1].copy()

    sample_dist = np.arange(0.0, total, spacing)
    if sample_dist.size == 0 or sample_dist[-1] != total:
        sample_dist = np.append(sample_dist, total)

    x = np.interp(sample_dist, cumdist, xy[:, 0])
    y = np.interp(sample_dist, cumdist, xy[:, 1])

    if np.all(np.isnan(z)):
        z_out = np.full_like(x, np.nan, dtype=float)
    else:
        valid = ~np.isnan(z)
        if valid.sum() == 1:
            z_out = np.full_like(x, z[valid][0], dtype=float)
        else:
            z_out = np.interp(sample_dist, cumdist[valid], z[valid])

    out = np.column_stack([x, y, z_out])

    if closed:
        if not np.allclose(out[0, :2], out[-1, :2]):
            out = np.vstack([out, out[0]])
        else:
            out[-1] = out[0]

    return out


def _line_from_coords(coords_xyz):
    if np.isnan(coords_xyz[:, 2]).all():
        return LineString(coords_xyz[:, :2])
    return LineString([(r[0], r[1], r[2]) for r in coords_xyz])


def _ring_from_coords(coords_xyz):
    if coords_xyz.shape[0] < 4:
        return None
    if np.isnan(coords_xyz[:, 2]).all():
        return LinearRing(coords_xyz[:, :2])
    return LinearRing([(r[0], r[1], r[2]) for r in coords_xyz])


def densify_geometries(gdf_or_path, spacing, geometry_col="geometry"):
    if isinstance(gdf_or_path, (str, Path)):
        gdf = gpd.read_file(gdf_or_path)
    else:
        gdf = gdf_or_path.copy()

    def _densify_one(geom):
        if geom is None or geom.is_empty:
            return geom

        if isinstance(geom, LineString):
            c = _coords_array_with_optional_z(geom)
            return _line_from_coords(_densify_coords(c, spacing, closed=False))

        if isinstance(geom, MultiLineString):
            parts = []
            for part in geom.geoms:
                c = _coords_array_with_optional_z(part)
                parts.append(_line_from_coords(_densify_coords(c, spacing, closed=False)))
            return MultiLineString(parts)

        if isinstance(geom, Polygon):
            ext = _coords_array_with_optional_z(geom.exterior)
            ext_d = _ring_from_coords(_densify_coords(ext, spacing, closed=True))
            if ext_d is None:
                return geom

            holes_d = []
            for ring in geom.interiors:
                rc = _coords_array_with_optional_z(ring)
                ring_d = _ring_from_coords(_densify_coords(rc, spacing, closed=True))
                if ring_d is not None:
                    holes_d.append(ring_d)
            return Polygon(ext_d, holes_d)

        if isinstance(geom, MultiPolygon):
            polys = []
            for poly in geom.geoms:
                ext = _coords_array_with_optional_z(poly.exterior)
                ext_d = _ring_from_coords(_densify_coords(ext, spacing, closed=True))
                if ext_d is None:
                    continue
                holes_d = []
                for ring in poly.interiors:
                    rc = _coords_array_with_optional_z(ring)
                    ring_d = _ring_from_coords(_densify_coords(rc, spacing, closed=True))
                    if ring_d is not None:
                        holes_d.append(ring_d)
                polys.append(Polygon(ext_d, holes_d))
            return MultiPolygon(polys) if polys else geom

        if isinstance(geom, GeometryCollection):
            densified = []
            for g in geom.geoms:
                if isinstance(g, (LineString, MultiLineString, Polygon, MultiPolygon)):
                    densified.append(_densify_one(g))
                else:
                    densified.append(g)
            return GeometryCollection(densified)

        return geom

    gdf[geometry_col] = gdf[geometry_col].apply(_densify_one)
    return gdf


def geometries_to_points(gdf_or_path, geometry_col="geometry", z_col="z"):
    if isinstance(gdf_or_path, (str, Path)):
        gdf = gpd.read_file(gdf_or_path)
    else:
        gdf = gdf_or_path.copy()

    rows_out = []

    def _emit_points(base_attrs, linear_geom, parent_type, part_id=0, ring_id=0):
        coords = _coords_array_with_optional_z(linear_geom)
        for i, c in enumerate(coords):
            row = base_attrs.copy()
            row["parent_geom_type"] = parent_type
            row["part_id"] = part_id
            row["ring_id"] = ring_id
            row["vertex_id"] = i
            row[z_col] = None if np.isnan(c[2]) else float(c[2])
            row["geometry"] = Point(c[0], c[1])
            rows_out.append(row)

    for _, row in gdf.iterrows():
        geom = row[geometry_col]
        if geom is None or geom.is_empty:
            continue

        attrs = row.drop(labels=[geometry_col]).to_dict()

        if isinstance(geom, LineString):
            _emit_points(attrs, geom, "LineString")

        elif isinstance(geom, MultiLineString):
            for p_idx, part in enumerate(geom.geoms):
                _emit_points(attrs, part, "MultiLineString", part_id=p_idx)

        elif isinstance(geom, Polygon):
            _emit_points(attrs, geom.exterior, "Polygon")
            for r_idx, ring in enumerate(geom.interiors, start=1):
                _emit_points(attrs, ring, "Polygon", ring_id=r_idx)

        elif isinstance(geom, MultiPolygon):
            for p_idx, poly in enumerate(geom.geoms):
                _emit_points(attrs, poly.exterior, "MultiPolygon", part_id=p_idx)
                for r_idx, ring in enumerate(poly.interiors, start=1):
                    _emit_points(attrs, ring, "MultiPolygon", part_id=p_idx, ring_id=r_idx)

    out = gpd.GeoDataFrame(rows_out, geometry="geometry", crs=gdf.crs)
    out["geometry_x"] = out.geometry.x
    out["geometry_y"] = out.geometry.y
    try:
        out["geometry_z"] = out.geometry.apply(lambda p: p.z if hasattr(p, "z") else None)
    except Exception:
        out["geometry_z"] = 0
    return out


def bufferize_2d_polygons(
    geopandas_2d_polygons,
    feature_cols,
    level_col_or_z=0,
    increments=(0.1, 1, 10, 50),
    repeats=(2, 10, 10, 10),
    segment_max_length=5,
    return_polydata=True,
    verbose=True,
    max_extra_steps=100,
):
    if not hasattr(geopandas_2d_polygons, "geometry"):
        raise TypeError("geopandas_2d_polygons must be a GeoDataFrame with a geometry column.")

    if len(increments) != len(repeats):
        raise ValueError("increments and repeats must have the same length.")

    if segment_max_length <= 0:
        raise ValueError("segment_max_length must be > 0.")

    if isinstance(feature_cols, str):
        feature_cols = [feature_cols]
    elif feature_cols is None:
        feature_cols = []
    else:
        feature_cols = list(feature_cols)

    gdf = geopandas_2d_polygons.copy()

    missing_feature_cols = [c for c in feature_cols if c not in gdf.columns]
    if missing_feature_cols:
        raise KeyError(f"Missing feature columns: {missing_feature_cols}")

    increments_arr = np.asarray(increments, dtype=float)
    repeats_arr = np.asarray(repeats, dtype=int)
    if np.any(repeats_arr <= 0):
        raise ValueError("All repeats values must be positive integers.")

    configured_buffers = -np.abs(np.repeat(increments_arr, repeats_arr).cumsum())
    extension_step = float(np.abs(increments_arr[-1] * 2))

    buffered_layers = []
    step = 0

    while True:
        if step < len(configured_buffers):
            buffer_distance = float(configured_buffers[step])
        else:
            extra_step_index = step - len(configured_buffers) + 1
            if extra_step_index > max_extra_steps:
                break
            buffer_distance = float(configured_buffers[-1] - extension_step * extra_step_index)

        if verbose:
            print(f"Processing buffer {buffer_distance}")

        layer = gdf.copy()
        layer["geometry"] = layer.geometry.buffer(buffer_distance)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="GeoSeries.notna*", category=UserWarning)
            valid_mask = layer.geometry.notna() & (~layer.geometry.is_empty)
        layer = layer[valid_mask].copy()

        if layer.empty:
            break

        layer["_buffer_distance"] = buffer_distance
        buffered_layers.append(layer)
        step += 1

    zcol = level_col_or_z if isinstance(level_col_or_z, str) else "z"

    if not buffered_layers:
        empty = gpd.GeoDataFrame(columns=["x", "y", zcol, *feature_cols, "geometry"])
        empty = empty.set_geometry("geometry")
        empty.crs = gdf.crs
        if return_polydata:
            try:
                import pyvista as pv
            except ImportError as exc:
                raise ImportError("return_polydata=True requires pyvista.") from exc
            return empty, pv.PolyData(np.empty((0, 3)))
        return empty

    buffered_gdf = gpd.GeoDataFrame(
        pd.concat(buffered_layers, ignore_index=True),
        geometry="geometry",
        crs=gdf.crs,
    )

    densified = densify_geometries(buffered_gdf, spacing=segment_max_length, geometry_col="geometry")
    points_gdf = geometries_to_points(densified, geometry_col="geometry", z_col="z")

    outdf = pd.DataFrame(
        {
            "x": points_gdf["geometry_x"].to_numpy(),
            "y": points_gdf["geometry_y"].to_numpy(),
        }
    )

    if isinstance(level_col_or_z, str):
        if level_col_or_z not in points_gdf.columns:
            raise KeyError(f"Column '{level_col_or_z}' not found in input features.")
        outdf[zcol] = points_gdf[level_col_or_z].to_numpy()
    else:
        if np.isscalar(level_col_or_z):
            outdf[zcol] = float(level_col_or_z)
        else:
            level_arr = np.asarray(level_col_or_z).reshape(-1)
            if level_arr.size != len(outdf):
                raise ValueError("When level_col_or_z is array-like, its size must match output point count.")
            outdf[zcol] = level_arr

    for feat in feature_cols:
        outdf[feat] = points_gdf[feat].to_numpy()

    z_numeric = pd.to_numeric(outdf[zcol], errors="coerce")
    geometry = gpd.points_from_xy(outdf["x"], outdf["y"], z=z_numeric)
    out_gdf = gpd.GeoDataFrame(outdf.copy(), geometry=geometry, crs=gdf.crs)

    if not return_polydata:
        return out_gdf

    try:
        import pyvista as pv
    except ImportError as exc:
        raise ImportError("return_polydata=True requires pyvista.") from exc

    points_xyz = pd.DataFrame(
        {
            "x": out_gdf["x"].to_numpy(),
            "y": out_gdf["y"].to_numpy(),
            "z": z_numeric.to_numpy(),
        }
    )
    points = pv.wrap(points_xyz.to_numpy())

    def _to_ascii_text(value):
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return ""
        text = str(value)
        normalized = unicodedata.normalize("NFKD", text)
        return normalized.encode("ascii", "ignore").decode("ascii")

    for feat in feature_cols:
        if is_numeric_dtype(out_gdf[feat]):
            points.point_data[feat] = out_gdf[feat].to_numpy()
        else:
            safe_text = out_gdf[feat].map(_to_ascii_text).to_numpy(dtype="U")
            points.point_data[feat] = safe_text.astype("S")

    return out_gdf, points
