from __future__ import annotations

import unicodedata
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from pandas.api.types import is_numeric_dtype
from shapely import make_valid
from shapely.geometry import (
    GeometryCollection,
    LineString,
    LinearRing,
    MultiLineString,
    MultiPolygon,
    Point,
    Polygon,
)
from shapely.geometry.polygon import orient as orient_polygon


def _extract_polygonal_geometry(geom):
    if geom is None or geom.is_empty:
        return geom

    if isinstance(geom, (Polygon, MultiPolygon)):
        return geom

    if isinstance(geom, GeometryCollection):
        polygons = []
        for part in geom.geoms:
            polygonal = _extract_polygonal_geometry(part)
            if polygonal is None or polygonal.is_empty:
                continue
            if isinstance(polygonal, Polygon):
                polygons.append(polygonal)
            elif isinstance(polygonal, MultiPolygon):
                polygons.extend(list(polygonal.geoms))

        if not polygons:
            return None
        if len(polygons) == 1:
            return polygons[0]
        return MultiPolygon(polygons)

    return None


def _orient_polygonal_geometry(geom):
    if geom is None or geom.is_empty:
        return geom

    if isinstance(geom, Polygon):
        return orient_polygon(geom, sign=1.0)

    if isinstance(geom, MultiPolygon):
        return MultiPolygon([orient_polygon(poly, sign=1.0) for poly in geom.geoms])

    return geom


def clean_polygon_geometries(gdf_or_path, geometry_col="geometry", verbose=True, drop_empty=True):
    if isinstance(gdf_or_path, (str, Path)):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*invalid winding order.*|.*organizePolygons\(\).*|.*cannot be translated to Simple Geometry.*",
                category=RuntimeWarning,
            )
            gdf = gpd.read_file(gdf_or_path)
    else:
        gdf = gdf_or_path.copy()

    gdf = gdf.copy()
    invalid_before = int((~gdf[geometry_col].is_valid).fillna(False).sum())

    if verbose:
        print(
            "Pre-cleaning polygon geometries: fixing invalid shapes, keeping polygonal parts, and normalizing ring orientation."
        )

    def _clean_one(geom):
        if geom is None or geom.is_empty:
            return geom
        cleaned = make_valid(geom)
        cleaned = _extract_polygonal_geometry(cleaned)
        cleaned = _orient_polygonal_geometry(cleaned)
        return cleaned

    gdf[geometry_col] = gdf[geometry_col].apply(_clean_one)

    dropped = 0
    if drop_empty:
        valid_mask = gdf[geometry_col].notna() & (~gdf[geometry_col].is_empty)
        dropped = int((~valid_mask).sum())
        gdf = gdf.loc[valid_mask].copy()

    invalid_after = int((~gdf[geometry_col].is_valid).fillna(False).sum())

    if verbose:
        print(
            f"Pre-clean complete: invalid geometries {invalid_before} -> {invalid_after}; dropped {dropped} empty/non-polygon features."
        )

    return gdf


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

        if isinstance(geom, Point):
            _emit_points(attrs, geom, "Point")

        elif isinstance(geom, LineString):
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


def _mm_on_ground(map_scale):
    """Ground length (CRS units, assumed metres) of 1 mm on a map at `map_scale`."""
    return float(map_scale) / 1000.0


def _centreline(poly, spacing, min_frac=0.3):
    """Approximate medial axis of a polygon, resampled every `spacing`, or None.

    Built from the Voronoi edges of the densified boundary that lie strictly
    inside the polygon. Spurs running out to boundary vertices are pruned by
    dropping edges closer to the boundary than `min_frac` times the mean
    half-width (area / perimeter), and leftover fragments shorter than
    ``spacing / 2`` are dropped (the longest piece is always kept).
    """
    half_width = poly.area / poly.length
    # Sample the boundary finer than the polygon is wide, but cap the sample
    # count (long polygons) and floor the spacing at 1 unit (slivers).
    sample = max(half_width / 2.0, poly.length / 20000.0, 1.0)
    boundary_pts = shapely.MultiPoint(shapely.get_coordinates(shapely.segmentize(poly, sample)))
    edges = shapely.get_parts(shapely.voronoi_polygons(boundary_pts, only_edges=True))
    edges = edges[shapely.contains_properly(poly, edges)]
    if len(edges) == 0:
        return None
    mid = shapely.line_interpolate_point(edges, 0.5, normalized=True)
    edges = edges[shapely.distance(mid, poly.boundary) >= min_frac * half_width]
    if len(edges) == 0:
        return None
    pieces = shapely.get_parts(shapely.line_merge(MultiLineString(list(edges))))
    lengths = shapely.length(pieces)
    keep = (lengths >= spacing / 2.0) | (lengths == lengths.max())
    lines = []
    for line, length in zip(pieces[keep], lengths[keep]):
        n = max(int(np.ceil(length / spacing)), 1)
        lines.append(LineString(shapely.line_interpolate_point(line, np.linspace(0.0, length, n + 1))))
    return MultiLineString(lines)


def _adaptive_layers(gdf, configured_buffers, narrow_half_width, min_half_width, ring_fraction, spacing):
    """Extra geometries for polygons too narrow for the configured buffers.

    Returns a GeoDataFrame of polygons (extra inward rings), lines
    (centrelines) and points (fallback), carrying the input attributes and a
    `_source` column.
    """
    parts = gdf.explode(ignore_index=True)
    second = configured_buffers[1] if len(configured_buffers) > 1 else configured_buffers[0]
    rows = []
    for _, row in parts.iterrows():
        poly = row.geometry
        if poly is None or poly.is_empty or poly.area <= 0:
            continue
        half_width = poly.area / poly.length
        few_rings = poly.buffer(second).is_empty  # configured buffers give <= 1 ring
        produced = not poly.buffer(configured_buffers[0]).is_empty

        if few_rings and half_width >= min_half_width:
            ring = poly.buffer(-ring_fraction * half_width)
            if not ring.is_empty:
                rows.append({**row.to_dict(), "geometry": ring, "_source": "adaptive_ring"})
                produced = True

        if half_width < narrow_half_width:
            line = _centreline(poly, spacing)
            if line is not None:
                rows.append({**row.to_dict(), "geometry": line, "_source": "centreline"})
                produced = True

        if not produced:
            rows.append({**row.to_dict(), "geometry": poly.point_on_surface(), "_source": "fallback_point"})

    if not rows:
        return None
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=gdf.crs)


def bufferize_2d_polygons(
    geopandas_2d_polygons,
    feature_cols,
    level_col_or_z=0,
    increments=None,
    repeats=None,
    segment_max_length=None,
    return_polydata=True,
    verbose=True,
    max_extra_steps=100,
    map_scale=None,
    adaptive=False,
    simplify_tolerance=None,
    narrow_half_width=None,
    min_half_width=None,
    ring_fraction=0.4,
):
    """
    Turn 2D polygons into labelled points on a series of inward buffers.

    Each polygon is shrunk by increasing distances; every buffer outline is
    densified and converted to points that carry the polygon's attributes.
    Buffering continues past the configured steps (every 2x the last
    increment) until each polygon has shrunk away.

    Parameters
    ----------
    geopandas_2d_polygons : geopandas.GeoDataFrame
        Polygons in a projected CRS with metre units.
    feature_cols : str or list of str
        Columns copied onto the output points (e.g. the unit code).
    level_col_or_z : str, float or array-like, optional
        Z for the points: a column name, a constant (default 0), or an array
        matching the output point count.
    increments, repeats : sequence of float / int, optional
        Buffer step sizes and how many times each is repeated. The cumulative
        sum gives the buffer distances, e.g. (50, 200) with (2, 5) gives
        -50, -100, -300, ..., -1100 m. Pass these to match the interior
        point spacing to your model resolution.
    segment_max_length : float, optional
        Maximum point spacing along each buffer outline.
    return_polydata : bool, optional
        If True (default) also return a PyVista point cloud.
    verbose : bool, optional
        Print progress.
    max_extra_steps : int, optional
        Maximum buffers added after the configured ones.
    map_scale : float, optional
        Scale denominator of the source map (e.g. 100_000 for 1:100,000).
        When given, every parameter left as None is derived from it using the
        map's drawing precision; see the table below. Explicit values always
        win.
    adaptive : bool, optional
        If True, polygons too narrow for the configured buffers also get
        points sized to their own width (default False):

        - a centreline (approximate medial axis) when the mean half-width
          ``area / perimeter`` is below `narrow_half_width`; it stays
          continuous through necks where inward buffers break apart;
        - one extra ring at ``ring_fraction * half-width`` when the
          configured buffers give at most one ring and the half-width is at
          least `min_half_width` (thinner polygons are effectively lines at
          map scale, so they only get the centreline);
        - a single interior point as a last resort, so no polygon is left
          without points.

        Adds a ``point_source`` column: ``ring``, ``adaptive_ring``,
        ``centreline`` or ``fallback_point``.
    simplify_tolerance : float, optional
        Douglas-Peucker tolerance applied to the input polygons first
        (topology preserving; polygons that collapse are dropped).
    narrow_half_width, min_half_width : float, optional
        Adaptive-mode thresholds on the mean half-width (see `adaptive`).
    ring_fraction : float, optional
        Adaptive-ring distance as a fraction of the mean half-width (0.4).

    Defaults
    --------
    With ``m = map_scale / 1000`` (1 mm on the map, in metres on the ground;
    contacts are drawn to about 0.5 mm):

    ====================  ==================  ===========  ===============
    Parameter             From `map_scale`    1:100,000    No `map_scale`
    ====================  ==================  ===========  ===============
    simplify_tolerance    0.25 m              25           0 (off)
    segment_max_length    1 m                 100          5
    increments            (0.5 m, 2 m)        (50, 200)    (0.1, 1, 10, 50)
    repeats               (2, 5)              (2, 5)       (2, 10, 10, 10)
    narrow_half_width     1 m                 100          2 x increments[0]
    min_half_width        0.25 m              25           increments[0] / 2
    ====================  ==================  ===========  ===============

    Returns
    -------
    geopandas.GeoDataFrame, or (GeoDataFrame, pyvista.PolyData) if
    `return_polydata` is True.
    """
    if not hasattr(geopandas_2d_polygons, "geometry"):
        raise TypeError("geopandas_2d_polygons must be a GeoDataFrame with a geometry column.")

    if map_scale is not None:
        if map_scale <= 0:
            raise ValueError("map_scale must be > 0.")
        m = _mm_on_ground(map_scale)
        scale_defaults = dict(
            simplify_tolerance=0.25 * m,
            segment_max_length=1.0 * m,
            increments=(0.5 * m, 2.0 * m),
            repeats=(2, 5),
        )
    else:
        scale_defaults = dict(
            simplify_tolerance=0.0,
            segment_max_length=5,
            increments=(0.1, 1, 10, 50),
            repeats=(2, 10, 10, 10),
        )
    if (increments is None) != (repeats is None):
        raise ValueError("Pass increments and repeats together (or neither).")
    if increments is None:
        increments, repeats = scale_defaults["increments"], scale_defaults["repeats"]
    if segment_max_length is None:
        segment_max_length = scale_defaults["segment_max_length"]
    if simplify_tolerance is None:
        simplify_tolerance = scale_defaults["simplify_tolerance"]
    first_step = float(abs(increments[0]))
    if narrow_half_width is None:
        narrow_half_width = 1.0 * _mm_on_ground(map_scale) if map_scale is not None else 2.0 * first_step
    if min_half_width is None:
        min_half_width = 0.25 * _mm_on_ground(map_scale) if map_scale is not None else first_step / 2.0

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

    if simplify_tolerance > 0:
        gdf["geometry"] = gdf.geometry.simplify(simplify_tolerance, preserve_topology=True)
        keep = gdf.geometry.notna() & ~gdf.geometry.is_empty & (gdf.geometry.area > 0)
        if verbose and (~keep).any():
            print(f"Simplify ({simplify_tolerance:g}): dropped {int((~keep).sum())} collapsed polygons.")
        gdf = gdf[keep].copy()

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

    for layer in buffered_layers:
        layer["_source"] = "ring"

    if adaptive:
        extra = _adaptive_layers(
            gdf,
            configured_buffers,
            narrow_half_width=narrow_half_width,
            min_half_width=min_half_width,
            ring_fraction=ring_fraction,
            spacing=segment_max_length,
        )
        if extra is not None:
            if verbose:
                print(f"Adaptive: {extra['_source'].value_counts().to_dict()}")
            buffered_layers.append(extra)

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

    if adaptive:
        outdf["point_source"] = points_gdf["_source"].to_numpy()

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
