"""Base surface of a sedimentary basin from a polygon draped over a DEM (surfaces only, no solids).

The base surface Z_base(x, y) is continuous over the whole bounding box:

* inside the polygon (signed distance d >= 0):  Z_base = Z_top - z_max * (1 - exp(-d / L))
  (a floor that is 0 m deep on the edge and saturates towards z_max);
* outside (d < 0):  Z_base = Z_top + h(|d|),  h = h_min * smoothstep(|d| / w) + h_slope * |d|
  so the base lies strictly above the topography and meets it exactly on the ring.

Top and base share ONE constrained Delaunay triangulation of the bounding box (constraints: the polygon rings and the
box edges), graded from `edge_spacing` at the rings to coarser towards the box edge, so vertices correspond one to one.

Modules, in the order they are used: `prepare` (frame, validity, densify), `distance` (signed distance field),
`elevation` (Z_top, Z_base), `meshing` (graded triangulation), `qa` (checks, transects, volume), `build` (all of it).
Everything works in a local metric frame (origin subtracted) to keep float precision; the origin is returned.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------------------------------------
# parameters (all lengths in metres, slopes in m/m)
# ----------------------------------------------------------------------------------------------------------


@dataclass
class BasinParams:
    z_max: float = 60.0           # maximum basin depth
    L: float = 150.0              # saturation length of the depth profile
    edge_spacing: float = 40.0    # target triangle edge on the polygon ring
    h_slope: float = 0.20         # slope of the ramp outside the polygon
    h_min: float = 5.0            # minimum offset above the topography just outside
    ramp_width: float = 100.0     # w: distance over which h_min is reached
    smooth_cells: float = 3.0     # sigma (grid cells) of the Gaussian smoothing of d inside the polygon
    grid_cell: float = 20.0       # cell of the distance-field grid
    grow: float = 0.15            # mesh size grows by this per metre of distance from the ring
    max_edge: float = 1500.0      # largest triangle edge at the box boundary
    refine_passes: int = 3
    box_edge_spacing: float | None = None   # spacing of the points put on the box edge (default: max_edge)


# ----------------------------------------------------------------------------------------------------------
# 1. prepare: metric frame, valid polygon, densified ring
# ----------------------------------------------------------------------------------------------------------


@dataclass
class Frame:
    crs: str
    origin: tuple[float, float, float]   # x0, y0, z0 subtracted from every coordinate

    def to_json(self) -> str:
        return json.dumps({"crs": self.crs, "origin": list(self.origin)})


def prepare(polygons, crs, bbox, spacing, z0: float = 0.0):
    """Reproject to a metric CRS if needed, repair validity, merge, densify, shift to a local origin.

    `polygons`: shapely geometry or iterable of them, in `crs`. `bbox`: (xmin, xmax, ymin, ymax) in `crs` (already
    metric) or in the polygon CRS if that is geographic (then it is reprojected too).
    Returns (polygon_local, bbox_local, Frame).
    """
    import pyproj
    import shapely
    from shapely.ops import unary_union

    geoms = [polygons] if hasattr(polygons, "geom_type") else list(polygons)
    src = pyproj.CRS.from_user_input(crs)
    if src.is_geographic:
        raise ValueError("Give the polygons and the box in a metric CRS (UTM); geographic input is not reprojected here.")
    xmin, xmax, ymin, ymax = bbox
    box = shapely.box(xmin, ymin, xmax, ymax)
    union = shapely.buffer(unary_union([g if g.is_valid else shapely.make_valid(g) for g in geoms]), 0)
    inner = shapely.box(xmin + 2 * spacing, ymin + 2 * spacing, xmax - 2 * spacing, ymax - 2 * spacing)
    kept = union.intersection(inner)
    if kept.is_empty:
        raise ValueError("The polygon does not intersect the box.")
    if kept.area < union.area * (1 - 1e-9):
        print(f"warning: {100 * (1 - kept.area / union.area):.2f}% of the polygon area lies within two edge spacings of the "
              "box edge or outside it and was cut off; use a larger box")
    union = shapely.segmentize(kept, spacing)
    x0, y0 = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
    local = shapely.transform(union, lambda c: c - np.array([x0, y0]))
    bbox_local = (xmin - x0, xmax - x0, ymin - y0, ymax - y0)
    return local, bbox_local, Frame(crs=src.to_string(), origin=(x0, y0, z0))


# ----------------------------------------------------------------------------------------------------------
# 2. signed distance field (d > 0 inside, < 0 outside, 0 on the ring)
# ----------------------------------------------------------------------------------------------------------


class DistanceField:
    """Signed distance on a grid (for sampling everywhere) plus exact distances near the real edges (for the vertices).

    `cover` (optional): younger polygons lying over or beside the basin. Their outlines are NOT basin edges: the cells they
    occupy are unknown, and the field there is filled harmonically (Laplace equation) from the known inside (d > 0) and
    outside (d < 0) cells around them, so the basin edge continues smoothly under the cover instead of closing on it.
    """

    def __init__(self, polygon, bbox, params: BasinParams, cover=None):
        import shapely
        from rasterio import features
        from rasterio.transform import from_origin
        from scipy import ndimage

        self.polygon, self.bbox, self.p = polygon, bbox, params
        self.cover = cover if cover is not None and not cover.is_empty else None
        xmin, xmax, ymin, ymax = bbox
        c = params.grid_cell
        self.nx, self.ny = int(np.ceil((xmax - xmin) / c)) + 1, int(np.ceil((ymax - ymin) / c)) + 1
        self.x0, self.y0, self.c = xmin, ymin, c
        transform = from_origin(xmin - c / 2, ymin + (self.ny - 0.5) * c, c, c)   # row 0 = northmost

        def raster(geom):
            return features.rasterize([(geom, 1)], out_shape=(self.ny, self.nx), transform=transform, fill=0,
                                      dtype="uint8", all_touched=False).astype(bool)[::-1]   # row 0 = south

        if self.cover is None:
            inside = raster(polygon)
            d_in = ndimage.distance_transform_edt(inside) - 0.5
            d_out = ndimage.distance_transform_edt(~inside) - 0.5
            d = np.where(inside, d_in, -d_out) * c
            real = polygon.boundary
        else:
            unknown = raster(self.cover)
            inside = raster(polygon.difference(self.cover)) & ~unknown
            outside = ~inside & ~unknown
            d_in = ndimage.distance_transform_edt(~outside) - 0.5 if outside.any() else np.full(inside.shape, 1e6)
            d_out = ndimage.distance_transform_edt(~inside) - 0.5 if inside.any() else np.full(inside.shape, 1e6)
            d = np.where(inside, d_in, np.where(outside, -d_out, 0.0)) * c
            d = _harmonic_fill(d, unknown)
            real = polygon.boundary.difference(self.cover.buffer(1.5 * c))
        d_smooth = ndimage.gaussian_filter(np.where(d > 0, d, 0.0), params.smooth_cells)
        f = np.clip(d / (3 * c), 0.0, 1.0)                 # 0 at the edge: exact there; smooth from 3 cells in
        self.grid = np.where(d > 0, (1 - f) * d + f * d_smooth, d)
        self.real_edges = real
        segs = []
        for line in _lines(real):
            xy = np.asarray(line.coords)
            segs += [shapely.linestrings(xy[i:i + 2]) for i in range(len(xy) - 1)]
        self.tree = shapely.STRtree(segs) if segs else None
        self.near = 3.0 * c

    def sample_grid(self, x, y):
        from scipy import ndimage
        return ndimage.map_coordinates(self.grid, [(np.asarray(y) - self.y0) / self.c, (np.asarray(x) - self.x0) / self.c],
                                       order=1, mode="nearest")

    def at(self, x, y):
        """Signed distance at points: grid value, replaced by the exact distance where closer than 3 cells to a real edge."""
        import shapely
        x, y = np.atleast_1d(np.asarray(x, float)), np.atleast_1d(np.asarray(y, float))
        d = self.sample_grid(x, y)
        close = np.abs(d) < self.near
        if self.cover is not None and close.any():
            close &= ~shapely.contains_xy(self.cover, x, y)
        if close.any() and self.tree is not None:
            pts = shapely.points(x[close], y[close])
            idx, dist = self.tree.query_nearest(pts, return_distance=True, all_matches=False)
            exact = np.empty(len(pts))
            exact[idx[0]] = dist
            inside = shapely.contains_xy(self.polygon, x[close], y[close])
            exact = np.where(inside, exact, -exact)
            exact = np.where(np.abs(exact) < 1e-6, 0.0, exact)
            f = np.clip(exact / self.near, 0.0, 1.0)
            smooth = d[close]
            d[close] = np.where(exact > 0, (1 - f) * exact + f * smooth, exact)
        return d


def _harmonic_fill(d, unknown):
    """Fill the `unknown` cells of d by solving Laplace's equation with the known neighbours as boundary values."""
    from scipy import sparse
    from scipy.sparse.linalg import spsolve

    if not unknown.any():
        return d
    ny, nx = d.shape
    n = int(unknown.sum())
    idx = -np.ones(d.shape, int)
    idx[unknown] = np.arange(n)
    rows, cols, vals, rhs = [], [], [], np.zeros(n)
    ys, xs = np.nonzero(unknown)
    diag = np.zeros(n)
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        y2, x2 = ys + dy, xs + dx
        ok = (y2 >= 0) & (y2 < ny) & (x2 >= 0) & (x2 < nx)
        i = idx[ys[ok], xs[ok]]
        yy, xx = y2[ok], x2[ok]
        j_unknown = unknown[yy, xx]
        np.add.at(diag, i, 1.0)
        rows += list(i[j_unknown]); cols += list(idx[yy[j_unknown], xx[j_unknown]]); vals += [-1.0] * int(j_unknown.sum())
        np.add.at(rhs, i[~j_unknown], d[yy[~j_unknown], xx[~j_unknown]])
    rows += list(range(n)); cols += list(range(n)); vals += list(diag + 1e-9)
    a = sparse.csc_matrix((vals, (rows, cols)), shape=(n, n))
    out = d.copy()
    out[unknown] = spsolve(a, rhs)
    return out


def _lines(geom):
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "LineString":
        return [geom]
    if geom.geom_type in ("MultiLineString", "GeometryCollection"):
        return [g for part in geom.geoms for g in _lines(part)]
    return []


def _rings(polygon):
    geoms = polygon.geoms if hasattr(polygon, "geoms") else [polygon]
    for g in geoms:
        yield g.exterior
        yield from g.interiors


# ----------------------------------------------------------------------------------------------------------
# 3. elevation functions
# ----------------------------------------------------------------------------------------------------------


def smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


class Dem:
    """Bilinear DEM sampler in the local frame (extends the edge value outside the raster, and counts how often)."""

    def __init__(self, path, frame: Frame):
        import rasterio

        with rasterio.open(path) as src:
            self.z = src.read(1).astype(float)
            self.transform, self.nodata = src.transform, src.nodata
            self.crs = src.crs
        if self.nodata is not None:
            self.z[self.z == self.nodata] = np.nan
        self.frame, self.outside = frame, 0
        if np.isnan(self.z).any():
            from scipy import ndimage
            idx = ndimage.distance_transform_edt(np.isnan(self.z), return_distances=False, return_indices=True)
            self.z = self.z[tuple(idx)]
            self.filled = True
        else:
            self.filled = False

    def __call__(self, x, y):
        from scipy import ndimage

        x0, y0, z0 = self.frame.origin
        inv = ~self.transform
        xw, yw = np.asarray(x) + x0, np.asarray(y) + y0
        col, row = inv.a * xw + inv.b * yw + inv.c, inv.d * xw + inv.e * yw + inv.f
        col, row = col - 0.5, row - 0.5                       # pixel centres
        h, w = self.z.shape
        self.outside += int(((col < 0) | (col > w - 1) | (row < 0) | (row > h - 1)).sum())
        return ndimage.map_coordinates(self.z, [row, col], order=1, mode="nearest") - z0


def z_base(z_top, d, p: BasinParams):
    """Base elevation from the top elevation and the signed distance (vectorised)."""
    d = np.asarray(d, float)
    inside = d >= 0
    depth = p.z_max * (1.0 - np.exp(-np.where(inside, d, 0.0) / p.L))
    ahead = np.abs(np.where(inside, 0.0, d))
    h = p.h_min * smoothstep(ahead / p.ramp_width) + p.h_slope * ahead
    return np.where(inside, z_top - depth, z_top + h)


# ----------------------------------------------------------------------------------------------------------
# 4. meshing: one graded constrained Delaunay triangulation of the box
# ----------------------------------------------------------------------------------------------------------


def _pslg(polygon, bbox, box_spacing, real_edges=None):
    """Planar straight-line graph: the polygon rings (or, with a cover, only the real edges) and the box edges."""
    xmin, xmax, ymin, ymax = bbox
    verts, segs, seen = [], [], {}

    def vid(pt):
        key = (round(pt[0], 6), round(pt[1], 6))
        if key not in seen:
            seen[key] = len(verts)
            verts.append(tuple(pt))
        return seen[key]

    lines = list(_rings(polygon)) if real_edges is None else _lines(real_edges)
    for line in lines:
        xy = np.asarray(line.coords)
        ids = [vid(q) for q in xy]
        segs.extend((a, b) for a, b in zip(ids[:-1], ids[1:]) if a != b)
    nx = max(1, int(np.ceil((xmax - xmin) / box_spacing)))
    ny = max(1, int(np.ceil((ymax - ymin) / box_spacing)))
    pts = ([(x, ymin) for x in np.linspace(xmin, xmax, nx + 1)[:-1]] + [(xmax, y) for y in np.linspace(ymin, ymax, ny + 1)[:-1]]
           + [(x, ymax) for x in np.linspace(xmax, xmin, nx + 1)[:-1]] + [(xmin, y) for y in np.linspace(ymax, ymin, ny + 1)[:-1]])
    ids = [vid(q) for q in pts]
    segs.extend((ids[i], ids[(i + 1) % len(ids)]) for i in range(len(ids)))
    return np.asarray(verts, float), np.asarray(segs, int)


def mesh(polygon, bbox, field: DistanceField, p: BasinParams):
    """Graded constrained Delaunay triangulation (triangle 'pq30', refined with a size field of |d|)."""
    import triangle as tr

    v, s = _pslg(polygon, bbox, p.box_edge_spacing or p.max_edge,
                 real_edges=field.real_edges if field.cover is not None else None)
    t = tr.triangulate({"vertices": v, "segments": s}, "pq30")
    for _ in range(p.refine_passes):
        tri = t["triangles"]
        cen = t["vertices"][tri].mean(axis=1)
        d = np.abs(field.sample_grid(cen[:, 0], cen[:, 1]))
        h = np.clip(p.edge_spacing + p.grow * d, p.edge_spacing, p.max_edge)
        t["triangle_max_area"] = (np.sqrt(3.0) / 4.0) * h * h
        t = tr.triangulate(t, "rpq30a")
    return t["vertices"], t["triangles"]


# ----------------------------------------------------------------------------------------------------------
# 5. QA
# ----------------------------------------------------------------------------------------------------------


def triangle_quality(v, f):
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    area2 = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    e = np.stack([np.linalg.norm(b - c, axis=1), np.linalg.norm(c - a, axis=1), np.linalg.norm(a - b, axis=1)], axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        cosA = np.stack([(e[:, 1] ** 2 + e[:, 2] ** 2 - e[:, 0] ** 2) / (2 * e[:, 1] * e[:, 2]),
                         (e[:, 0] ** 2 + e[:, 2] ** 2 - e[:, 1] ** 2) / (2 * e[:, 0] * e[:, 2]),
                         (e[:, 0] ** 2 + e[:, 1] ** 2 - e[:, 2] ** 2) / (2 * e[:, 0] * e[:, 1])], axis=1)
    ang = np.degrees(np.arccos(np.clip(cosA, -1, 1)))
    aspect = e.max(axis=1) ** 2 / np.maximum(np.abs(area2), 1e-12)    # longest edge / height on that edge
    return area2 / 2.0, ang.min(axis=1), aspect


def qa(v, f, z_t, z_b, d, polygon, field, p: BasinParams, out_dir: Path | None = None, name="basin", verbose=True,
       z_offset: float = 0.0):
    """Checks required of the surfaces. Returns a dict of numbers and booleans; prints them and saves transects."""
    import shapely

    area, min_ang, aspect = triangle_quality(v, f)
    ring = np.abs(d) < 1e-6
    depth = z_t - z_b
    res = {
        "n_vertices": int(len(v)), "n_triangles": int(len(f)),
        "degenerate_triangles": int((np.abs(area) < 1e-9).sum()),
        "flipped_triangles": int((area < 0).sum()),
        "min_angle_deg": {"min": float(min_ang.min()), "p1": float(np.percentile(min_ang, 1)), "median": float(np.median(min_ang))},
        "aspect_ratio": {"max": float(aspect.max()), "p99": float(np.percentile(aspect, 99)), "median": float(np.median(aspect))},
        "ring_vertices": int(ring.sum()),
        "ring_max_abs_dz": float(np.abs(depth[ring]).max()) if ring.any() else None,
        "inside_base_below_top": bool((depth[d > 1e-6] > 0).all()) if (d > 1e-6).any() else True,
        "outside_base_above_top": bool((depth[d < -1e-6] < 0).all()) if (d < -1e-6).any() else True,
    }
    cen = v[f].mean(axis=1)
    inside = ((field.sample_grid(cen[:, 0], cen[:, 1]) > 0) if field.cover is not None
              else shapely.contains_xy(polygon, cen[:, 0], cen[:, 1]))
    mean_depth = depth[f].mean(axis=1)
    res["basin_volume_m3"] = float((area[inside] * mean_depth[inside]).sum())
    res["polygon_area_m2"] = float(area[inside].sum())
    res["mean_depth_m"] = res["basin_volume_m3"] / res["polygon_area_m2"] if res["polygon_area_m2"] else 0.0
    if verbose:
        print(json.dumps(res, indent=2))
    if out_dir is not None:
        transects(v, f, z_t, z_b, polygon, field, p, Path(out_dir) / f"{name}_transects.png", z_offset=z_offset)
    return res


def transects(v, f, z_t, z_b, polygon, field, p: BasinParams, path: Path, n=3, z_offset: float = 0.0):
    """Three transects through the polygon: mesh Z_top and Z_base against the analytic base z_top - z_max*(1-exp(-d/L))."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    import shapely

    interp_t = mtri.LinearTriInterpolator(mtri.Triangulation(v[:, 0], v[:, 1], f), z_t)
    interp_b = mtri.LinearTriInterpolator(mtri.Triangulation(v[:, 0], v[:, 1], f), z_b)
    big = max(polygon.geoms, key=lambda g: g.area) if hasattr(polygon, "geoms") else polygon
    cx, cy = big.centroid.x, big.centroid.y
    half = min(float(v[:, 0].max() - v[:, 0].min()), float(v[:, 1].max() - v[:, 1].min())) / 2.0
    fig, axes = plt.subplots(n, 1, figsize=(10, 3.2 * n), sharey=False)
    for ax, ang in zip(np.atleast_1d(axes), np.linspace(0, 180, n, endpoint=False)):
        s = np.linspace(-half, half, 1200)
        x, y = cx + s * np.cos(np.radians(ang)), cy + s * np.sin(np.radians(ang))
        ok = (x >= v[:, 0].min()) & (x <= v[:, 0].max()) & (y >= v[:, 1].min()) & (y <= v[:, 1].max())
        s, x, y = s[ok], x[ok], y[ok]
        zt, zb = np.ma.filled(interp_t(x, y), np.nan), np.ma.filled(interp_b(x, y), np.nan)
        d = field.at(x, y)
        analytic = z_base(zt, d, p) + z_offset
        zt, zb = zt + z_offset, zb + z_offset
        ax.plot(s, zt, color="#7a6a4f", lw=1.2, label="Z_top (mesh)")
        ax.plot(s, zb, color="#1f5aa6", lw=1.4, label="Z_base (mesh)")
        ax.plot(s, analytic, "--", color="#d1495b", lw=1.0, label="Z_base analytic: z_top - z_max(1-exp(-d/L))")
        ax.fill_between(s, zb, zt, where=(d > 0), color="#1f5aa6", alpha=0.15)
        ax.set_title(f"transect at {ang:.0f} deg through the largest polygon", fontsize=9)
        ax.set_xlabel("distance along transect (m)"); ax.set_ylabel("elevation (m a.s.l.)")
        ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ----------------------------------------------------------------------------------------------------------
# 6. everything
# ----------------------------------------------------------------------------------------------------------


@dataclass
class BasinResult:
    vertices: np.ndarray        # local frame, top surface (x, y, z_top)
    faces: np.ndarray
    z_top: np.ndarray
    z_base: np.ndarray
    d: np.ndarray
    frame: Frame
    params: BasinParams
    qa: dict = field(default_factory=dict)
    dist_field: object = None

    def surface(self, which: str, real_coords: bool = False) -> np.ndarray:
        z = self.z_top if which == "top" else self.z_base
        xyz = np.column_stack([self.vertices[:, 0], self.vertices[:, 1], z])
        if real_coords:
            xyz = xyz + np.asarray(self.frame.origin)
        return xyz


def _contour_polygons(x, y, z, level=0.0):
    """Polygons of the region z > level, edges placed by linear interpolation between grid nodes (marching squares)."""
    import contourpy
    import shapely
    from contourpy import FillType

    gen = contourpy.contour_generator(x, y, z, fill_type=FillType.ChunkCombinedOffsetOffset)
    points, offsets, outer = gen.filled(level, np.inf)
    polys = []
    for pts, off, out in zip(points, offsets, outer):
        if pts is None:
            continue
        for k in range(len(out) - 1):
            rings = [pts[off[i]:off[i + 1]] for i in range(out[k], out[k + 1])]
            polys.append(shapely.Polygon(rings[0], rings[1:]))
    return polys


def filled_outline(result: BasinResult, min_area: float = 0.0, simplify: float = 0.1):
    """Outline of the basin as the surface sees it, in real coordinates: the original polygon plus the parts under the cover.

    Where the edge is a real map edge the outline IS the original polygon edge (exact vertices). Only under the cover polygons,
    where the surface continues the basin by the harmonic fill, is the edge taken from the zero contour of the distance grid,
    interpolated between nodes (marching squares), so no raster staircase. `simplify` is a tolerance in grid cells.
    Returns a shapely (Multi)Polygon.
    """
    import shapely
    from shapely.ops import unary_union

    fld = result.dist_field
    c = fld.c
    x0, y0, _ = result.frame.origin
    poly = fld.polygon if fld.polygon.is_valid else shapely.make_valid(fld.polygon)
    if fld.cover is None:
        out = poly
    else:
        xs = fld.x0 + c * np.arange(fld.nx)
        ys = fld.y0 + c * np.arange(fld.ny)
        under = unary_union(_contour_polygons(xs, ys, fld.grid)).intersection(fld.cover)
        under = under.simplify(simplify * c, preserve_topology=True)
        # the grid edge and the exact cover edge differ by up to a cell: close the thin seams, but only inside the cover,
        # so the real map edges outside it stay exactly as drawn
        joined = unary_union([poly, under]).buffer(0)
        closed = shapely.buffer(shapely.buffer(joined, c, join_style="round"), -c, join_style="round")
        out = unary_union([poly, closed.intersection(fld.cover)]).buffer(0)
        holes = [shapely.Polygon(h) for g in getattr(out, "geoms", [out]) for h in g.interiors]
        holes = [h for h in holes if h.area < (3 * c) ** 2 or (h.intersects(fld.cover) and h.buffer(-c).is_empty)]
        out = unary_union([out] + holes).buffer(0)      # specks and hairline slits left by the seams
    out = unary_union([g for g in getattr(out, "geoms", [out]) if g.area >= min_area])
    out = shapely.transform(out, lambda xy: xy + np.array([x0, y0]))
    return shapely.buffer(out, 0)


def write_mesh(path, xyz, faces):
    """ASCII PLY or OBJ (by extension). Coordinates with 4 decimals; faces counter-clockwise seen from above."""
    path = Path(path)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        if path.suffix.lower() == ".ply":
            fh.write(f"ply\nformat ascii 1.0\nelement vertex {len(xyz)}\nproperty double x\nproperty double y\nproperty double z\n"
                     f"element face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n")
            np.savetxt(fh, xyz, fmt="%.4f")
            np.savetxt(fh, np.column_stack([np.full(len(faces), 3), faces]), fmt="%d")
        else:
            np.savetxt(fh, xyz, fmt="v %.4f %.4f %.4f")
            np.savetxt(fh, faces + 1, fmt="f %d %d %d")


def _local(geom, frame):
    import shapely

    x0, y0, _ = frame.origin
    g = shapely.buffer(geom if geom.is_valid else shapely.make_valid(geom), 0)
    return shapely.transform(g, lambda c: c - np.array([x0, y0]))


def build_basin_surface(polygons, crs, dem_path, bbox, params: BasinParams | None = None, out_dir=None, name="basin",
                        verbose=True, cover=None, younger=None, gap: float = 1.0) -> BasinResult:
    """Polygon + DEM + box -> top and base surfaces on one triangulation. See the module docstring.

    cover:   younger polygons (shapely, same CRS) lying over/beside this basin; their outlines are not basin edges and the
             distance field is filled across them (see DistanceField).
    younger: BasinResult of the younger basin. Where this basin is deeper than 0, its base is kept at least `gap` metres
             below the younger base, so the two surfaces never touch (no coincident points for the CSG).
    """
    p = params or BasinParams()
    out = Path(out_dir) if out_dir else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
    local, bbox_l, frame = prepare(polygons, crs, bbox, p.edge_spacing)
    # local z origin: the mean elevation inside the box, so heights are small too
    dem0 = Dem(dem_path, Frame(frame.crs, frame.origin))
    xs, ys = np.meshgrid(np.linspace(bbox_l[0], bbox_l[1], 25), np.linspace(bbox_l[2], bbox_l[3], 25))
    z0 = float(np.round(np.mean(dem0(xs.ravel(), ys.ravel())), 0))
    frame = Frame(frame.crs, (frame.origin[0], frame.origin[1], z0))
    dem = Dem(dem_path, frame)
    cover_l = None
    if cover is not None:
        from shapely.ops import unary_union
        geoms = [cover] if hasattr(cover, "geom_type") else list(cover)
        cover_l = _local(unary_union(geoms), frame)
    field = DistanceField(local, bbox_l, p, cover=cover_l)
    v, f = mesh(local, bbox_l, field, p)
    d = field.at(v[:, 0], v[:, 1])
    zt = dem(v[:, 0], v[:, 1])
    zb = z_base(zt, d, p)
    if younger is not None:
        import matplotlib.tri as mtri
        shift = np.asarray(younger.frame.origin[:2]) - np.asarray(frame.origin[:2])        # younger local -> this local
        tri = mtri.Triangulation(younger.vertices[:, 0] + shift[0], younger.vertices[:, 1] + shift[1], younger.faces)
        yb = np.ma.filled(mtri.LinearTriInterpolator(tri, younger.z_base)(v[:, 0], v[:, 1]), np.nan)
        yb = yb + (younger.frame.origin[2] - frame.origin[2])
        ok = np.isfinite(yb) & (d > 0)
        ramp = smoothstep(d / (2.0 * p.edge_spacing))
        zb = np.where(ok, np.minimum(zb, yb - gap * ramp), zb)
    res = BasinResult(vertices=v, faces=f, z_top=zt, z_base=zb, d=d, frame=frame, params=p)
    if dem.outside and verbose:
        print(f"note: {dem.outside} vertex lookups fell outside the DEM (edge values were used)")
    res.dist_field = field
    res.qa = qa(v, f, zt, zb, d, local, field, p, out_dir=out, name=name, verbose=verbose, z_offset=frame.origin[2])
    if out:
        for which in ("top", "base"):
            write_mesh(out / f"{name}_{which}_local.ply", res.surface(which), f)
            write_mesh(out / f"{name}_{which}.obj", res.surface(which, real_coords=True), f)
        (out / f"{name}_frame.json").write_text(res.frame.to_json(), encoding="utf-8")
        (out / f"{name}_params_qa.json").write_text(json.dumps({"params": asdict(p), "qa": res.qa}, indent=2), encoding="utf-8")
    return res
