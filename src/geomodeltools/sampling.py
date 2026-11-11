"""Points spread inside polygons, no two closer than a minimum distance (e.g. candidate collars on drill pads).

Plain arrays in, a DataFrame out: no file reading here. Needs shapely and scipy (``pip install geomodeltools[all]``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _area(xy):
    x, y = xy[:, 0], xy[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _perimeter(xy):
    return np.linalg.norm(np.roll(xy, -1, axis=0) - xy, axis=1).sum()


def _candidates(poly, xy, d_min, rng, max_grid):
    """Lightly shuffled grid inside the polygon, fine enough that thin strips still get points.

    The step follows the polygon's width (about 2 * area / perimeter for a strip), not just d_min, so a
    6 m wide strip is sampled every ~1.5 m instead of being missed.
    """
    import shapely

    a = _area(xy)
    width = 2 * a / _perimeter(xy)
    (x0, y0), (x1, y1) = xy.min(0), xy.max(0)
    h = min(d_min / 4, width / 4)
    h = max(h, np.sqrt((x1 - x0) * (y1 - y0) / max_grid))     # cap the grid size on huge boxes
    gx, gy = np.meshgrid(np.arange(x0 + h / 2, x1, h), np.arange(y0 + h / 2, y1, h))
    c = np.column_stack([gx.ravel(), gy.ravel()])
    c += rng.uniform(-h / 2, h / 2, c.shape)                   # no artificial lattice
    c = c[shapely.contains_xy(poly, c[:, 0], c[:, 1])]
    if len(c) == 0:                                            # grid stepped over it: one interior point
        p = shapely.point_on_surface(poly)
        c = np.array([[p.x, p.y]])
    return c


def _as_polygon(item):
    """(vertices, extra fields) from an (n,2)/(n,3) array or a dict with 'xy' (or 'vertices') plus extras."""
    if isinstance(item, dict):
        extra = dict(item)
        key = "xy" if "xy" in extra else "vertices"
        if key not in extra:
            raise ValueError("a polygon given as a dict needs its vertices under 'xy' or 'vertices'")
        v = extra.pop(key)
    else:
        v, extra = item, {}
    v = np.asarray(v, dtype=float)
    if v.ndim != 2 or v.shape[1] not in (2, 3):
        raise ValueError(f"polygon vertices must be an (n, 2) or (n, 3) array, got shape {v.shape}")
    if "z" not in extra:
        extra["z"] = float(v[:, 2].mean()) if v.shape[1] == 3 else np.nan
    return v[:, :2], extra


def points_in_polygons(polygons, d_min, *, margin=0.0, seed=0, max_grid=2_000_000) -> pd.DataFrame:
    """Points inside the polygons, no two closer than ``d_min`` (across all polygons).

    ``polygons``: a list of (n, 2) or (n, 3) vertex arrays, or dicts holding the vertices under ``"xy"`` (or
    ``"vertices"``) plus any extra fields (``z``, ``layer``, ...), which are copied onto each of that polygon's
    points. Without a ``z`` field, z is the mean vertex z of an (n, 3) polygon and NaN for an (n, 2) one.

    ``margin``: wanted distance from each point to its polygon's edge. A polygon narrower than 2 x margin still
    gets one point, at its most interior spot, flagged ``margin_ok = False``.

    Candidates are a lightly shuffled grid (step from d_min and the polygon's width), thinned in two passes:
    first one point per polygon, smallest polygon first, at its most interior spot still free; then the rest
    in random order. The same ``seed`` gives the same points.

    Returns a DataFrame with columns x, y, z, poly_id (index in ``polygons``), edge_dist, margin_ok and the
    extra fields. ``result.attrs["missing"]`` lists the polygons that got no point (another polygon's point
    within d_min, or zero area).
    """
    import shapely
    from scipy.spatial import cKDTree

    if d_min <= 0:
        raise ValueError("d_min must be positive")
    rng = np.random.default_rng(seed)
    rows = []
    for i, item in enumerate(polygons):
        xy, extra = _as_polygon(item)
        if len(xy) < 3 or _area(xy) <= 0:
            continue
        poly = shapely.make_valid(shapely.Polygon(xy))
        if poly.area <= 0:
            continue
        c = _candidates(poly, xy, d_min, rng, max_grid)
        depth = shapely.distance(poly.boundary, shapely.points(c))
        ok = depth >= margin
        if ok.any():
            c, depth = c[ok], depth[ok]
        else:                                                  # too narrow for the margin: most interior spot
            k = [int(depth.argmax())]
            c, depth = c[k], depth[k]
        rows.append(pd.DataFrame({"x": c[:, 0], "y": c[:, 1], "poly_id": i, "edge_dist": depth,
                                  "margin_ok": bool(ok.any()), "_area": poly.area, **extra}))

    cols = ["x", "y", "z", "poly_id", "edge_dist", "margin_ok"]
    if not rows:
        out = pd.DataFrame({c: pd.Series(dtype=float if c in ("x", "y", "z", "edge_dist") else
                                         int if c == "poly_id" else bool) for c in cols})
        out.attrs["missing"] = list(range(len(polygons)))
        return out
    cand = pd.concat(rows, ignore_index=True)

    pts = cand[["x", "y"]].to_numpy()
    tree = cKDTree(pts)
    alive = np.ones(len(pts), bool)
    keep = []

    def take(j):
        keep.append(j)
        alive[tree.query_ball_point(pts[j], r=d_min * (1 - 1e-12))] = False

    # pass 1: one point per polygon, smallest first (it has the fewest options), at its most interior free
    # spot, so thin polygons next to big ones are not crowded out
    pid, depth = cand["poly_id"].to_numpy(), cand["edge_dist"].to_numpy()
    for i in cand.groupby("poly_id")["_area"].first().sort_values(kind="stable").index:
        free = np.flatnonzero((pid == i) & alive)
        if len(free):
            take(free[depth[free].argmax()])
    # pass 2: the rest in random order
    for j in rng.permutation(len(pts)):
        if alive[j]:
            take(j)

    out = cand.iloc[keep].drop(columns="_area").reset_index(drop=True)
    out = out[cols + [c for c in out.columns if c not in cols]]
    out.attrs["missing"] = sorted(set(range(len(polygons))) - set(out["poly_id"].tolist()))
    return out
