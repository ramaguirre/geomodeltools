"""Orientation of a geological contact from its outcrop trace on the topography (the rule of V's, as a plane fit).

A planar contact that crosses relief traces V's in the valleys; the strike is the direction of equal elevation along the trace, the V
points down-dip, and the dip follows from the elevation range against the horizontal offset. Fitting a plane to the 3D points of the
trace (x, y, z from a DEM) does all of that at once: the plane's gradient gives dip and dip direction, so the V's direction needs no
separate rule, and the exception (dip in the stream direction and gentler than the stream gradient) is handled because the plane is
fitted in 3D.

The fit is only meaningful when the contact really is a plane and really crosses relief; `fit_contact_plane` says when it is not.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ContactFit:
    strike: float                 # degrees clockwise from north, right-hand rule (dip is to the right of the strike)
    dip: float                    # degrees from horizontal
    dip_azimuth: float            # degrees clockwise from north
    rmse: float                   # m, residual of the plane along the normal
    relief: float                 # m, z range of the trace points used
    extent: float                 # m, horizontal length of the longest dimension
    aspect: float                 # second / first horizontal principal spread (0 = points on a line in plan), informational
    dip_se: float                 # degrees, standard error of the dip from the regression of z on x, y
    n: int
    ok: bool
    reasons: list = field(default_factory=list)   # why the fit should not be trusted (empty when ok)

    def as_dict(self) -> dict:
        return {k: (round(v, 2) if isinstance(v, float) else v) for k, v in self.__dict__.items() if k != "reasons"} | {
            "reasons": "; ".join(self.reasons)}


def _plane(xyz: np.ndarray):
    c = xyz.mean(axis=0)
    u, s, vt = np.linalg.svd(xyz - c, full_matrices=False)
    n = vt[2]
    if n[2] < 0:
        n = -n
    return c, n, s


def fit_contact_plane(xyz, min_relief: float = 100.0, max_rmse_frac: float = 0.06, max_dip_se: float = 5.0,
                      max_dip: float = 75.0, min_dip: float = 3.0, min_points: int = 12, trim: float = 2.5) -> ContactFit:
    """Plane through the 3D points of a contact trace. Points further than `trim` robust sigmas are dropped once (mapping noise).

    The fit is flagged (ok False, reasons listed) when it should not be used:
      * fewer than `min_points` points;
      * relief (z range) below `min_relief` m: a trace on flat ground carries no dip information;
      * the dip is poorly determined (standard error above `max_dip_se` degrees): the trace is too straight in plan or too short for
        the V's to constrain the plane;
      * residual above `max_rmse_frac` of the relief: the contact is not a plane (fold, pluton margin, fault, cover);
      * dip above `max_dip`: V's become too small to read; dip below `min_dip`: indistinguishable from horizontal.
    """
    p = np.asarray(xyz, float)
    p = p[np.isfinite(p).all(axis=1)]
    reasons = []
    if len(p) < min_points:
        return ContactFit(np.nan, np.nan, np.nan, np.nan, float(np.ptp(p[:, 2])) if len(p) else 0.0, 0.0, 0.0, float('nan'), len(p), False,
                          [f"only {len(p)} points"])
    c, n, s = _plane(p)
    res = (p - c) @ n
    sig = 1.4826 * np.median(np.abs(res - np.median(res))) + 1e-9
    keep = np.abs(res) <= trim * sig + 1e-9
    if keep.sum() >= min_points and (~keep).any():
        p = p[keep]
        c, n, s = _plane(p)
        res = (p - c) @ n
    rmse = float(np.sqrt(np.mean(res ** 2)))
    relief = float(np.ptp(p[:, 2]))
    xy = p[:, :2] - p[:, :2].mean(axis=0)
    ev = np.sort(np.linalg.eigvalsh(np.cov(xy.T)))[::-1]
    aspect = float(np.sqrt(max(ev[1], 0) / ev[0])) if ev[0] > 0 else 0.0
    extent = float(max(np.ptp(p[:, 0]), np.ptp(p[:, 1])))
    A = np.column_stack([p[:, 0] - p[:, 0].mean(), p[:, 1] - p[:, 1].mean(), np.ones(len(p))])
    coef, rss, *_ = np.linalg.lstsq(A, p[:, 2] - p[:, 2].mean(), rcond=None)
    sz2 = float(np.sum((p[:, 2] - p[:, 2].mean() - A @ coef) ** 2)) / max(len(p) - 3, 1)
    cov = sz2 * np.linalg.pinv(A.T @ A)
    sl = float(np.hypot(coef[0], coef[1]))
    if sl > 0:
        g = np.array([coef[0], coef[1]]) / sl
        dip_se = float(np.degrees(np.sqrt(g @ cov[:2, :2] @ g) / (1 + sl ** 2)))
    else:
        dip_se = float("inf")
    dip = float(np.degrees(np.arccos(np.clip(n[2], -1, 1))))
    dip_az = float(np.degrees(np.arctan2(n[0], n[1])) % 360)       # downhill = (nx, ny): the plane dips away from its upward gradient
    strike = (dip_az - 90) % 360
    if relief < min_relief:
        reasons.append(f"relief {relief:.0f} m < {min_relief:.0f} m")
    if dip_se > max_dip_se:
        reasons.append(f"dip poorly determined (standard error {dip_se:.1f} deg)")
    if relief > 0 and rmse > max_rmse_frac * relief:
        reasons.append(f"not planar (rmse {rmse:.0f} m is {100 * rmse / relief:.0f}% of the relief)")
    if dip > max_dip:
        reasons.append(f"dip {dip:.0f} deg too steep for V's")
    if dip < min_dip:
        reasons.append(f"dip {dip:.1f} deg indistinguishable from horizontal")
    return ContactFit(strike, dip, dip_az, rmse, relief, extent, aspect, dip_se, len(p), not reasons, reasons)


def contact_points(lines, dem, spacing: float = 30.0) -> np.ndarray:
    """Densify shapely (Multi)LineStrings every `spacing` m and read z from `dem(x, y)` (a callable taking arrays)."""
    import shapely

    pts = []
    for g in lines:
        for part in getattr(g, "geoms", [g]):
            if part.is_empty or part.length == 0:
                continue
            d = np.arange(0, part.length, spacing)
            d = np.append(d, part.length)
            xy = shapely.line_interpolate_point(part, d)
            x, y = shapely.get_x(xy), shapely.get_y(xy)
            pts.append(np.column_stack([x, y, dem(x, y)]))
    return np.vstack(pts) if pts else np.empty((0, 3))


def pooled_fit(trace_sets: list, **kw) -> ContactFit:
    """One plane for several traces of the same contact surface (concatenate their points)."""
    pts = [np.asarray(t, float) for t in trace_sets if len(t)]
    return fit_contact_plane(np.vstack(pts) if pts else np.empty((0, 3)), **kw)
