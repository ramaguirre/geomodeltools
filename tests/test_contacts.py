"""Synthetic terrain and planes of known orientation: the fit must recover them, and refuse when the V-rule does not apply."""
import numpy as np
import pytest

from geomodeltools import contacts as ct


def terrain(x, y):
    return 3000 + 350 * np.sin(x / 700.0) * np.cos(y / 900.0) + 0.05 * x + 0.12 * y + 120 * np.sin((x + y) / 300.0)


def plane_z(dip, az, x0=0.0, y0=0.0, z0=3000.0):
    """Plane dipping `dip` towards azimuth `az` (clockwise from north) through (x0, y0, z0)."""
    ex, ny = np.sin(np.radians(az)), np.cos(np.radians(az))
    t = np.tan(np.radians(dip))
    return lambda x, y: z0 - t * ((x - x0) * ex + (y - y0) * ny)


def trace(surface_z, extent=6000.0, step=40.0, seed=None, noise=0.0):
    """Outcrop trace of the surface on the terrain: the zero contour of terrain - surface, as 3D points."""
    import contourpy
    xs = np.arange(-extent / 2, extent / 2, step)
    ys = np.arange(-extent / 2, extent / 2, step)
    X, Y = np.meshgrid(xs, ys)
    diff = terrain(X, Y) - surface_z(X, Y)
    lines = contourpy.contour_generator(xs, ys, diff).lines(0.0)
    pts = np.vstack([l for l in lines if len(l) > 3])
    z = terrain(pts[:, 0], pts[:, 1])
    if noise:
        z = z + np.random.default_rng(seed).normal(0, noise, len(z))
    return np.column_stack([pts, z])


@pytest.mark.parametrize("dip,az", [(25, 120), (12, 300), (40, 45), (55, 200)])
def test_recovers_dip_and_dip_direction(dip, az):
    fit = ct.fit_contact_plane(trace(plane_z(dip, az)))
    assert fit.ok, fit.reasons
    assert fit.dip == pytest.approx(dip, abs=1.5)
    d = (fit.dip_azimuth - az + 180) % 360 - 180
    assert abs(d) < 3.0
    assert fit.strike == pytest.approx((az - 90) % 360, abs=3.0)


def test_noise_in_the_trace_is_tolerated():
    fit = ct.fit_contact_plane(trace(plane_z(30, 80), noise=8.0, seed=1))
    assert fit.ok and fit.dip == pytest.approx(30, abs=3.0)


def test_horizontal_contact_has_no_relief_and_is_refused():
    fit = ct.fit_contact_plane(trace(lambda x, y: 3000 + 0 * x + 0 * y))
    assert not fit.ok          # follows the contours: dip cannot be read from V's (and the relief is the terrain's, not the contact's)


def test_vertical_contact_is_refused():
    # a vertical plane crosses the terrain as a straight vertical wall: dip above the limit
    ys = np.linspace(-3000, 3000, 200)
    xyz = np.column_stack([np.full_like(ys, 100.0) + 0.5 * ys, ys, terrain(100.0 + 0.5 * ys, ys)])
    fit = ct.fit_contact_plane(xyz)
    assert not fit.ok


def test_curved_contact_is_not_planar():
    # a cone (pluton-like) margin: dip direction changes all round
    th = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    r = 1500.0 + 0 * th
    x, y = r * np.cos(th), r * np.sin(th)
    z = 3000 + 0.35 * r + 120 * np.sin(3 * th)             # same radius, z wobbles with azimuth: no single plane
    fit = ct.fit_contact_plane(np.column_stack([x, y, z]), min_relief=50)
    assert not fit.ok and any("planar" in r or "determined" in r for r in fit.reasons)


def test_few_points_are_refused():
    fit = ct.fit_contact_plane(np.random.default_rng(0).normal(size=(5, 3)))
    assert not fit.ok and "points" in fit.reasons[0]


def test_contact_points_densifies_and_drapes():
    import shapely
    line = shapely.LineString([(0, 0), (300, 0)])
    pts = ct.contact_points([line], lambda x, y: x * 0.1 + 5, spacing=100)
    assert pts.shape == (4, 3) and pts[-1, 2] == pytest.approx(35.0)
