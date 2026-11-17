"""Curved polylines (Bezier handles) and sides: `Polyline.controls` / `.normals`, and how DXF, OBJ and OMF keep them."""
from types import SimpleNamespace

import numpy as np
import pytest

from geomodeltools.export import write_obj
from geomodeltools.omf_io import read_omf, write_omf
from geomodeltools.shapes import LineSet, Polyline

E, N = 350000.0, 7400000.0                                           # UTM magnitudes, as in the other tests
V = np.array([[E, N, 900.0], [E + 100, N, 880.0], [E + 200, N, 890.0]])


def straight_controls(v):
    c = np.repeat(v[:, None], 2, axis=1).copy()
    d = np.diff(v, axis=0) / 3
    c[:-1, 1] += d
    c[1:, 0] -= d
    return c


def bent():
    c = straight_controls(V)
    c[0, 1, 2] += 30                                                 # first segment bulges up
    c[1, 0, 2] += 30
    nrm = np.tile([0.0, 0.0, 1.0], (3, 1))
    return Polyline("contact", V, c, nrm)


def test_a_plain_polyline_is_unchanged():
    p = Polyline("a", V)
    assert not p.curved and p.controls is None and p.normals is None
    np.testing.assert_array_equal(p.curve_points(), V)
    assert p.length == pytest.approx(np.linalg.norm(np.diff(V, axis=0), axis=1).sum())


def test_shapes_are_checked():
    with pytest.raises(ValueError, match="controls must be N x 2 x 3"):
        Polyline("a", V, np.zeros((3, 3)))
    with pytest.raises(ValueError, match="normals must be N x 3"):
        Polyline("a", V, None, np.zeros((2, 3)))
    with pytest.raises(ValueError, match="finite"):
        Polyline("a", V, np.full((3, 2, 3), np.nan))


def test_straight_handles_give_the_straight_line():
    p = Polyline("a", V, straight_controls(V))
    pts = p.curve_points(5.0)
    assert len(pts) > 30
    # every point on the chord of its segment
    for q in pts:
        k = 0 if q[0] <= V[1, 0] else 1
        t = (q[0] - V[k, 0]) / (V[k + 1, 0] - V[k, 0])
        np.testing.assert_allclose(q, V[k] + t * (V[k + 1] - V[k]), atol=1e-9)
    assert p.length == pytest.approx(Polyline("b", V).length, rel=1e-9)


def test_a_curve_passes_through_its_vertices_and_bends():
    p = bent()
    pts = p.curve_points(1.0)
    for v in V:
        assert np.min(np.linalg.norm(pts - v, axis=1)) < 1e-9          # every vertex exactly
    gaps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    assert gaps.max() < 1.3                                            # about 1 m apart
    mid = pts[np.argmin(np.abs(pts[:, 0] - (E + 50)))]
    assert mid[2] == pytest.approx(890 + 0.75 * 30, abs=0.5)          # chord at 890 there; the bulge is 0.75 x 30 at t = 1/2
    assert p.length > Polyline("s", V).length


def test_from_edit_keeps_handles_and_sides():
    b = bent()
    p = Polyline.from_edit(SimpleNamespace(name="x", vertices=b.vertices, controls=b.controls, normals=b.normals))
    np.testing.assert_array_equal(p.controls, b.controls)
    np.testing.assert_array_equal(p.normals, b.normals)
    q = Polyline.from_edit(SimpleNamespace(name="y", vertices=V))   # an edit without handles
    assert not q.curved and q.normals is None


def test_obj_and_line_sets_get_points_along_the_curve(tmp_path):
    p = bent()
    lines = write_obj(tmp_path / "c.obj", [p], curve_step=2.0).read_text().splitlines()
    v = np.array([[float(x) for x in s.split()[1:]] for s in lines if s.startswith("v ")])
    np.testing.assert_allclose(v, p.curve_points(2.0))
    ls = LineSet.from_polylines("set", [p, Polyline("flat", V)], curve_step=2.0)
    assert len(ls.vertices) == len(p.curve_points(2.0)) + 3


def test_omf_keeps_the_curve_for_geomodeltools_and_its_shape_for_others(tmp_path):
    f = write_omf(tmp_path / "c.omf", [bent(), Polyline("flat", V)], curve_step=2.0)
    back = read_omf(f)
    assert isinstance(back[0], Polyline) and back[0].name == "contact"
    np.testing.assert_array_equal(back[0].vertices, V)
    np.testing.assert_array_equal(back[0].controls, bent().controls)
    np.testing.assert_array_equal(back[0].normals, bent().normals)
    assert isinstance(back[1], LineSet)                               # a plain polyline is a line set, as before
    omf = pytest.importorskip("omf")
    proj = omf.OMFReader(str(f)).get_project()
    geom = proj.elements[0].geometry                                  # other readers see the curve's shape
    np.testing.assert_allclose(np.asarray(geom.vertices.array) + np.asarray(geom.origin), bent().curve_points(2.0))


def test_sides_alone_survive_omf(tmp_path):
    p = Polyline("sided", V, None, np.tile([0.0, 0.0, 1.0], (3, 1)))
    back = read_omf(write_omf(tmp_path / "s.omf", [p]))[0]
    assert isinstance(back, Polyline) and not back.curved
    np.testing.assert_array_equal(back.normals, p.normals)


def test_an_ordinary_description_is_not_taken_for_a_curve(tmp_path):
    from geomodeltools.omf_io import _polyline_from_note
    assert _polyline_from_note("a", "") is None
    assert _polyline_from_note("a", "{not json") is None
    assert _polyline_from_note("a", '{"other": 1}') is None
