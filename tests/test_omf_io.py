"""OMF v1 writer and reader: round trips through our own reader, and the `omf` package (a test dependency only) both opening
our files and writing files our reader opens."""
import struct
import warnings

import numpy as np
import pytest

from geomodeltools.omf_io import read_omf, write_omf
from geomodeltools.shapes import LineSet, PointSet, Polyline, Surface

E, N = 350_123.456789, 6_543_210.987654                     # real UTM magnitudes: float64 keeps them exactly


def _elements():
    rng = np.random.default_rng(0)
    pts = PointSet("assays", rng.random((5, 3)) * 100 + [E, N, 1200], {"cu": [0.1, 0.2, np.nan, 0.4, 0.5], "au": np.arange(5)})
    traces = LineSet.from_polylines("traces", [Polyline("A", [[E, N, 1000], [E, N, 900], [E + 10, N, 800]]),
                                               Polyline("B", [[E + 50, N, 1000], [E + 50, N + 5, 950]])])
    single = Polyline("fault", [[E, N, 0], [E + 1, N + 1, 1], [E + 2, N, 2], [E + 3, N + 3, 3]])
    surf = Surface("topo", [[E, N, 1], [E + 10, N, 2], [E, N + 10, 3], [E + 10, N + 10, 4]], [[0, 1, 2], [1, 3, 2]])
    return pts, traces, single, surf


def test_header_is_60_bytes_and_points_at_the_json(tmp_path):
    path = write_omf(tmp_path / "x.omf", _elements(), name="demo")
    raw = path.read_bytes()
    assert raw[:4] == b"\x84\x83\x82\x81" and raw[4:14] == b"OMF-v0.9.0" and raw[14:36] == b"\x00" * 22
    start = struct.unpack("<Q", raw[52:60])[0]
    assert raw[start:start + 1] == b"{" and start > 60


def test_round_trip_own_reader(tmp_path):
    pts, traces, single, surf = _elements()
    back = read_omf(write_omf(tmp_path / "x.omf", [pts, traces, single, surf]))
    assert [type(e).__name__ for e in back] == ["PointSet", "LineSet", "LineSet", "Surface"]
    assert [e.name for e in back] == ["assays", "traces", "fault", "topo"]
    np.testing.assert_array_equal(back[0].vertices, pts.vertices)
    np.testing.assert_array_equal(back[0].data["cu"], pts.data["cu"])          # NaN kept
    np.testing.assert_array_equal(back[0].data["au"], [0, 1, 2, 3, 4])
    np.testing.assert_array_equal(back[1].segments, [[0, 1], [1, 2], [3, 4]])
    lines = back[1].to_polylines()
    assert [p.name for p in lines] == ["traces_1", "traces_2"]
    np.testing.assert_array_equal(lines[0].vertices, [[E, N, 1000], [E, N, 900], [E + 10, N, 800]])
    (fault,) = back[2].to_polylines()
    assert fault.name == "fault"
    np.testing.assert_array_equal(fault.vertices, single.vertices)
    np.testing.assert_array_equal(back[3].vertices, surf.vertices)
    np.testing.assert_array_equal(back[3].triangles, surf.triangles)


def test_bad_input_is_refused(tmp_path):
    with pytest.raises(TypeError, match="cannot write"):
        write_omf(tmp_path / "x.omf", [np.zeros((3, 3))])
    with pytest.raises(ValueError, match="outside"):
        Surface("s", [[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 3]])
    with pytest.raises(ValueError, match="values for"):
        PointSet("p", [[0, 0, 0]], {"v": [1, 2]})
    (tmp_path / "bad.omf").write_bytes(b"not omf at all" * 10)
    with pytest.raises(ValueError, match="not an OMF file"):
        read_omf(tmp_path / "bad.omf")


def test_the_omf_package_opens_our_files(tmp_path):
    omf = pytest.importorskip("omf")
    pts, traces, single, surf = _elements()
    path = write_omf(tmp_path / "x.omf", [pts, traces, single, surf], name="demo", description="d", units="m")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                              # omf 1.0 and its dependencies warn on recent numpy/Python
        project = omf.OMFReader(str(path)).get_project()
        project.validate()
    assert (project.name, project.description, project.units) == ("demo", "d", "m")
    p, l, f, s = project.elements
    assert isinstance(p, omf.PointSetElement) and isinstance(l, omf.LineSetElement) and isinstance(s, omf.SurfaceElement)
    np.testing.assert_array_equal(np.asarray(p.geometry.vertices), pts.vertices)
    assert {d.name: np.asarray(d.array) for d in p.data}.keys() == {"cu", "au"}
    np.testing.assert_array_equal(np.asarray(p.data[0].array), pts.data["cu"])
    np.testing.assert_array_equal(np.asarray(l.geometry.segments), traces.segments)
    np.testing.assert_array_equal(np.asarray(f.geometry.vertices), single.vertices)
    np.testing.assert_array_equal(np.asarray(s.geometry.triangles), surf.triangles)
    np.testing.assert_array_equal(np.asarray(s.geometry.vertices), surf.vertices)


def test_our_reader_opens_omf_package_files(tmp_path):
    omf = pytest.importorskip("omf")
    v = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 1]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        project = omf.Project(name="p", origin=[E, N, 0.0])
        project.elements = [
            omf.PointSetElement(name="pts", geometry=omf.PointSetGeometry(vertices=v, origin=[0, 0, 10.0]),
                                data=[omf.ScalarData(name="v", array=np.arange(4.0), location="vertices"),
                                      omf.StringData(name="s", array=["a", "b", "c", "d"], location="vertices")]),
            omf.LineSetElement(name="ls", geometry=omf.LineSetGeometry(vertices=v, segments=[[0, 1], [1, 2], [2, 3]])),
            omf.SurfaceElement(name="sf", geometry=omf.SurfaceGeometry(vertices=v, triangles=[[0, 1, 2], [1, 3, 2]])),
        ]
        project.validate()
        omf.OMFWriter(project, str(tmp_path / "theirs.omf"))
    with pytest.warns(UserWarning, match="data 's' of 'pts'"):
        pts, ls, sf = read_omf(tmp_path / "theirs.omf")
    np.testing.assert_array_equal(pts.vertices, v + [E, N, 10.0])  # project and geometry origins added
    assert pts.data.keys() == {"v"}
    np.testing.assert_array_equal(pts.data["v"], np.arange(4.0))
    np.testing.assert_array_equal(ls.segments, [[0, 1], [1, 2], [2, 3]])
    np.testing.assert_array_equal(sf.triangles, [[0, 1, 2], [1, 3, 2]])
    np.testing.assert_array_equal(sf.vertices, v + [E, N, 0.0])
