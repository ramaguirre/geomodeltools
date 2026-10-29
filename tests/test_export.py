"""DXF (ezdxf, extra [dxf]), OBJ and CSV export."""
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from geomodeltools.export import dxf_layer_name, write_csv, write_dxf, write_obj
from geomodeltools.shapes import LineSet, PointSet, Polyline, Surface

E, N = 350_123.456789, 6_543_210.987654


def _shapes():
    hole = Polyline("DDH-001", [[E, N, 1000], [E, N, 900], [E + 10, N, 800]])
    traces = LineSet.from_polylines("traces", [Polyline("A", [[E, N, 0], [E, N, -10]]), Polyline("B", [[E + 5, N, 0], [E + 5, N, -5]])])
    surf = Surface("topo surface", [[E, N, 1], [E + 10, N, 2], [E, N + 10, 3], [E + 10, N + 10, 4]], [[0, 1, 2], [1, 3, 2]])
    return hole, traces, surf


def test_dxf_polylines_are_3d_polylines_and_surfaces_3d_faces(tmp_path):
    ezdxf = pytest.importorskip("ezdxf")
    hole, traces, surf = _shapes()
    path = write_dxf(tmp_path / "x.dxf", [hole, traces, surf])
    msp = ezdxf.readfile(path).modelspace()
    polys = msp.query("POLYLINE")
    assert len(polys) == 3 and all(p.is_3d_polyline for p in polys)
    assert [p.dxf.layer for p in polys] == ["DDH-001", "A", "B"]
    np.testing.assert_allclose([tuple(v.dxf.location) for v in polys[0].vertices], hole.vertices, rtol=0, atol=1e-9)
    faces = msp.query("3DFACE")
    assert len(faces) == 2 and {f.dxf.layer for f in faces} == {"topo surface"}
    np.testing.assert_allclose([faces[1].dxf.vtx0, faces[1].dxf.vtx1, faces[1].dxf.vtx2], surf.vertices[[1, 3, 2]], atol=1e-9)
    one = ezdxf.readfile(write_dxf(tmp_path / "y.dxf", [hole, surf], layer="all/of:it")).modelspace()
    assert {e.dxf.layer for e in one} == {"all_of_it"}


def test_dxf_layer_names_and_bad_input():
    assert dxf_layer_name('a<b>c/d\\e"f:g;h?i*j|k=l`m') == "a_b_c_d_e_f_g_h_i_j_k_l_m"
    assert dxf_layer_name("  ") == "0"
    with pytest.raises(TypeError, match="cannot export PointSet"):
        write_obj("unused.obj", [PointSet("p", [[0, 0, 0]])])


def test_write_dxf_without_ezdxf_names_the_extra(tmp_path):
    code = ("import sys\nsys.modules['ezdxf'] = None\nfrom geomodeltools.export import write_dxf\n"
            f"try:\n    write_dxf({str(tmp_path / 'x.dxf')!r}, [])\nexcept ImportError as e:\n    print(e)\n")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert "pip install geomodeltools[dxf]" in r.stdout, r.stderr


def _read_obj(path):
    objs, verts = {}, []
    for line in path.read_text().splitlines():
        tag, *rest = line.split()
        if tag == "o":
            cur = objs.setdefault(rest[0], {"l": [], "f": []})
        elif tag == "v":
            verts.append([float(x) for x in rest])
        else:
            cur[tag].append([int(k) - 1 for k in rest])
    return objs, np.array(verts)


def test_obj_lines_and_faces_in_full_precision(tmp_path):
    hole, traces, surf = _shapes()
    objs, v = _read_obj(write_obj(tmp_path / "x.obj", [hole, traces, surf]))
    assert list(objs) == ["DDH-001", "A", "B", "topo_surface"]
    np.testing.assert_array_equal(v[objs["DDH-001"]["l"][0]], hole.vertices)          # exact: floats in full
    np.testing.assert_array_equal(v[objs["B"]["l"][0]], [[E + 5, N, 0], [E + 5, N, -5]])
    np.testing.assert_array_equal(np.array(objs["topo_surface"]["f"]), surf.triangles + 7)  # after 3 + 2 + 2 line vertices
    np.testing.assert_array_equal(v[7:], surf.vertices)


def test_csv_of_a_table_round_trips(tmp_path):
    t = pd.DataFrame({"HOLEID": ["A", "B"], "DEPTH": [1.5, 2.0], "X": [E, E + 0.1], "Y": [N, N], "Z": [0.1 + 0.2, -5.0]})
    back = pd.read_csv(write_csv(tmp_path / "t.csv", t))
    pd.testing.assert_frame_equal(back, t)
    assert (tmp_path / "t.csv").read_text().splitlines()[0] == "HOLEID,DEPTH,X,Y,Z"
    assert pd.read_csv(write_csv(tmp_path / "s.csv", t, sep=";"), sep=";").shape == (2, 5)
