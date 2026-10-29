"""`Polyline` and drillhole `traces`."""
import numpy as np
import pandas as pd
import pytest

from geomodeltools.desurvey import desurvey, trace_depths
from geomodeltools.drillholes import traces
from geomodeltools.shapes import Polyline


def test_polyline_holds_float64_n_by_3():
    v = [[0, 0, 0], [3, 4, 0], [3, 4, 12]]
    p = Polyline(7, v)
    assert p.name == "7" and p.vertices.dtype == np.float64 and p.vertices.shape == (3, 3)
    assert p.length == 17.0
    v[0][0] = 99                                                    # a copy, not a view of the caller's data
    assert p.vertices[0, 0] == 0
    with pytest.raises(ValueError, match="N x 3"):
        Polyline("a", [[0, 0], [1, 1]])
    with pytest.raises(ValueError, match="at least 2"):
        Polyline("a", [[0, 0, 0]])


def test_traces_one_polyline_per_hole_on_the_desurveyed_arc():
    c = pd.DataFrame({"HOLEID": ["A", 17], "X": [1000.0, 0.0], "Y": [2000.0, 0.0], "Z": [500.0, 0.0], "LENGTH": [150.0, np.nan]})
    s = pd.DataFrame({"HOLEID": ["A", "A", 17, 17], "DEPTH": [0.0, 100.0, 0.0, 40.0], "AZIMUTH": [0.0, 90.0, 0.0, 0.0],
                      "DIP": [90.0, 60.0, 90.0, 90.0]})                 # positive down
    t = traces(c, s)
    assert [p.name for p in t] == ["A", "17"]
    a = t[0]
    depths = trace_depths([0, 100], [0, 90], [-90, -60], 150.0)
    np.testing.assert_allclose(a.vertices, desurvey([1000, 2000, 500], [0, 100], [0, 90], [-90, -60], depths), atol=1e-9)
    assert len(a.vertices) > 3 and a.length == pytest.approx(150.0, abs=0.05)
    np.testing.assert_allclose(t[1].vertices, [[0, 0, 0], [0, 0, -40]], atol=1e-9)   # no length: deepest station
    neg = s.assign(DIP=-s.DIP)
    np.testing.assert_allclose(traces(c, neg, dip_positive_down=False)[0].vertices, a.vertices, atol=1e-9)


def test_traces_leave_out_what_cannot_be_placed():
    c = pd.DataFrame({"HOLEID": ["A", "C", "D"], "X": 0.0, "Y": 0.0, "Z": 0.0, "AZIMUTH": [np.nan, np.nan, 90.0],
                      "DIP": [np.nan, np.nan, 0.0]})
    s = pd.DataFrame({"HOLEID": ["A", "A"], "DEPTH": [0.0, 10.0], "AZIMUTH": 0.0, "DIP": 90.0})
    with pytest.warns(UserWarning, match="2 holes were not desurveyed") as w:
        t = traces(c, s)
    assert [p.name for p in t] == ["A"]                              # C: no survey; D: a direction but no length
    assert "no survey" in str(w[0].message) and "no hole length" in str(w[0].message)
    assert w[0].filename == __file__
