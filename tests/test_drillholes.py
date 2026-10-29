"""Depth tables in real coordinates (`interval_table`, `point_table`), wrapping `desurvey_intervals`."""
import numpy as np
import pandas as pd
import pytest

from geomodeltools.drillholes import interval_table, point_table

R = 100 / np.radians(30)  # arc radius of a hole turning 30 degrees over 100 m


def _arc(deg):
    """Position on hole A after turning `deg` degrees (vertical at 0 m, -60 towards east at 100 m)."""
    return [1000 + R * (1 - np.cos(np.radians(deg))), 2000, 500 - R * np.sin(np.radians(deg))]


def _tables(positive_down=True):
    sign = 1 if positive_down else -1
    c = pd.DataFrame({"BHID": ["A", "B"], "East": [1000.0, 0.0], "North": [2000.0, 0.0], "RL": [500.0, 0.0], "DEPTH": 100.0})
    s = pd.DataFrame({"BHID": ["A", "A", "B"], "AT": [0.0, 100.0, 0.0], "AZIMUTH": [0.0, 90.0, 0.0],
                      "DIP": [sign * 90.0, sign * 60.0, sign * 90.0]})
    return c, s


def test_interval_table_columns_and_positions():
    c, s = _tables()
    i = pd.DataFrame({"BHID": ["A", "A", "B"], "FROM": [0.0, 50.0, 0.0], "TO": [50.0, 100.0, 10.0], "CU": [1.0, 2.0, 3.0],
                      "LITH": ["x", "y", "z"]})
    t = interval_table(c, s, i)
    assert list(t.columns) == ["HOLEID", "FROM", "TO", "CU", "LITH", "TOP_X", "TOP_Y", "TOP_Z", "MID_X", "MID_Y", "MID_Z",
                               "BOTTOM_X", "BOTTOM_Y", "BOTTOM_Z"]
    a = t[t.HOLEID == "A"]
    np.testing.assert_allclose(a[["TOP_X", "TOP_Y", "TOP_Z"]].to_numpy(), [_arc(0), _arc(15)], atol=1e-6)
    np.testing.assert_allclose(a[["MID_X", "MID_Y", "MID_Z"]].to_numpy(), [_arc(7.5), _arc(22.5)], atol=1e-6)
    np.testing.assert_allclose(a[["BOTTOM_X", "BOTTOM_Y", "BOTTOM_Z"]].to_numpy(), [_arc(15), _arc(30)], atol=1e-6)
    np.testing.assert_allclose(t.loc[t.HOLEID == "B", ["BOTTOM_X", "BOTTOM_Y", "BOTTOM_Z"]].to_numpy(), [[0, 0, -10]], atol=1e-9)
    assert list(t.LITH) == ["x", "y", "z"] and list(t.CU) == [1.0, 2.0, 3.0]
    assert list(interval_table(c, s, i, attributes=["CU"]).columns[:4]) == ["HOLEID", "FROM", "TO", "CU"]


def test_interval_table_dip_flag_is_passed_through():
    c, s = _tables(positive_down=False)
    i = pd.DataFrame({"BHID": ["A"], "FROM": [0.0], "TO": [100.0]})
    t = interval_table(c, s, i, dip_positive_down=False)
    np.testing.assert_allclose(t[["BOTTOM_X", "BOTTOM_Y", "BOTTOM_Z"]].to_numpy()[0], _arc(30), atol=1e-6)
    up = interval_table(c, s, i)                                     # default positive-down reads these dips as upwards
    assert up["BOTTOM_Z"].iloc[0] > 500


def test_point_table_columns_and_positions():
    c, s = _tables()
    p = pd.DataFrame({"BHID": ["A", "A", "B"], "MD": [0.0, 50.0, 25.0], "SG": [2.6, 2.7, 2.8]}, index=[10, 20, 30])
    t = point_table(c, s, p)
    assert list(t.columns) == ["HOLEID", "DEPTH", "SG", "X", "Y", "Z"]
    np.testing.assert_allclose(t[["X", "Y", "Z"]].to_numpy(), [_arc(0), _arc(15), [0, 0, -25]], atol=1e-6)
    assert list(t.DEPTH) == [0.0, 50.0, 25.0] and list(t.SG) == [2.6, 2.7, 2.8]
    cn, sn = _tables(positive_down=False)
    tn = point_table(cn, sn, p, dip_positive_down=False, depth="MD", attributes=[])
    assert list(tn.columns) == ["HOLEID", "DEPTH", "X", "Y", "Z"]
    np.testing.assert_allclose(tn[["X", "Y", "Z"]].to_numpy(), t[["X", "Y", "Z"]].to_numpy(), atol=1e-9)


def test_unplaceable_holes_are_left_out_with_a_warning():
    c, s = _tables()
    c = pd.concat([c, pd.DataFrame({"BHID": ["C"], "East": [5.0], "North": [5.0], "RL": [5.0], "DEPTH": [10.0]})])
    p = pd.DataFrame({"BHID": ["A", "C", "Z"], "DEPTH": [10.0, 5.0, 1.0]})
    with pytest.warns(UserWarning) as w:
        t = point_table(c, s, p)
    assert list(t.HOLEID) == ["A"]
    text = " ".join(str(m.message) for m in w)
    assert "1 holes were not desurveyed" in text and "no collar" in text
    assert all(m.filename == __file__ for m in w)                  # the warning points at the caller
    with pytest.warns(UserWarning, match="not desurveyed"):
        assert list(interval_table(c, s, pd.DataFrame({"BHID": ["A", "C"], "FROM": 0.0, "TO": 5.0})).HOLEID) == ["A"]
