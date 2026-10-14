"""Desurvey (minimum curvature), moved from geoview with its tests. The table-level function takes `dip_positive_down` (True by
default); the tables below store dip negative-down, as geoview's did, so they pass `dip_positive_down=False`."""
import warnings

import numpy as np
import pandas as pd
import pytest

from geomodeltools.desurvey import check_tables, desurvey, desurvey_intervals, directions, dogleg_severity, trace_depths

R = 100 / np.radians(30)  # arc radius of a hole turning 30 degrees over 100 m
NEG = {"dip_positive_down": False}


def test_directions_convention():
    np.testing.assert_allclose(directions(0, -90), [0, 0, -1], atol=1e-12)
    np.testing.assert_allclose(directions(90, 0), [1, 0, 0], atol=1e-12)   # east
    np.testing.assert_allclose(directions(0, -60), [0, 0.5, -np.sqrt(3) / 2], atol=1e-12)  # north, down


def test_straight_holes():
    down = desurvey([0, 0, 0], [0], [0], [-90], [0, 100])
    np.testing.assert_allclose(down, [[0, 0, 0], [0, 0, -100]], atol=1e-9)
    inclined = desurvey([10, 20, 30], [0], [90], [-60], [100])
    np.testing.assert_allclose(inclined[0], [10 + 50, 20, 30 - 100 * np.sqrt(3) / 2], atol=1e-9)


def test_minimum_curvature_matches_hand_calculation():
    """Vertical at 0 m, -60 towards east at 100 m: a circular arc of radius R."""
    pos = desurvey([0, 0, 0], [0, 100], [0, 90], [-90, -60], [50, 100])
    np.testing.assert_allclose(pos[1], [R * (1 - np.cos(np.radians(30))), 0, -R * np.sin(np.radians(30))], atol=1e-6)
    np.testing.assert_allclose(pos[0], [R * (1 - np.cos(np.radians(15))), 0, -R * np.sin(np.radians(15))], atol=1e-6)


def test_extra_depths_do_not_change_the_path():
    """Splitting a segment at extra depths must reproduce the same arc (why slerp is used)."""
    st = ([0, 80, 200], [10, 40, 75], [-85, -70, -50])
    coarse = desurvey([0, 0, 0], *st, [200])
    fine = desurvey([0, 0, 0], *st, np.append(np.arange(0, 200, 7.0), 200))
    np.testing.assert_allclose(fine[-1], coarse[0], atol=1e-6)
    beyond = desurvey([0, 0, 0], *st, [200, 250])  # past the last station: straight on
    np.testing.assert_allclose(np.linalg.norm(beyond[1] - beyond[0]), 50, atol=1e-9)


def test_dogleg_severity():
    sev = dogleg_severity([0, 90], [-90, -60], [0, 100])
    np.testing.assert_allclose(sev, [0.0, 30 / 100 * 30], atol=1e-9)           # per 30 m (Leapfrog)
    np.testing.assert_allclose(dogleg_severity([0, 90], [-90, -60], [0, 100], per=100), [0.0, 30.0], atol=1e-9)
    np.testing.assert_allclose(dogleg_severity([90, 0], [-60, -90], [100, 0]), [9.0, 0.0], atol=1e-9)
    sev = dogleg_severity([0, 0, 0, 0], [-90, -90, -90, -90], [0, 10, 10, 10.0000001])
    assert sev[0] == 0 and sev[1] == 0 and np.isnan(sev[2]) and sev[3] == 0
    assert len(dogleg_severity([], [], [])) == 0


def test_trace_depths_include_stations_and_end():
    d = trace_depths([0, 100], [0, 90], [-90, -60], 150.0)
    assert d[0] == 0 and d[-1] == 150 and 100 in d and len(d) > 3


def _hole_tables():
    """A: fine. B: two survey rows at 5 m (leaning east, then west). C: no survey and no azimuth/dip."""
    c = pd.DataFrame({"HOLEID": ["A", "B", "C"], "X": [0.0, 100.0, 200.0], "Y": 0.0, "Z": 0.0, "DEPTH": 10.0})
    s = pd.DataFrame({"HOLEID": ["A", "A", "B", "B", "B", "B"],
                      "DEPTH": [0.0, 10.0, 0.0, 5.0, 5.0, 10.0],
                      "AZIMUTH": [0.0, 0.0, 0.0, 90.0, 270.0, 0.0], "DIP": [-90.0, -90.0, -90.0, -45.0, -45.0, -90.0]})
    return c, s


def test_unplaceable_holes_are_left_out_not_guessed():
    c, s = _hole_tables()
    i = pd.DataFrame({"HOLEID": ["A", "B", "C"], "FROM": 0.0, "TO": 10.0})
    with pytest.warns(UserWarning, match="2 holes were not desurveyed"):
        r = desurvey_intervals(c, s, i, **NEG)
    assert set(r.intervals["HOLEID"]) == {"A"}
    assert dict(r.skipped) == {"B": "a repeated survey depth", "C": "no survey"}   # C is NOT drawn vertical
    np.testing.assert_allclose(r.intervals[["end_x", "end_y", "end_z"]].to_numpy()[-1], [0, 0, -10], atol=1e-9)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="no hole could be desurveyed"):
            desurvey_intervals(c.iloc[[2]], None, i[i.HOLEID == "C"], **NEG)


def test_a_collar_with_its_own_direction_needs_no_survey():
    c, _ = _hole_tables()
    c["AZIMUTH"], c["DIP"] = 90.0, 0.0
    i = pd.DataFrame({"HOLEID": ["C"], "FROM": 0.0, "TO": 10.0})
    r = desurvey_intervals(c[c.HOLEID == "C"], None, i, **NEG)       # east, horizontal: stated, not assumed
    np.testing.assert_allclose(r.intervals[["end_x", "end_y", "end_z"]].to_numpy()[-1], [210, 0, 0], atol=1e-9)


def test_duplicate_survey_rules():
    c, s = _hole_tables()
    c, s = c[c.HOLEID == "B"], s[s.HOLEID == "B"]
    i = pd.DataFrame({"HOLEID": ["B"], "FROM": 0.0, "TO": 10.0})
    ends = {}
    for rule in ("first", "last", lambda g: g.iloc[[-1]]):
        with pytest.warns(UserWarning, match="repeated survey depths"):
            r = desurvey_intervals(c, s, i, duplicates=rule, **NEG)
        ends[rule if isinstance(rule, str) else "function"] = r.intervals[["end_x", "end_y", "end_z"]].to_numpy()[-1]
    assert not r.report.empty                                       # the repeated depth is still reported
    assert ends["first"][0] > 100 and ends["last"][0] < 100 and ends["function"][0] < 100   # east vs west of the collar
    with pytest.raises(ValueError, match="duplicates must be"):
        desurvey_intervals(c, s, i, duplicates="average", **NEG)


def test_check_tables_reports_what_leapfrog_reports():
    c, s = _hole_tables()
    i = pd.DataFrame({"HOLEID": ["A", "A", "B"], "FROM": [0.0, 5.0, 0.0], "TO": [5.0, 10.0000001, 4.0]})
    r = check_tables(c, s, i)
    got = {(k, h): (row, conf) for k, h, row, conf in zip(r.kind, r.hole, r.row, r.conflicting_row)}
    assert got[("duplicate_survey_depth", "B")] == (4, 5)            # rows count from 1, in the table as given
    assert got[("no_survey", "C")][0] == 3 and r.loc[r.kind == "no_survey", "message"].iloc[0] == "no surveys for collar, row 3"
    assert got[("no_intervals", "C")][0] == 3
    assert got[("interval_beyond_depth", "A")] == (2, 1)
    assert (r.type[r.kind == "duplicate_survey_depth"] == "error").all() and (r.type[r.kind == "no_survey"] == "warning").all()
    assert check_tables(c, s, i, tolerance=1e-3).kind.eq("interval_beyond_depth").sum() == 0
    assert check_tables(c.iloc[[0]], s[s.HOLEID == "A"]).empty


def test_intervals_have_start_mid_end_on_the_arc_and_their_row():
    c = pd.DataFrame({"HOLEID": ["A"], "X": 0.0, "Y": 0.0, "Z": 0.0, "DEPTH": 100.0})
    s = pd.DataFrame({"HOLEID": ["A", "A"], "DEPTH": [0.0, 100.0], "AZIMUTH": [0.0, 90.0], "DIP": [-90.0, -60.0]})
    i = pd.DataFrame({"HOLEID": ["A"], "FROM": [0.0], "TO": [100.0], "CU": [1.0]})
    df = desurvey_intervals(c, s, i, **NEG).intervals
    assert list(df.columns) == ["HOLEID", "FROM", "TO", "CU", "_row", "start_x", "start_y", "start_z", "mid_x", "mid_y", "mid_z",
                                "end_x", "end_y", "end_z"]
    mid_chord = (df[["start_x", "start_y", "start_z"]].to_numpy() + df[["end_x", "end_y", "end_z"]].to_numpy()) / 2
    pos15 = [R * (1 - np.cos(np.radians(15))), 0, -R * np.sin(np.radians(15))]   # half way along the arc: 15 degrees turned
    np.testing.assert_allclose(df[["mid_x", "mid_y", "mid_z"]].to_numpy()[0], pos15, atol=1e-6)
    assert np.linalg.norm(mid_chord[0] - pos15) > 0.1             # the chord midpoint is NOT on the arc
    assert list(df["_row"]) == [0]


def test_row_numbers_follow_the_input_even_when_holes_are_left_out():
    c, s = _hole_tables()
    i = pd.DataFrame({"HOLEID": ["C", "A", "B", "A"], "FROM": [0.0, 0.0, 0.0, 5.0], "TO": [5.0, 5.0, 5.0, 10.0]})
    with pytest.warns(UserWarning):
        r = desurvey_intervals(c, s, i, **NEG)
    assert list(r.intervals["_row"]) == [1, 3]                    # rows 0 (C) and 2 (B) were left out


def test_dip_sign_default_is_positive_down_and_columns_are_found_by_name():
    c = pd.DataFrame({"bhid": ["A"], "xcentre": [0.0], "ycentre": [0.0], "zcentre": [0.0]})
    s = pd.DataFrame({"bhid": ["A"], "distance": [0.0], "brg": [0.0], "dip": [90.0]})   # 90 = down
    i = pd.DataFrame({"bhid": ["A"], "from": [0.0], "to": [10.0]})
    r = desurvey_intervals(c, s, i)                                    # default: positive-down
    np.testing.assert_allclose(r.intervals[["end_x", "end_y", "end_z"]].to_numpy()[0], [0, 0, -10], atol=1e-9)
    r2 = desurvey_intervals(c, s.assign(dip=-90.0), i, dip_positive_down=False)
    np.testing.assert_allclose(r2.intervals[["end_z"]].to_numpy()[0], [-10], atol=1e-9)
    up = desurvey_intervals(c, s.assign(dip=-90.0), i)                 # same table read with the wrong sign: goes UP
    assert up.intervals["end_z"].iloc[0] > 0


def test_holes_without_intervals_are_cut_every_trace_step():
    c = pd.DataFrame({"HOLEID": ["A"], "X": 0.0, "Y": 0.0, "Z": 0.0, "DEPTH": 25.0, "AZIMUTH": 0.0, "DIP": -90.0})
    r = desurvey_intervals(c, trace_step=10, **NEG)
    assert list(r.intervals["FROM"]) == [0, 10, 20] and list(r.intervals["TO"]) == [10, 20, 25]
    assert (r.intervals["_row"] == -1).all()
