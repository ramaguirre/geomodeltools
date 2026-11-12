"""Drillhole sets: tables recognised by their columns and sorted into sets by the holes they share; nothing unclear is guessed."""
import numpy as np
import pandas as pd
import pytest

from geomodeltools import DrillholeSet, drillhole_role, group_tables
from geomodeltools.drillholes import interval_table


def _campaign(prefix, x0):
    holes = [f"{prefix}{k}" for k in range(1, 4)]
    collars = pd.DataFrame({"HOLEID": holes, "X": [x0, x0 + 50, x0 + 100], "Y": 0.0, "Z": 100.0, "LENGTH": 60.0})
    surveys = pd.DataFrame({"HOLEID": np.repeat(holes, 2), "DEPTH": [0.0, 60] * 3, "AZIMUTH": 90.0, "DIP": [60.0, 62] * 3})
    assays = pd.DataFrame({"HOLEID": np.repeat(holes, 3), "FROM": [0.0, 20, 40] * 3, "TO": [20.0, 40, 60] * 3, "CU": 0.5})
    lith = pd.DataFrame({"HOLEID": np.repeat(holes, 2), "FROM": [0.0, 30] * 3, "TO": [30.0, 60] * 3, "ROCK": ["P", "B"] * 3})
    structures = pd.DataFrame({"HOLEID": holes, "DEPTH": [10.0, 25, 40], "ALPHA": [30.0, 45, 60]})
    return {f"{prefix}_collars": collars, f"{prefix}_surveys": surveys, f"{prefix}_assays": assays, f"{prefix}_lith": lith,
            f"{prefix}_structures": structures}


def test_roles_are_found_from_columns_and_free_points_are_not_set_tables():
    t = _campaign("A", 0.0)
    assert [drillhole_role(df) for df in t.values()] == ["collars", "surveys", "intervals", "intervals", "points"]
    photos = pd.DataFrame({"HOLEID": ["A1"] * 30, "X": np.arange(30.0), "Y": 0.0, "Z": 0.0})   # X/Y/Z on many rows per hole
    assert drillhole_role(photos) is None
    spanish = t["A_collars"].rename(columns={"HOLEID": "Sondaje", "X": "Este", "Y": "Norte", "Z": "Cota"})
    assert drillhole_role(spanish) == "collars"
    assert drillhole_role(pd.DataFrame({"X": [1.0], "Y": [2.0], "Z": [3.0]})) is None


def test_two_campaigns_loaded_together_land_in_two_sets():
    tables = {**_campaign("A", 0.0), **_campaign("B", 1000.0)}
    g = group_tables(dict(reversed(list(tables.items()))))        # any order
    assert {s.name for s in g.sets} == {"A_collars", "B_collars"} and not g.unsure and not g.other
    a = next(s for s in g.sets if s.name == "A_collars")
    assert a.surveys_name == "A_surveys" and sorted(a.intervals) == ["A_assays", "A_lith"] and list(a.points) == ["A_structures"]
    assert a.tables()[0] == ("A_collars", "collars")


def test_unclear_tables_are_returned_not_guessed():
    a, b = _campaign("A", 0.0), _campaign("B", 1000.0)
    both = pd.concat([a["A_assays"], b["B_assays"]], ignore_index=True)              # half its holes in each set
    stray = pd.DataFrame({"HOLEID": ["Z1", "Z2", "A1"], "FROM": [0.0] * 3, "TO": [1.0] * 3})  # mostly unknown holes
    g = group_tables({"A_collars": a["A_collars"], "B_collars": b["B_collars"], "A_surveys": a["A_surveys"],
                      "again": a["A_surveys"], "both": both, "stray": stray})
    why = {u.name: u.why for u in g.unsure}
    assert set(why) == {"again", "both", "stray"}
    assert "already has a survey table" in why["again"] and "about equally" in why["both"] and "only 33%" in why["stray"]
    assert next(u for u in g.unsure if u.name == "stray").scores[0] == ("A_collars", 0.333)
    no_collars = group_tables({"lonely": a["A_assays"]})
    assert no_collars.unsure[0].why.startswith("there is no collar table")


def test_a_table_can_be_reassigned_and_a_role_given():
    a, b = _campaign("A", 0.0), _campaign("B", 1000.0)
    g = group_tables({"A_collars": a["A_collars"], "B_collars": b["B_collars"], "lith": a["A_lith"]})
    sa, sb = g.sets
    assert "lith" in sa.intervals
    sb.add("lith", sa.remove("lith"))                                                # the user moves it
    assert "lith" not in sa.intervals and "lith" in sb.intervals and sb.missing(sb.intervals["lith"]) == {"A1", "A2", "A3"}
    g2 = group_tables({"A_collars": a["A_collars"], "odd": a["A_structures"].rename(columns={"DEPTH": "AT"})},
                      roles={"odd": "points"})
    assert "odd" in g2.sets[0].points
    with pytest.raises(ValueError, match="collar table starts its own set"):
        sa.add("more collars", a["A_collars"])


def test_a_set_desurveys_as_the_tables_do_on_their_own():
    a = _campaign("A", 0.0)
    s = DrillholeSet("A", a["A_collars"], a["A_surveys"], dip_positive_down=True)
    s.add("assays", a["A_assays"])
    s.add("structures", a["A_structures"])
    out = s.desurvey()
    pd.testing.assert_frame_equal(out["assays"], interval_table(a["A_collars"], a["A_surveys"], a["A_assays"], dip_positive_down=True))
    assert list(out["structures"].columns[-3:]) == ["X", "Y", "Z"] and len(s.traces()) == 3
    no_survey = DrillholeSet("vertical", a["A_collars"].assign(AZIMUTH=0.0, DIP=90.0))   # holes oriented by their collars
    no_survey.add("assays", a["A_assays"])
    assert len(no_survey.interval_table("assays")) == 9
