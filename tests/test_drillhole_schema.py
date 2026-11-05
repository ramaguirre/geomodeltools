"""Drillhole sets from specs: read -> guess -> review -> confirm -> validate -> save -> open (the onboarding of collar.csv, survey.csv and
intervals.csv), and the pieces on their own."""
import numpy as np
import pandas as pd
import pytest

from geomodeltools import (DrillholeSet, NeedsReview, ParseOptions, SpecMismatch, TableSpec, guess_table, interval_table, read_table)
from geomodeltools.guess import detect_dip_convention, table_kind
from geomodeltools.schema import ColumnSpec, signature


def _write_campaign(folder):
    """A small campaign in Spanish, as Chilean tables are: semicolons and decimal commas in the collars, a "<0.01" in the assays, one
    overlapping interval, and one hole written differently in the assays."""
    holes = ["DDH-01", "DDH-02", "DDH-03"]
    collars = pd.DataFrame({"SONDAJE": holes, "ESTE": [1000.5, 1100.25, 1200.0], "NORTE": [5000.0, 5000.0, 5050.0], "COTA": [2500.0, 2510.0, 2490.0],
                            "LARGO": [100.0, 120.0, 80.0]})
    surveys = pd.DataFrame({"SONDAJE": np.repeat(holes, 2), "PROFUNDIDAD": [0.0, 100, 0, 120, 0, 80], "AZIMUT": [90.0, 92, 180, 182, 0, 5],
                            "INCLINACION": [-60.0, -62, -70, -71, -90, -88]})
    lith = ["AND", "DIO", "AND", "POR", "DIO", "AND", "AND", "POR", "DIO"]
    assays = pd.DataFrame({"SONDAJE": ["DDH-01"] * 3 + ["DDH-02"] * 3 + ["ddh03"] * 3,
                           "DESDE": [0.0, 30, 60, 0, 40, 80, 0, 25, 50], "HASTA": [30.0, 60, 100, 45, 80, 120, 25, 50, 80],
                           "CU": ["0.52", "0.81", "<0.01", "0.3", "0.44", "0.9", "1.2", "0.1", "0.05"] + [],
                           "LITO": lith, "OBS": [f"observacion larga numero {k} del intervalo" for k in range(9)]})
    collars.to_csv(folder / "collar.csv", sep=";", decimal=",", index=False)
    surveys.to_csv(folder / "survey.csv", index=False)
    assays.to_csv(folder / "intervals.csv", index=False)
    return collars, surveys, assays


# ---- reading -------------------------------------------------------------------------------------------------------------------


def test_a_semicolon_file_with_decimal_commas_is_read_and_says_how(tmp_path):
    _write_campaign(tmp_path)
    df, parse = read_table(tmp_path / "collar.csv")
    assert parse.delimiter == ";" and parse.decimal == "," and df["ESTE"].tolist() == [1000.5, 1100.25, 1200.0]
    again, _ = read_table((tmp_path / "collar.csv").read_bytes(), parse, name="collar.csv")
    pd.testing.assert_frame_equal(again, df)
    assert ParseOptions.from_dict(parse.to_dict()) == parse


# ---- guessing ------------------------------------------------------------------------------------------------------------------


def test_the_guess_finds_kinds_roles_types_and_says_what_to_review(tmp_path):
    collars, surveys, _ = _write_campaign(tmp_path)
    assays, _ = read_table(tmp_path / "intervals.csv")
    g = guess_table(assays, "intervals.csv", known_holes=set(collars["SONDAJE"]))
    assert g.kind == "intervals" and g.roles() == {"hole": "SONDAJE", "from": "DESDE", "to": "HASTA"}
    by = {c.name: c for c in g.columns}
    assert by["LITO"].dtype == "category" and by["OBS"].dtype == "text" and by["CU"].dtype == "number"
    assert by["DESDE"].unit == "m" and any("DESDE < HASTA in 100%" in r for r in by["DESDE"].reasons)
    assert "CU" in g.needs_review() and "'<0.01'" in by["CU"].reasons[0]          # mostly numbers: asks for a look
    assert "<- review" in str(g)
    assert guess_table(surveys).roles() == {"hole": "SONDAJE", "depth": "PROFUNDIDAD", "azimuth": "AZIMUT", "dip": "INCLINACION"}


def test_the_kind_rules_are_those_drillhole_sets_always_used():
    c = pd.DataFrame({"HOLEID": ["A"], "X": [1.0], "Y": [2.0], "Z": [3.0]})
    assert table_kind(c) == "collars"
    assert table_kind(pd.DataFrame({"HOLEID": ["A"] * 30, "X": np.arange(30.0), "Y": 0.0, "Z": 0.0})) == "xyz"   # photos with a hole id
    assert table_kind(pd.DataFrame({"X": [1.0], "Y": [1.0], "Z": [1.0], "DX": [5.0], "DY": [5.0], "DZ": [5.0]})) == "blocks"
    assert table_kind(pd.DataFrame({"HOLEID": ["A"], "AT": [3.0]})) == "points"
    assert table_kind(pd.DataFrame({"A": [1]})) is None


def test_values_lower_a_guess_made_from_names():
    bad = pd.DataFrame({"HOLEID": ["A", "B"], "DEPTH": [0.0, 10], "AZIMUTH": [400.0, 90], "DIP": [-60.0, -50]})
    az = next(c for c in guess_table(bad).columns if c.name == "AZIMUTH")
    assert az.confidence < 0.8 and "outside 0-360" in "; ".join(az.reasons)
    generic = guess_table(pd.DataFrame({"ID": ["A", "B"], "X": [1.0, 2], "Y": [1.0, 2], "Z": [1.0, 2]}))
    assert generic.kind == "collars" and generic.needs_review() == ["ID"]           # "ID" matches too much to be trusted alone


def test_edits_confirm_a_guess_and_a_template_is_used_as_saved():
    df = pd.DataFrame({"BHID": ["A", "A"], "M_FROM": [0.0, 1], "M_TO": [1.0, 2], "CU": [0.1, 0.2], "NOTE": ["x", "y"]})
    g = guess_table(df, kind="intervals")
    assert g.missing_roles() == ["to"] and g.column("from") == "M_FROM" and "M_FROM" in g.needs_review()   # "contains FROM": weak
    with pytest.raises(ValueError, match="needs a column for To"):
        g.to_spec()
    spec = g.to_spec("assays", edits={"M_FROM": {"role": "from"}, "M_TO": {"role": "to"}, "CU": {"unit": "%"}, "NOTE": {"selected": False}})
    assert spec.roles() == {"hole": "BHID", "from": "M_FROM", "to": "M_TO"} and spec.attributes() == ["CU"]
    assert spec.spec("CU").unit == "%" and spec.spec("CU").source == "user" and spec.spec("BHID").source == "guess"
    again = guess_table(df.iloc[::-1], templates=[spec])                           # same columns, any order: the saved settings
    assert again.template == "assays" and again.is_sure() and again.to_spec().roles() == spec.roles()


# ---- specs ----------------------------------------------------------------------------------------------------------------------


def test_a_spec_types_the_table_keeps_the_names_and_reports_what_it_could_not_read():
    spec = TableSpec("t", "intervals", [ColumnSpec("H", "text", role="hole"), ColumnSpec("F", "number", role="from"), ColumnSpec("T", "number", role="to"),
                                        ColumnSpec("CU", "number"), ColumnSpec("WHEN", "date"), ColumnSpec("FROM", "category")])
    df = pd.DataFrame({"H": [" A ", "B"], "F": ["0", "1,5"], "T": [1.5, 3], "CU": ["0.2", "ND"], "WHEN": ["31/12/2025", "01/01/2026"], "FROM": ["x", "y"]})
    typed, bad = spec.apply(df, report=True)
    assert list(typed.columns) == list(df.columns) and typed["H"].tolist() == ["A", "B"] and typed["F"].tolist() == [0.0, 1.5]
    assert bad == {"CU": (1, ["ND"])} and typed["WHEN"].dt.month.tolist() == [12, 1]
    canon = spec.canonical(typed)
    assert list(canon.columns) == ["HOLEID", "FROM", "TO", "CU", "WHEN", "FROM_original"]   # a value column named like a role keeps apart
    assert spec.check(df.drop(columns="CU").assign(NEW=1)) == {"missing": ["CU"], "new": ["NEW"]}
    assert TableSpec.from_dict(spec.to_dict()) == spec and spec.signature == signature(df.columns)
    with pytest.raises(ValueError, match="both the Hole id"):
        TableSpec("t", "intervals", [ColumnSpec("A", "text", role="hole"), ColumnSpec("B", "text", role="hole")])


# ---- onboarding, validating, saving --------------------------------------------------------------------------------------------


def test_onboarding_stops_on_what_is_unsure_then_validates_saves_and_opens_again(tmp_path):
    collars, surveys, assays = _write_campaign(tmp_path)
    files = ["collar.csv", "survey.csv", "intervals.csv"]
    with pytest.raises(NeedsReview) as e:
        DrillholeSet.onboard(tmp_path, files, "Centinela 2025")
    assert e.value.unsure == {"intervals.csv": ["CU"]} and "'<0.01'" in str(e.value)
    s = DrillholeSet.onboard(tmp_path, files, "Centinela 2025", edits={"intervals.csv": {"CU": {"unit": "%"}, "OBS": {"selected": False}}})
    assert s.tables() == [("collar", "collars"), ("survey", "surveys"), ("intervals", "intervals")]
    assert s.dip_positive_down is False and "negative" in s.dip_note
    assert list(s.intervals["intervals"].columns) == ["SONDAJE", "DESDE", "HASTA", "CU", "LITO"]

    kinds = {(i.kind, i.hole) for i in s.validate()}
    assert ("unparsed_values", None) in kinds and ("hole_id_mismatch", "ddh03") in kinds
    assert ("interval_without_collar", "ddh03") in kinds
    frame = s.issues_frame()
    assert frame["level"].iloc[0] == "warning" and set(frame.columns) == {"level", "kind", "table", "hole", "row", "detail"}

    # desurveyed as the tables are on their own, with English names
    en = interval_table(collars.rename(columns={"SONDAJE": "HOLEID", "ESTE": "X", "NORTE": "Y", "COTA": "Z", "LARGO": "LENGTH"}),
                        surveys.rename(columns={"SONDAJE": "HOLEID", "PROFUNDIDAD": "DEPTH", "AZIMUT": "AZIMUTH", "INCLINACION": "DIP"}),
                        assays.rename(columns={"SONDAJE": "HOLEID", "DESDE": "FROM", "HASTA": "TO"})[["HOLEID", "FROM", "TO"]],
                        dip_positive_down=False, warn=False)
    mine = s.interval_table("intervals", warn=False)
    np.testing.assert_allclose(mine[["TOP_X", "TOP_Y", "TOP_Z"]].to_numpy(), en[["TOP_X", "TOP_Y", "TOP_Z"]].to_numpy())

    path = s.save(tmp_path / "centinela.dhset.yaml")
    text = path.read_text(encoding="utf-8")
    assert "path: intervals.csv" in text and "unit: '%'" in text and "dip_positive_down" not in text   # read from the signs: not saved
    again = DrillholeSet.open(path)
    assert again.tables() == s.tables() and again.review == {}
    pd.testing.assert_frame_equal(again.interval_table("intervals", warn=False), mine)
    assert again.spec_of("intervals").spec("OBS").selected is False

    # a column added to a file is guessed and listed; a column that disappeared stops the open
    pd.read_csv(tmp_path / "intervals.csv").assign(AU_PPM=0.1).to_csv(tmp_path / "intervals.csv", index=False)
    grown = DrillholeSet.open(path)
    assert grown.review == {"intervals": ["AU_PPM"]} and grown.spec_of("intervals").spec("AU_PPM").unit == "ppm"
    pd.read_csv(tmp_path / "intervals.csv").drop(columns="HASTA").to_csv(tmp_path / "intervals.csv", index=False)
    with pytest.raises(SpecMismatch, match="missing HASTA"):
        DrillholeSet.open(path)


def test_interval_problems_are_found():
    collars = pd.DataFrame({"HOLEID": ["A", "A", "B"], "X": 0.0, "Y": 0.0, "Z": 0.0, "LENGTH": 50.0, "AZIMUTH": 0.0, "DIP": 90.0})
    ints = pd.DataFrame({"HOLEID": ["A", "A", "A", "A", "B"], "FROM": [0.0, 8, 20, 30, 5], "TO": [10.0, 15, 25, 30, 60]})
    pts = pd.DataFrame({"HOLEID": ["A", "Z"], "DEPTH": [70.0, 1]})
    s = DrillholeSet("s", collars)
    s.add("ints", ints)
    s.add("pts", pts, "points")
    kinds = [(i.kind, i.hole, i.row) for i in s.validate()]
    for k in [("duplicate_collar", "A", 2), ("from_ge_to", "A", 4), ("overlapping_intervals", "A", 2), ("gap", "A", 3),
              ("interval_beyond_depth", "B", 5), ("point_beyond_depth", "A", 1), ("point_without_collar", "Z", 2)]:
        assert k in kinds, k
    assert [i.level for i in s.validate()][0] == "error"


def test_dip_convention_is_read_from_the_signs():
    assert detect_dip_convention(pd.DataFrame({"HOLEID": ["A", "B"], "DIP": [60.0, 70]}))["positive_down"] is True
    assert detect_dip_convention(pd.DataFrame({"HOLEID": ["A", "B"], "INCL": [-60.0, -70]}))["positive_down"] is False
    assert detect_dip_convention(pd.DataFrame({"HOLEID": ["A"]}))["positive_down"] is None


def test_dates_mixing_text_and_excel_day_numbers_and_empty_columns_need_no_review():
    """As database exports write them (seen in the Centinela collar export)."""
    df = pd.DataFrame({"HOLEID": ["A", "B", "C", "D"], "X": 0.0, "Y": 0.0, "Z": 0.0, "STARTDATE": ["40089", "5-Dec-2016", "41727", "10-Apr-2011"],
                       "TENEMENTID": [np.nan] * 4})
    g = guess_table(df)
    by = {c.name: c for c in g.columns}
    assert by["STARTDATE"].dtype == "date" and by["TENEMENTID"].selected is False and g.is_sure()
    typed = g.to_spec().apply(df)
    assert typed["STARTDATE"].dt.strftime("%Y-%m-%d").tolist() == ["2009-10-03", "2016-12-05", "2014-03-29", "2011-04-10"]
    assert "TENEMENTID" not in typed
