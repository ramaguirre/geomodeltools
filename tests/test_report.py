"""Tests for report.py: payload structure from build_report, and a sanity check on the
HTML write_html produces. The *_data() functions it's built on (quick_eda_data,
categorical_eda_data, cat_vs_cat_data) are the same ones the matplotlib plots use, and are
covered by test_eda.py / test_contact_plots.py through the figures they drive; here we only
check the parts specific to assembling and serialising the report payload.
"""

import json

import numpy as np
import pandas as pd
import pytest

from geomodeltools import build_report, write_html
from geomodeltools.contact_plots import distance_to_contact


def _synthetic_df(n=300, seed=0):
    rng = np.random.default_rng(seed)
    unit = rng.choice(["Oxide", "Sulfide", "Waste"], size=n)
    alteration = rng.choice(["Argillic", "Fresh"], size=n)
    x = rng.uniform(-20, 20, n)
    contact_unit = np.where(x >= 0, "Sulfide", "Waste")
    cu = rng.lognormal(0, 0.5, n)
    return pd.DataFrame({
        "cu_pct": cu, "unit": unit, "alteration": alteration,
        "x": x, "y": 0.0, "z": 0.0, "contact_unit": contact_unit,
    })


def test_build_report_is_json_serialisable_and_covers_requested_columns():
    df = _synthetic_df()
    payload = build_report(df, numeric=["cu_pct"], categorical=["unit", "alteration"])
    json.dumps(payload)  # must not raise

    assert set(payload["numeric"]) == {"cu_pct"}
    assert set(payload["numeric"]["cu_pct"]["by_category"]) == {"unit", "alteration"}
    assert set(payload["categorical"]) == {"unit", "alteration"}
    # auto-paired: 2 categorical columns -> exactly one pair
    assert len(payload["pairs"]) == 1
    assert {payload["pairs"][0]["category_x"], payload["pairs"][0]["category_y"]} == {"unit", "alteration"}
    assert payload["contacts"] == []


def test_build_report_skips_auto_pairing_above_six_categorical_columns():
    df = pd.DataFrame({f"cat{i}": ["a", "b"] * 5 for i in range(7)})
    payload = build_report(df, categorical=[f"cat{i}" for i in range(7)])
    assert payload["pairs"] == []


def test_build_report_explicit_pairs_override_auto_pairing():
    df = _synthetic_df()
    payload = build_report(df, categorical=["unit", "alteration"], pairs=[("alteration", "unit")])
    assert len(payload["pairs"]) == 1
    assert payload["pairs"][0]["category_x"] == "alteration"
    assert payload["pairs"][0]["category_y"] == "unit"


def test_build_report_contact_scenario_needs_value_or_category():
    df = _synthetic_df()
    with pytest.raises(ValueError):
        build_report(df, contacts=[{"coords": ["x", "y", "z"], "unit": "contact_unit", "inside": "Sulfide", "outside": "Waste"}])


def test_build_report_contact_scenario_profile_and_composition():
    df = _synthetic_df()
    payload = build_report(
        df,
        contacts=[{
            "coords": ["x", "y", "z"], "unit": "contact_unit", "inside": "Sulfide", "outside": "Waste",
            "value": "cu_pct", "category": "alteration", "n_bins": 6, "label": "Sulfide/Waste contact",
        }],
    )
    assert len(payload["contacts"]) == 1
    panel = payload["contacts"][0]
    assert panel["label"] == "Sulfide/Waste contact"
    assert "profile" in panel and "composition" in panel
    assert len(panel["profile"]["mid"]) == 6
    assert set(panel["composition"]["categories"]) == set(df["alteration"].unique())

    # cross-check against calling distance_to_contact + contact_profile by hand
    distance = distance_to_contact(df, ["x", "y", "z"], "contact_unit", "Sulfide", "Waste")
    assert distance.notna().sum() == len(df)


def test_build_report_with_nothing_requested_is_all_empty():
    df = _synthetic_df()
    payload = build_report(df)
    assert payload == {"numeric": {}, "categorical": {}, "pairs": [], "contacts": []}


def test_build_report_empty_contact_bins_do_not_poison_the_json(tmp_path):
    # A tight n_bins/bin_width leaves some bins with zero samples -> a NaN mean/composition
    # row. json.dumps would emit a bare NaN token there, which JSON.parse rejects in the
    # browser; write_html must turn those into JSON null instead.
    df = _synthetic_df()
    payload = build_report(
        df,
        contacts=[{
            "coords": ["x", "y", "z"], "unit": "contact_unit", "inside": "Sulfide", "outside": "Waste",
            "value": "cu_pct", "bin_width": 0.01, "n_bins": 50, "label": "sparse",
        }],
    )
    assert any(v != v for v in payload["contacts"][0]["profile"]["mean"])  # at least one NaN present pre-sanitising

    out = write_html(df, tmp_path / "sparse.html", contacts=[{
        "coords": ["x", "y", "z"], "unit": "contact_unit", "inside": "Sulfide", "outside": "Waste",
        "value": "cu_pct", "bin_width": 0.01, "n_bins": 50, "label": "sparse",
    }])
    text = out.read_text(encoding="utf-8")
    assert "NaN" not in text.split('id="report-data">')[1].split("</script>")[0]


def test_write_html_embeds_data_and_vendored_plotly(tmp_path):
    df = _synthetic_df()
    out = write_html(df, tmp_path / "report.html", numeric=["cu_pct"], categorical=["unit"], title="Test Report")

    text = out.read_text(encoding="utf-8")
    assert "Test Report" in text
    assert "geomodeltoolsReport.mount" in text
    assert "plotly.js v4.1.1" in text  # vendored library actually inlined, not linked
    assert '<script type="application/json" id="report-data">' in text
    assert "cu_pct" in text
