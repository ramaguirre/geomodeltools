import pytest

from geomodeltools import palette as pl


def test_ciede2000_matches_the_published_test_pair():
    # Sharma, Wu, Dalal (2005), pair 1
    d = pl.delta_e2000((50.0, 2.6772, -79.7751), (50.0, 0.0, -82.7485))
    assert d == pytest.approx(2.0425, abs=1e-3)


def test_lab_roundtrip():
    for rgb in [(255, 255, 0), (166, 216, 74), (40, 40, 200), (200, 10, 10)]:
        assert pl.lab_to_rgb(pl.rgb_to_lab(rgb)) == tuple(rgb)


def test_ics_colours_are_valid_hex_and_stage_epoch_order():
    for name, h in pl.ICS.items():
        assert len(h) == 6 and int(h, 16) >= 0, name
    assert pl.ICS["Miocene"] == "FFFF00" and pl.ICS["Late Cretaceous"] == "A6D84A"


def test_family_ramp_is_lighter_when_younger_and_distinguishable():
    ramp = pl.family_ramp(20, 6)
    L = [pl.rgb_to_lab(c)[0] for c in ramp]
    assert L == sorted(L, reverse=True)
    cols = {f"u{i}": c for i, c in enumerate(ramp)}
    assert pl.min_pairwise_de(cols)[0] > 8


def test_spread_lightness_separates_a_group():
    base = {"a": pl.hex_to_rgb("F2FA8C"), "b": pl.hex_to_rgb("E6F47F"), "c": pl.hex_to_rgb("E8F585"), "z": pl.hex_to_rgb("73B2FF")}
    out = pl.spread_lightness(base, ["a", "b", "c"], min_de=8.0)
    assert pl.min_pairwise_de({k: out[k] for k in "abc"})[0] >= 8.0
    assert out["z"] == base["z"]


def test_ramp_about_keeps_hue_and_orders_lightness():
    base = pl.hex_to_rgb("E6F47F")
    ramp = pl.ramp_about(base, 4, span=24)
    labs = [pl.rgb_to_lab(c) for c in ramp]
    assert [x[0] for x in labs] == sorted([x[0] for x in labs], reverse=True)
    assert pl.min_pairwise_de({f"u{i}": c for i, c in enumerate(ramp)})[0] > 4
