"""Tests for the ArcGIS CIM colour and .mapx symbology readers (no ArcGIS needed: tiny JSON fixtures)."""

import json

import pandas as pd
import pytest

from geomodeltools import cim_color_to_rgb, cim_symbol_color, read_mapx_symbology, symbology_colors


def rgb(r, g, b):
    return {"type": "CIMRGBColor", "values": [r, g, b, 100]}


def fill(color):
    return {"type": "CIMSolidFill", "enable": True, "color": color}


def stroke(color, width=1):
    return {"type": "CIMSolidStroke", "enable": True, "width": width, "color": color}


def test_grey_and_lab_colours():
    assert cim_color_to_rgb({"type": "CIMGrayColor", "values": [50, 100]}) == (128, 128, 128)
    assert cim_color_to_rgb({"type": "CIMLABColor", "values": [100, 0, 0]}) == (255, 255, 255)
    assert cim_color_to_rgb({"type": "CIMLABColor", "values": [0, 0, 0]}) == (0, 0, 0)


def test_unknown_colour_type_still_raises():
    with pytest.raises(ValueError):
        cim_color_to_rgb({"type": "CIMSpotColor", "values": [1]})


def test_polygon_prefers_fill_over_outline_and_skips_disabled_layers():
    sym = {"symbolLayers": [stroke(rgb(0, 0, 0)), {**fill(rgb(9, 9, 9)), "enable": False}, fill(rgb(10, 20, 30))]}
    assert cim_symbol_color(sym, "polygon") == (10, 20, 30)


def test_polygon_without_fill_uses_hatch_line_then_outline():
    hatch = {"type": "CIMHatchFill", "lineSymbol": {"symbolLayers": [stroke(rgb(1, 2, 3))]}}
    assert cim_symbol_color({"symbolLayers": [hatch, stroke(rgb(9, 9, 9))]}, "polygon") == (1, 2, 3)
    assert cim_symbol_color({"symbolLayers": [stroke(rgb(9, 9, 9))]}, "polygon") == (9, 9, 9)


def test_pattern_fill_takes_the_colour_replacing_white():
    pic = {"type": "CIMPictureFill", "colorSubstitutions": [{"oldColor": rgb(255, 255, 255), "newColor": rgb(5, 6, 7)}]}
    assert cim_symbol_color({"symbolLayers": [pic]}, "polygon") == (5, 6, 7)


def test_line_uses_stroke_and_falls_back_to_marker_colour():
    assert cim_symbol_color({"symbol": {"symbolLayers": [stroke(rgb(7, 8, 9))]}}, "line") == (7, 8, 9)
    marker = {"type": "CIMVectorMarker", "markerGraphics": [{"symbol": {"symbolLayers": [fill(rgb(4, 5, 6))]}}]}
    assert cim_symbol_color({"symbolLayers": [marker]}, "line") == (4, 5, 6)


def _mapx(tmp_path, layers):
    path = tmp_path / "map.mapx"
    path.write_text(json.dumps({"layerDefinitions": layers}), encoding="utf-8")
    return path


def _layer(name, dataset, renderer):
    return {"name": name, "featureTable": {"dataConnection": {"dataset": dataset}}, "renderer": renderer}


UNIQUE = {
    "type": "CIMUniqueValueRenderer",
    "fields": ["unit"],
    "groups": [{"classes": [
        {"values": [{"fieldValues": ["A"]}], "symbol": {"symbol": {"symbolLayers": [fill(rgb(255, 0, 0))]}}},
        {"values": [{"fieldValues": [""]}], "symbol": {"symbol": {"symbolLayers": [fill(rgb(0, 255, 0))]}}},
    ]}],
    "defaultSymbol": {"symbol": {"symbolLayers": [fill(rgb(1, 1, 1))]}},
}
SIMPLE = {"type": "CIMSimpleRenderer", "symbol": {"symbol": {"symbolLayers": [stroke(rgb(2, 2, 2))]}}}


def test_read_mapx_prefers_unique_value_layer_and_reports_unknown_dataset(tmp_path):
    mapx = _mapx(tmp_path, [_layer("plain", "contacts", SIMPLE), _layer("by unit", "contacts", UNIQUE)])
    sym = read_mapx_symbology(mapx, "contacts", "polygon")
    assert sym["layer"] == "by unit" and sym["fields"] == ["unit"]
    assert sym["classes"][("A",)] == (255, 0, 0) and sym["default"] == (1, 1, 1)
    assert read_mapx_symbology(mapx, "missing", "polygon") is None


def test_simple_renderer_has_no_fields(tmp_path):
    sym = read_mapx_symbology(_mapx(tmp_path, [_layer("faults", "faults", SIMPLE)]), "faults", "line")
    assert sym["fields"] == [] and sym["default"] == (2, 2, 2)


def test_symbology_colors_by_class_blank_missing_and_default(tmp_path):
    sym = read_mapx_symbology(_mapx(tmp_path, [_layer("u", "u", UNIQUE)]), "u", "polygon")
    df = pd.DataFrame({"unit": ["A", " A ", "", None, "Z"]})
    assert symbology_colors(df, sym) == [(255, 0, 0), (255, 0, 0), (0, 255, 0), (0, 255, 0), (1, 1, 1)]
    assert symbology_colors(df, None) == []
    simple = read_mapx_symbology(_mapx(tmp_path, [_layer("f", "f", SIMPLE)]), "f", "line")
    assert symbology_colors(df, simple) == [(2, 2, 2)] * 5
