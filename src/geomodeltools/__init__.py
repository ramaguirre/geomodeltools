"""geomodeltools: helpers for 3D geological modelling.

Names are loaded on first use, so `import geomodeltools` is instant and `from geomodeltools import desurvey_intervals` needs only
numpy and pandas. A name that needs an optional package (geopandas, shapely, rasterio and requests for the geometry and DEM tools;
scipy and matplotlib for the plots) says which extra to install: `pip install geomodeltools[geo]`, `[plots]` or `[all]`.
"""
from __future__ import annotations

import importlib

# public name -> the module that defines it
_LAZY = {
    "bufferize_2d_polygons": "geometry", "clean_polygon_geometries": "geometry", "densify_geometries": "geometry",
    "geometries_to_points": "geometry",
    "add_z_from_opentopography": "dem", "download_opentopography_dem": "dem",
    "cim_color_to_rgb": "colour", "cim_symbol_color": "colour", "leapfrog_colour_palette": "colour",
    "leapfrog_colour2dictionary": "colour", "lfc2dict": "colour", "read_mapx_symbology": "colour", "symbology_colors": "colour",
    "cat_vs_cat": "eda", "cat_vs_cat_data": "eda", "categorical_eda": "eda", "categorical_eda_data": "eda",
    "category_summary": "eda", "quick_eda": "eda", "quick_eda_data": "eda",
    "contact_boxplot_bins": "contact_plots", "contact_composition": "contact_plots", "contact_plot": "contact_plots",
    "contact_profile": "contact_plots", "distance_to_contact": "contact_plots", "plot_contact_boxplot": "contact_plots",
    "plot_contact_composition": "contact_plots", "plot_contact_profile": "contact_plots",
    "build_report": "report", "write_html": "report",
    "plot_soft_boundary_boxplots": "soft_boundary", "soft_boundary_pairs": "soft_boundary", "soft_boundary_values": "soft_boundary",
    "Desurveyed": "desurvey", "desurvey_intervals": "desurvey",
    "latlon_to_xy": "collars", "block_model_bounds": "collars", "dem_for_block_model": "collars", "sample_dem_z": "collars",
    "collars_from_latlon": "collars",
}
_EXTRA = {"geopandas": "geo", "shapely": "geo", "rasterio": "geo", "requests": "geo", "scipy": "plots", "matplotlib": "plots"}

__all__ = sorted(_LAZY)


def __getattr__(name: str):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module 'geomodeltools' has no attribute {name!r}")
    try:
        value = getattr(importlib.import_module(f".{module}", __name__), name)
    except ModuleNotFoundError as e:
        extra = _EXTRA.get((e.name or "").split(".")[0])
        if extra is None:
            raise
        raise ImportError(f"geomodeltools.{name} needs the optional package {e.name!r}: "
                          f"pip install geomodeltools[{extra}] (or [all])") from e
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY))
