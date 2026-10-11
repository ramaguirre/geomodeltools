"""Self-contained HTML EDA/contact-plot report: one `write_html(df, path, ...)` call.

Computes everything in Python, through the same `*_data()` functions `quick_eda` and
`contact_plot` use for their matplotlib figures, and embeds the result as JSON in a static
HTML page. Same pattern as geoview's `export.py`: a vendored JS library inlined
(`static/plotly.min.js`), a JSON data blob, and a small "mount" script
(`static/report.js`). No server and no Node needed to open the result — only to edit
`static/report.js` itself, which is plain JS with no build step (see the module docstring
in that file).
"""

from __future__ import annotations

import html
import json
import math
from pathlib import Path

import pandas as pd

from .contacts import contact_composition, contact_profile, distance_to_contact
from .eda import cat_vs_cat_data, categorical_eda_data, quick_eda_data

STATIC = Path(__file__).parent / "static"

TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  body {{ margin: 0; font-family: system-ui, sans-serif; color: #222; }}
  header {{ padding: 12px 20px; background: #1f2d3d; color: white; }}
  header h1 {{ margin: 0; font-size: 1.1rem; font-weight: 600; }}
  nav {{ display: flex; gap: 4px; padding: 8px 20px 0; background: #f4f5f7; border-bottom: 1px solid #ddd; }}
  nav button {{ border: none; background: none; padding: 8px 14px; cursor: pointer; font-size: 0.9rem;
                border-bottom: 2px solid transparent; }}
  nav button.active {{ border-bottom-color: #1f2d3d; font-weight: 600; }}
  main {{ padding: 16px 20px; }}
  .panel {{ display: none; }}
  .panel.active {{ display: block; }}
  .controls {{ display: flex; gap: 16px; flex-wrap: wrap; align-items: center; margin-bottom: 12px; }}
  .controls label {{ font-size: 0.85rem; color: #444; display: flex; gap: 6px; align-items: center; }}
  select {{ padding: 4px 6px; }}
  .charts {{ display: flex; flex-wrap: wrap; gap: 8px; }}
  .chart {{ flex: 1 1 320px; min-width: 300px; height: 380px; }}
  .empty {{ color: #888; font-style: italic; padding: 24px 0; }}
</style>
</head>
<body>
<header><h1>{title}</h1></header>
<nav id="report-nav"></nav>
<main id="report-main"></main>
<script type="application/json" id="report-data">{data}</script>
<script>{plotly}</script>
<script>{app}</script>
<script>
  geomodeltoolsReport.mount(
    document.getElementById("report-nav"),
    document.getElementById("report-main"),
    JSON.parse(document.getElementById("report-data").textContent)
  );
</script>
</body>
</html>
"""


def _safe(text: str) -> str:
    # an inline <script> ends at the first "</script", wherever it appears
    return text.replace("</", r"<\/")


def _nan_to_none(value):
    """Replace NaN/Infinity with None, recursively: `json.dumps` writes bare `NaN`/`Infinity`
    tokens for these by default, which `JSON.parse` in the browser rejects as invalid JSON.
    An empty contact-profile bin, for one, has a NaN mean.
    """
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, dict):
        return {k: _nan_to_none(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_nan_to_none(v) for v in value]
    return value


def build_report(
    df: pd.DataFrame,
    *,
    numeric: list[str] = (),
    categorical: list[str] = (),
    pairs: list[tuple[str, str]] | None = None,
    contacts: list[dict] | None = None,
    weight: str | None = None,
) -> dict:
    """Assemble the JSON payload consumed by `static/report.js`.

    Parameters:
        numeric: numeric columns to include. Each gets an overall quick-EDA panel, plus one
            by-category breakdown per column in `categorical` that has overlapping data.
        categorical: categorical columns to include. Each gets a frequency panel.
        pairs: `(category_x, category_y)` pairs for a composition panel. Defaults to every
            pair within `categorical` (skipped if `categorical` has more than 6 columns —
            pass `pairs` explicitly for a larger set).
        contacts: list of dicts, each describing one contact-plot scenario: `coords`,
            `unit`, `inside`, `outside`, and optionally `contact`, `bin_width`, `n_bins`,
            `weight` (overrides the top-level `weight` for this scenario), `label`, and
            `value` and/or `category` (computes whichever of those two are given — at
            least one is required).
        weight: optional column of per-row weights, used everywhere unless a contact
            scenario names its own `weight`.

    Returns:
        A JSON-ready dict: `{"numeric": {...}, "categorical": {...}, "pairs": [...],
        "contacts": [...]}`.
    """
    numeric, categorical = list(numeric), list(categorical)
    if pairs is None:
        pairs = [(a, b) for i, a in enumerate(categorical) for b in categorical[i + 1 :]] if len(categorical) <= 6 else []

    numeric_panels = {}
    for value in numeric:
        by_category = {}
        for cat in categorical:
            try:
                by_category[cat] = quick_eda_data(df, value, category=cat, weight=weight)
            except ValueError:
                continue  # no rows with both value and this category set
        numeric_panels[value] = {"overall": quick_eda_data(df, value, weight=weight), "by_category": by_category}

    categorical_panels = {cat: categorical_eda_data(df, cat, weight=weight) for cat in categorical}
    pair_panels = [cat_vs_cat_data(df, x, y, weight=weight) for x, y in pairs]

    contact_panels = []
    for spec in contacts or []:
        spec = dict(spec)
        scenario_weight = spec.pop("weight", weight)
        value = spec.pop("value", None)
        category = spec.pop("category", None)
        bin_width = spec.pop("bin_width", None)
        n_bins = spec.pop("n_bins", 10)
        label = spec.pop("label", None) or f"{spec['inside']} / {spec['outside']}"
        if value is None and category is None:
            raise ValueError(f"Contact scenario {label!r} needs `value` and/or `category`.")

        distance = distance_to_contact(df, spec["coords"], spec["unit"], spec["inside"], spec["outside"], contact=spec.get("contact"))
        panel = {"label": label, "unit": spec["unit"], "inside": spec["inside"], "outside": spec["outside"]}

        if value is not None:
            table = contact_profile(df, distance, value, bin_width=bin_width, n_bins=n_bins, weight=scenario_weight)
            panel["profile"] = {"value": value, "mid": table["mid"].tolist(), "mean": table["mean"].tolist(), "n": table["n"].tolist()}
        if category is not None:
            table = contact_composition(df, distance, category, bin_width=bin_width, n_bins=n_bins, weight=scenario_weight)
            cats = [c for c in table.columns if c not in ("mid", "n")]
            panel["composition"] = {
                "category": category,
                "mid": table["mid"].tolist(),
                "n": table["n"].tolist(),
                "categories": [str(c) for c in cats],
                "matrix": table[cats].to_numpy().tolist(),
            }
        contact_panels.append(panel)

    return {"numeric": numeric_panels, "categorical": categorical_panels, "pairs": pair_panels, "contacts": contact_panels}


def write_html(df: pd.DataFrame, path, *, title: str = "geomodeltools — EDA report", **kwargs) -> Path:
    """Write a self-contained, interactive HTML EDA/contact-plot report for `df`.

    `**kwargs` (`numeric`, `categorical`, `pairs`, `contacts`, `weight`) are passed to
    `build_report`. The result is one HTML file with everything inlined (data, Plotly,
    the small app script): no server, no network access, and no Node needed to open it.
    """
    payload = _nan_to_none(build_report(df, **kwargs))
    plotly = (STATIC / "plotly.min.js").read_text(encoding="utf-8")
    app = (STATIC / "report.js").read_text(encoding="utf-8")
    page = TEMPLATE.format(title=html.escape(title), data=_safe(json.dumps(payload)), plotly=_safe(plotly), app=_safe(app))

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page, encoding="utf-8")
    return path
