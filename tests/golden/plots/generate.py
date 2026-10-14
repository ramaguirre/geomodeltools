"""Regenerate the EDA / contact-plot golden data from the Python reference (`geomodeltools.report.build_report`).

    python generate.py

Writes data.csv (synthetic), cases.json, expected/<case>.json (the payload `build_report` returns, NaN as null) and MANIFEST.json
(SHA-256 of every file, line endings normalised). Here the numbers come from Python, and the browser tool `geoplots_local`
must reproduce them from data.csv. Run it only when the contract changes and review the diff. Synthetic data only.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

import geomodeltools
from geomodeltools import report

HERE = Path(__file__).resolve().parent
CONTRACT_VERSION = 1

CASES = {
    "eda_weighted": {"numeric": ["cu", "au"], "categorical": ["alt", "zone"], "weight": "len"},
    "eda_unweighted": {"numeric": ["cu"], "categorical": ["lith", "alt"], "weight": None},
    "contact_default": {"numeric": [], "categorical": [], "weight": "len", "contacts": [
        {"coords": ["x", "y", "z"], "unit": "lith", "inside": "A", "outside": "B", "contact": "D", "value": "cu", "category": "alt"}]},
    "contact_options": {"numeric": [], "categorical": [], "weight": None, "contacts": [
        {"coords": ["x", "y", "z"], "unit": "lith", "inside": ["A"], "outside": ["B", "C"], "bin_width": 5, "n_bins": 8,
         "value": "au", "category": "zone", "label": "custom"}]},
}


def make_table(n=600, seed=7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x, y, z = rng.uniform(0, 200, n), rng.uniform(0, 100, n), rng.uniform(0, 50, n)
    lith = np.where(x < 60, "A", np.where(x < 64, "D", np.where(x < 130, "B", "C")))
    alt = rng.choice(["pot", "fil", "arg", "prop"], n, p=[0.4, 0.3, 0.2, 0.1]).astype(object)
    alt[rng.random(n) < 0.05] = None
    zone = rng.choice([1, 2, 3, 10], n).astype(float)
    zone[rng.random(n) < 0.04] = np.nan
    cu = np.exp(rng.normal(-1.0 + 0.01 * (130 - np.abs(x - 62)) / 10, 0.6, n))
    au = np.exp(rng.normal(-2.0, 0.5, n)) + 0.001 * x
    cu[rng.random(n) < 0.03] = np.nan
    au[rng.random(n) < 0.03] = np.nan
    df = pd.DataFrame({"x": x, "y": y, "z": z, "lith": lith, "alt": alt, "zone": zone, "cu": cu, "au": au,
                       "len": rng.choice([0.5, 1.0, 1.5, 2.0, 3.0], n) + rng.uniform(0, 0.01, n)})
    return df


def main() -> None:
    df = make_table()
    csv = HERE / "data.csv"
    df.to_csv(csv, index=False, lineterminator="\n", float_format="%.10g", na_rep="")
    # a few cells written as NA text, which the contract also treats as missing
    text = csv.read_text(encoding="utf-8").split("\n")
    for i in (5, 17, 40):
        parts = text[i].split(",")
        parts[6] = "NA"
        text[i] = ",".join(parts)
    csv.write_text("\n".join(text), encoding="utf-8", newline="")
    table = pd.read_csv(csv)
    (HERE / "expected").mkdir(exist_ok=True)
    cases = {}
    for name, kw in CASES.items():
        payload = report._nan_to_none(report.build_report(table, **kw))
        (HERE / "expected" / f"{name}.json").write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        cases[name] = kw
    (HERE / "cases.json").write_text(json.dumps({"data": "data.csv", "cases": cases}, indent=1), encoding="utf-8")
    commit = subprocess.run(["git", "-C", str(Path(geomodeltools.__file__).resolve().parents[2]), "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    files = sorted(p for p in HERE.rglob("*") if p.is_file() and p.name not in ("MANIFEST.json", "generate.py") and "__pycache__" not in p.parts)
    (HERE / "MANIFEST.json").write_text(json.dumps({
        "contract_version": CONTRACT_VERSION, "source": "geomodeltools (Python reference)", "geomodeltools_commit": commit,
        "sha256": {str(p.relative_to(HERE)).replace("\\", "/"): hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest() for p in files}},
        indent=1), encoding="utf-8")
    print("written", len(files), "files; geomodeltools", commit)


if __name__ == "__main__":
    main()
