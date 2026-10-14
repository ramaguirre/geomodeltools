"""Regenerate the desurvey golden data from the Python reference (`geomodeltools.desurvey.desurvey_intervals`).

    python generate.py

Writes the synthetic collar / survey / interval tables (collars.csv, surveys.csv, intervals.csv, and the same with dips negative-down:
collars_neg.csv, surveys_neg.csv), cases.json, expected/<case>.json and MANIFEST.json (SHA-256 of every file, line endings
normalised). The numbers come from Python; the browser tool `geoplots_local` must reproduce them from the CSV files alone.
Run it only when the contract changes and review the diff. Synthetic data only.

The tables are built to hit every rule: curved holes (positive-down dips, one up-hole), a hole whose survey is not in depth order, one
with a repeated survey depth, one with no survey but its own azimuth/dip on the collar, one with neither, survey rows and intervals of
a hole that has no collar, an interval that ends below its hole, and a hole with no intervals.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import geomodeltools
from geomodeltools.desurvey import desurvey_intervals

HERE = Path(__file__).resolve().parent
CONTRACT_VERSION = 1

COLS = {"hole": "HOLEID", "xyz": ["X", "Y", "Z"], "length": "DEPTH", "at": "AT", "azimuth": "AZIMUTH", "dip": "DIP",
        "from_": "FROM", "to": "TO"}
CASES = {
    "positive_down_skip": {"tables": "", "dip_positive_down": True, "duplicates": "skip"},
    "duplicates_first": {"tables": "", "dip_positive_down": True, "duplicates": "first"},
    "duplicates_last": {"tables": "", "dip_positive_down": True, "duplicates": "last"},
    "negative_down_skip": {"tables": "_neg", "dip_positive_down": False, "duplicates": "skip"},
    "traces_every_25": {"tables": "", "dip_positive_down": True, "duplicates": "skip", "intervals": False, "trace_step": 25.0},
}


def make_tables(seed=11):
    rng = np.random.default_rng(seed)
    collars, surveys, intervals = [], [], []
    for k in range(1, 10):
        hid = f"H{k:02d}"
        length = float(rng.integers(90, 190))
        az = rng.uniform(0, 360)
        dip = -20.0 if hid == "H08" else rng.uniform(55, 80)           # H08 points up (positive-down table)
        collars.append({"HOLEID": hid, "X": rng.uniform(1000, 1500), "Y": rng.uniform(5000, 5400), "Z": rng.uniform(200, 300),
                        "DEPTH": length, "AZIMUTH": np.nan, "DIP": np.nan})
        depths = np.arange(0, length + 1, 40.0)
        rows = []
        for d in depths:
            rows.append({"HOLEID": hid, "AT": float(d), "AZIMUTH": az % 360, "DIP": float(np.clip(dip, -89, 89))})
            az += rng.normal(0, 3)
            dip += rng.normal(0, 1.5)
        surveys += rows
    # H05: survey rows out of depth order
    idx = [i for i, r in enumerate(surveys) if r["HOLEID"] == "H05"]
    block = [surveys[i] for i in idx]
    block = [block[2], block[0], block[3], block[1]] + block[4:]
    for i, r in zip(idx, block):
        surveys[i] = r
    # H06: a repeated survey depth (two different directions at 40 m)
    i40 = next(i for i, r in enumerate(surveys) if r["HOLEID"] == "H06" and r["AT"] == 40.0)
    surveys.insert(i40 + 1, dict(surveys[i40], AZIMUTH=(surveys[i40]["AZIMUTH"] + 25) % 360, DIP=surveys[i40]["DIP"] - 6))
    # H10: no survey, but its own direction on the collar; H11: no survey and no direction (left out)
    collars.append({"HOLEID": "H10", "X": 1250.0, "Y": 5200.0, "Z": 250.0, "DEPTH": 60.0, "AZIMUTH": 135.0, "DIP": 45.0})
    collars.append({"HOLEID": "H11", "X": 1300.0, "Y": 5250.0, "Z": 255.0, "DEPTH": 60.0, "AZIMUTH": np.nan, "DIP": np.nan})
    # H12: surveyed, but no intervals at all
    collars.append({"HOLEID": "H12", "X": 1350.0, "Y": 5300.0, "Z": 260.0, "DEPTH": 80.0, "AZIMUTH": np.nan, "DIP": np.nan})
    surveys += [{"HOLEID": "H12", "AT": 0.0, "AZIMUTH": 200.0, "DIP": 70.0}, {"HOLEID": "H12", "AT": 80.0, "AZIMUTH": 205.0, "DIP": 66.0}]
    surveys += [{"HOLEID": "H99", "AT": 0.0, "AZIMUTH": 10.0, "DIP": 60.0}, {"HOLEID": "H99", "AT": 30.0, "AZIMUTH": 12.0, "DIP": 62.0}]
    for c in collars:
        if c["HOLEID"] == "H12":
            continue
        edges = np.unique(np.concatenate([np.arange(0, c["DEPTH"], 2.0), [c["DEPTH"]]]))
        # irregular lengths: drop every third cut
        edges = np.concatenate([edges[:1], edges[1:-1][np.arange(len(edges) - 2) % 3 != 2], edges[-1:]])
        for f, t in zip(edges[:-1], edges[1:]):
            intervals.append({"HOLEID": c["HOLEID"], "FROM": float(f), "TO": float(t),
                              "cu": float(np.exp(rng.normal(-1, 0.5))), "lith": rng.choice(["A", "B", "C"])})
    intervals.append({"HOLEID": "H99", "FROM": 0.0, "TO": 5.0, "cu": 0.3, "lith": "A"})
    intervals.append({"HOLEID": "H03", "FROM": 500.0, "TO": 505.0, "cu": 0.2, "lith": "B"})      # ends below the hole
    return pd.DataFrame(collars), pd.DataFrame(surveys), pd.DataFrame(intervals)


def write_csv(df, name):
    df.to_csv(HERE / name, index=False, lineterminator="\n", float_format="%.10g", na_rep="")


def records(report: pd.DataFrame) -> list:
    out = []
    for r in report.to_dict("records"):
        rec = {}
        for k, v in r.items():
            if k in ("row", "conflicting_row"):
                rec[k] = None if pd.isna(v) else int(v)
            elif k == "depth":
                rec[k] = None if pd.isna(v) else float(v)
            else:
                rec[k] = v
        out.append(rec)
    return out


def main() -> None:
    collars, surveys, intervals = make_tables()
    write_csv(collars, "collars.csv")
    write_csv(surveys, "surveys.csv")
    write_csv(intervals, "intervals.csv")
    write_csv(collars.assign(DIP=-collars["DIP"]), "collars_neg.csv")
    write_csv(surveys.assign(DIP=-surveys["DIP"]), "surveys_neg.csv")
    (HERE / "expected").mkdir(exist_ok=True)
    cases = {}
    for name, c in CASES.items():
        sfx = c["tables"]
        co, su, iv = (pd.read_csv(HERE / f"collars{sfx}.csv"), pd.read_csv(HERE / f"surveys{sfx}.csv"),
                      pd.read_csv(HERE / "intervals.csv"))
        kw = dict(COLS, dip_positive_down=c["dip_positive_down"], duplicates=c["duplicates"])
        if "trace_step" in c:
            kw["trace_step"] = c["trace_step"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = desurvey_intervals(co, su, iv if c.get("intervals", True) else None, warn=False, **kw)
        pos = [f"{p}_{a}" for p in ("start", "mid", "end") for a in "xyz"]
        table = res.intervals
        keep = ["HOLEID", "FROM", "TO", "_row", *pos]
        expected = {"intervals": {k: table[k].tolist() for k in keep},
                    "report": records(res.report), "skipped": {str(k): v for k, v in res.skipped.items()}}
        (HERE / "expected" / f"{name}.json").write_text(json.dumps(expected, separators=(",", ":")), encoding="utf-8")
        cases[name] = {"collars": f"collars{sfx}.csv", "surveys": f"surveys{sfx}.csv",
                       "intervals": "intervals.csv" if c.get("intervals", True) else None, "columns": COLS,
                       "dip_positive_down": c["dip_positive_down"], "duplicates": c["duplicates"],
                       "trace_step": c.get("trace_step", 10.0)}
    (HERE / "cases.json").write_text(json.dumps({"cases": cases}, indent=1), encoding="utf-8")
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
