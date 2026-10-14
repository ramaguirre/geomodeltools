"""The Python swath calculation against the golden data produced by the JavaScript engine of swath_local (contract version 1).

The same expected files are what swath_local's build gate compares its own engine to, so this test is what keeps the two
implementations in agreement. See tests/golden/swath/ (MANIFEST.json says which swath_local commit produced the numbers).
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from geomodeltools import swath as sw

GOLDEN = Path(__file__).parent / "golden" / "swath"
CASES = json.loads((GOLDEN / "cases.json").read_text(encoding="utf-8"))
RTOL, ATOL = 1e-12, 1e-13      # relative, plus absolute for differences of nearly equal grades (cancellation)


def _num(x):
    return np.nan if x is None else x


def _same(got: pd.Series, want, exact=False):
    want = np.array([_num(v) for v in want], dtype=float)
    got = got.to_numpy(dtype=float)
    assert len(got) == len(want)
    if exact:
        np.testing.assert_array_equal(got, want)
    else:
        np.testing.assert_allclose(got, want, rtol=RTOL, atol=ATOL, equal_nan=True)


def test_golden_files_match_their_manifest():
    man = json.loads((GOLDEN / "MANIFEST.json").read_text(encoding="utf-8"))
    assert man["contract_version"] == sw.CONTRACT_VERSION
    for rel, digest in man["sha256"].items():
        data = (GOLDEN / rel).read_bytes().replace(b"\r\n", b"\n")      # line endings differ between checkouts
        assert hashlib.sha256(data).hexdigest() == digest, rel


@pytest.mark.parametrize("case", list(CASES))
def test_python_matches_javascript(case):
    spec = CASES[case]
    p, d = spec["params"], GOLDEN / spec["data"]
    want = json.loads((GOLDEN / "expected" / f"{case}.json").read_text(encoding="utf-8"))
    grid = sw.Grid.from_header((d / p["header_file"]).read_text(), p["multiplier"])
    r = sw.swath_compare(pd.read_csv(d / p["blocks_file"]), pd.read_csv(d / p["samples_file"]), grid, p["blocks"], p["samples"],
                         value_min=p["value_min"], value_max=p["value_max"], min_samples=p["min_samples"],
                         min_blocks=p["min_blocks"], ue_filter=p.get("ue_filter"))
    assert list(grid.shape) == want["grid"]["shape"]
    np.testing.assert_allclose(grid.corner, want["grid"]["corner"], atol=1e-9)
    np.testing.assert_allclose(grid.size, want["grid"]["size"], rtol=1e-12)
    assert r.qa == want["qa"]
    assert list(r.summary["ue"]) == [row["ue"] for row in want["summary"]]
    for got, row in zip(r.summary.to_dict("records"), want["summary"]):
        assert got["estado"] == row["estado"] and got["paneles"] == row["paneles"]
        for k, v in row.items():
            if k not in ("ue", "estado", "paneles"):
                np.testing.assert_allclose(got[k], _num(v), rtol=RTOL, atol=ATOL, equal_nan=True)
    assert sorted(r.panels, key=sw.ue_sort_key) == sorted(want["panels"], key=sw.ue_sort_key)
    for ue, cols in want["panels"].items():
        for name, vals in cols.items():
            _same(r.panels[ue][name], vals, exact=name in ("panel", "i", "j", "k", "n_bloques", "n_muestras"))
        for ax, cols in want["swaths"][ue].items():
            for name, vals in cols.items():
                _same(r.swaths[ue][ax][name], vals, exact=name in ("indice", "n_bloques", "n_muestras", "paneles"))


def test_ue_key_normalises_codes():
    assert [sw.ue_key(v) for v in (22, "22", "22.0", " 22 ", 3.5, "AB", "", "NA", float("nan"))] == ["22", "22", "22", "22", "3.5", "AB", None, None, None]
