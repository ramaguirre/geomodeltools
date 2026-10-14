"""The EDA / contact-plot payload against its golden data (contract version 1).

tests/golden/plots/ holds a synthetic table and the payloads `report.build_report` produced from it. This test keeps the Python
reference from drifting; the browser tool `geoplots_local` is checked against the same files and must reproduce them from
the CSV alone. See docs/plots_contract.md.
"""
import hashlib
import json
import math
from pathlib import Path

import pandas as pd
import pytest

from geomodeltools import report

GOLDEN = Path(__file__).parent / "golden" / "plots"
SPEC = json.loads((GOLDEN / "cases.json").read_text(encoding="utf-8"))
RTOL, ATOL = 1e-12, 1e-13


def _compare(got, want, path="payload"):
    if isinstance(want, dict):
        assert isinstance(got, dict) and list(got) == list(want), f"{path}: keys {list(got)[:6]} != {list(want)[:6]}"
        for k in want:
            _compare(got[k], want[k], f"{path}.{k}")
    elif isinstance(want, list):
        assert isinstance(got, list) and len(got) == len(want), f"{path}: length"
        for i, (g, w) in enumerate(zip(got, want)):
            _compare(g, w, f"{path}[{i}]")
    elif isinstance(want, float):
        assert isinstance(got, (int, float)) and math.isclose(got, want, rel_tol=RTOL, abs_tol=ATOL), f"{path}: {got} != {want}"
    else:
        assert got == want, f"{path}: {got!r} != {want!r}"


def test_golden_files_match_their_manifest():
    man = json.loads((GOLDEN / "MANIFEST.json").read_text(encoding="utf-8"))
    for rel, digest in man["sha256"].items():
        data = (GOLDEN / rel).read_bytes().replace(b"\r\n", b"\n")
        assert hashlib.sha256(data).hexdigest() == digest, rel


@pytest.mark.parametrize("case", list(SPEC["cases"]))
def test_python_reference_reproduces_the_golden_payload(case):
    table = pd.read_csv(GOLDEN / SPEC["data"])
    got = json.loads(json.dumps(report._nan_to_none(report.build_report(table, **SPEC["cases"][case]))))
    _compare(got, json.loads((GOLDEN / "expected" / f"{case}.json").read_text(encoding="utf-8")))
