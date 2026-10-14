"""The desurvey golden data (contract version 1): Python keeps reproducing its own reference, and the files match their manifest.

tests/golden/desurvey/ holds synthetic collar / survey / interval tables and what `desurvey_intervals` returned for them. The browser
tool `geoplots_local` reproduces the same numbers in JavaScript from the CSV files alone. See docs/desurvey_contract.md.
"""
import hashlib
import json
import math
import warnings
from pathlib import Path

import pandas as pd
import pytest

from geomodeltools.desurvey import desurvey_intervals

GOLDEN = Path(__file__).parent / "golden" / "desurvey"
CASES = json.loads((GOLDEN / "cases.json").read_text(encoding="utf-8"))["cases"]
ATOL, RTOL = 1e-8, 1e-12      # metres: positions are of the order of 1e3 to 1e4


def test_golden_files_match_their_manifest():
    man = json.loads((GOLDEN / "MANIFEST.json").read_text(encoding="utf-8"))
    for rel, digest in man["sha256"].items():
        data = (GOLDEN / rel).read_bytes().replace(b"\r\n", b"\n")
        assert hashlib.sha256(data).hexdigest() == digest, rel


@pytest.mark.parametrize("case", list(CASES))
def test_python_reproduces_the_golden_positions(case):
    spec = CASES[case]
    c = spec["columns"]
    iv = pd.read_csv(GOLDEN / spec["intervals"]) if spec["intervals"] else None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = desurvey_intervals(pd.read_csv(GOLDEN / spec["collars"]), pd.read_csv(GOLDEN / spec["surveys"]), iv, warn=False,
                                 hole=c["hole"], xyz=c["xyz"], length=c["length"], at=c["at"], azimuth=c["azimuth"], dip=c["dip"],
                                 from_=c["from_"], to=c["to"], dip_positive_down=spec["dip_positive_down"],
                                 duplicates=spec["duplicates"], trace_step=spec["trace_step"])
    want = json.loads((GOLDEN / "expected" / f"{case}.json").read_text(encoding="utf-8"))
    assert {k: v for k, v in res.skipped.items()} == want["skipped"]
    assert len(res.report) == len(want["report"])
    for got, w in zip(res.report.to_dict("records"), want["report"]):
        for k in ("type", "kind", "hole", "table", "message"):
            assert got[k] == w[k], (k, got, w)
    table = res.intervals
    for k, vals in want["intervals"].items():
        got = table[k].tolist()
        assert len(got) == len(vals), k
        for g, v in zip(got, vals):
            if isinstance(v, float):
                assert math.isclose(g, v, rel_tol=RTOL, abs_tol=ATOL), (k, g, v)
            else:
                assert g == v, (k, g, v)


def test_positive_and_negative_down_tables_give_the_same_positions():
    a = json.loads((GOLDEN / "expected" / "positive_down_skip.json").read_text(encoding="utf-8"))["intervals"]
    b = json.loads((GOLDEN / "expected" / "negative_down_skip.json").read_text(encoding="utf-8"))["intervals"]
    for k in a:
        for x, y in zip(a[k], b[k]):
            assert x == y or math.isclose(x, y, abs_tol=1e-9), k
