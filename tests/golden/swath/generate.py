"""Regenerate the swath golden data from the JavaScript engine of `swath_local` (the validated source of the numbers).

    <python with numpy, pandas and nodejs-wheel> generate.py <path to the swath_local clone>

Writes data_*/ (synthetic model, composites and header), cases.json and expected/<case>.json, then MANIFEST.json (SHA-256 of every
file plus the swath_local commit and engine version). Run it only when the contract changes, and review the diff: the expected files
are the agreement between the JavaScript and the Python implementations. Synthetic data only: no real model belongs here.
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SL = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(SL))
from tests.conftest import make_dataset, params_for  # noqa: E402

DATA = {"unrotated": 90.0, "rotated": 65.0}
CASES = {
    "unrotated_default": ("unrotated", {}),
    "rotated_default": ("rotated", {}),
    "rotated_options": ("rotated", {"min_samples": 3, "min_blocks": 2, "value_max": 0.9, "multiplier": [5, 3, 2]}),
    "unrotated_ue_filter": ("unrotated", {"ue_filter": ["2", "12"], "value_min": 0.1}),
}

for name, bearing in DATA.items():
    make_dataset(HERE / f"data_{name}", bearing=bearing, n_samples=1500)

def node():
    from nodejs_wheel import executable
    d = Path(executable.__file__).parent
    return str(d / "node.exe") if (d / "node.exe").exists() else str(d / "bin" / "node")

cases = {}
(HERE / "expected").mkdir(exist_ok=True)
for case, (data, extra) in CASES.items():
    files = HERE / f"data_{data}"
    p = params_for("header.txt", "blocks.csv", "samples.csv", **extra)
    cfg = HERE / f"_{case}_cfg.json"
    cfg.write_text(json.dumps({"format": "swathweb-config", "version": 1, "app": "swath_local", "params": p}))
    out = HERE / f"_{case}_out.json"
    r = subprocess.run([node(), str(SL / "tools" / "run_node.mjs"), str(cfg), str(files), str(out)], capture_output=True, text=True, cwd=SL)
    assert r.returncode == 0, r.stderr
    res = json.loads(out.read_text(encoding="utf-8"))
    keep = {"grid": {k: res["meta"]["grid"][k] for k in ("corner", "size", "shape")}, "qa": res["meta"]["qa"],
            "summary": res["summary"], "panels": res["panels"], "swaths": res["swaths"]}
    (HERE / "expected" / f"{case}.json").write_text(json.dumps(keep, separators=(",", ":")), encoding="utf-8")
    cases[case] = {"data": f"data_{data}", "params": p}
    cfg.unlink(); out.unlink()
(HERE / "cases.json").write_text(json.dumps(cases, indent=1), encoding="utf-8")

commit = subprocess.run(["git", "-C", str(SL), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
dirty = bool(subprocess.run(["git", "-C", str(SL), "status", "--porcelain", "--", "src"], capture_output=True, text=True).stdout.strip())
version = next(l.split('"')[1] for l in (SL / "src" / "engine.js").read_text(encoding="utf-8").splitlines() if l.startswith("export const VERSION"))
files = sorted(p for p in HERE.rglob("*") if p.is_file() and p.name not in ("MANIFEST.json", "generate.py", "README.md") and "__pycache__" not in p.parts)
(HERE / "MANIFEST.json").write_text(json.dumps({
    "contract_version": 1, "source": "swath_local", "swath_local_commit": commit, "swath_local_src_dirty": dirty, "engine_version": version,
    "sha256": {str(p.relative_to(HERE)).replace("\\", "/"): hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest() for p in files}}, indent=1), encoding="utf-8")
print("written", len(files), "files; swath_local", commit, "dirty" if dirty else "clean", "engine", version)
