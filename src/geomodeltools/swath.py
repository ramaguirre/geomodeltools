"""Swath (drift) analysis: block-model grades against composites, panel by panel and per estimation unit (UE).

Calculation only (NumPy and pandas, no plotting). This is the reference implementation of the contract in `docs/swath_contract.md`;
the browser tool `swath_local` runs the same rules in JavaScript and is checked against the golden data in `tests/golden/swath/`.
Column names are those of the JavaScript engine (Spanish) so the two outputs can be compared field by field.

Rules, in the order they are applied to every row of a table (each excluded row is counted once, under the first rule it fails):
no value -> value outside [value_min, value_max] -> no coordinates -> no UE -> UE excluded by `ue_filter` -> invalid weight
(not finite or <= 0) -> outside the panel grid. Weights: blocks = dx * dy * dz (times density if given; 1 without dimensions),
composites = length (1 without a length column). Grades are weight-averaged; `media_simple_*` are unweighted means.
A panel is compared only when it holds at least `min_blocks` blocks and `min_samples` composites. Edge panels are kept.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

CONTRACT_VERSION = 1
AXES = ("x", "y", "z")
_NUM = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
_NULLS = {"", "NA", "NaN", "nan", "NULL", "null", "-", "*"}


# ---------------------------------------------------------------------------------------------- Vulcan header
def _nums(s: str) -> list[float]:
    return [float(x) for x in _NUM.findall(s)]


def parse_header(text: str) -> dict:
    """Read a Vulcan block-model header: origin, bearing/dip/plunge and the schemas (offsets and block sizes)."""
    origin, bdp = None, [90.0, 0.0, 0.0]
    schemas: list[dict] = []
    cur: dict | None = None
    pending = None
    in_vars = False

    def close():
        nonlocal cur
        if cur and "offset_min" in cur and "blocks_min" in cur:
            schemas.append({"offset_min": cur["offset_min"], "offset_max": cur.get("offset_max", cur["offset_min"]),
                            "blocks_min": cur["blocks_min"], "blocks_max": cur.get("blocks_max", cur["blocks_min"])})
        cur = None

    for raw in text.splitlines():
        line = raw.strip()
        low = line.lower()
        idx = line.find(":")
        key = (line[:idx] if idx >= 0 else line).strip().lower()
        val = line[idx + 1:] if idx >= 0 else ""
        if in_vars:
            if not line:
                in_vars = False
            continue
        if low.startswith("variables") and idx < 0:
            in_vars = True
            continue
        if low.startswith("schema"):
            close()
            cur, pending = {}, None
            continue
        if key == "origin":
            origin = _nums(val)[:3]
        elif key.startswith("bearing/dip/plunge"):
            v = _nums(val)
            bdp = [v[0] if len(v) > 0 else 90.0, v[1] if len(v) > 1 else 0.0, v[2] if len(v) > 2 else 0.0]
        elif cur is not None:
            v = _nums(val)[:3]
            if key.startswith("offset minimum"):
                cur["offset_min"], pending = v, "offset"
            elif key.startswith("blocks minimum"):
                cur["blocks_min"], pending = v, "blocks"
            elif key == "maximum" and pending:
                cur[pending + "_max"], pending = v, None
    close()
    if not origin or len(origin) != 3:
        raise ValueError("no 'Origin' line in the header")
    if not schemas:
        raise ValueError("no 'Schema' with 'Offset minimum' and 'Blocks minimum' in the header")
    return {"origin": origin, "bearing": bdp[0], "dip": bdp[1], "plunge": bdp[2], "schemas": schemas}


def rotation_matrix(bearing: float, dip: float, plunge: float) -> np.ndarray:
    """3x3 whose COLUMNS are the model's X, Y, Z axes: Rz(90 - bearing) . Rx(dip) . Rz(plunge) (Vulcan header convention)."""
    def rz(d):
        c, s = math.cos(math.radians(d)), math.sin(math.radians(d))
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])

    def rx(d):
        c, s = math.cos(math.radians(d)), math.sin(math.radians(d))
        return np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]])

    return rz(90 - bearing) @ rx(dip) @ rz(plunge)


# ---------------------------------------------------------------------------------------------- panel grid
@dataclass
class Grid:
    """Panel grid: `corner` is the world xyz of the minimum corner, `axes` has the unit X, Y, Z axes as COLUMNS."""
    corner: np.ndarray
    axes: np.ndarray
    size: np.ndarray
    shape: tuple[int, int, int]
    block_size: np.ndarray = field(default_factory=lambda: np.zeros(3))
    multiplier: tuple[int, int, int] = (1, 1, 1)

    @classmethod
    def from_header(cls, header: dict | str, multiplier=(4, 4, 4)) -> "Grid":
        """As the original tool: block size = largest minimum block size over the schemas (the parent block);
        offsets = component-wise extremes over the schemas; panels hold `multiplier` parent blocks per axis."""
        h = parse_header(header) if isinstance(header, str) else header
        omin = np.max([s["offset_min"] for s in h["schemas"]], axis=0)
        omax = np.max([s["offset_max"] for s in h["schemas"]], axis=0)
        bsize = np.max([s["blocks_min"] for s in h["schemas"]], axis=0)
        mult = tuple(int(m) for m in multiplier)
        size = bsize * np.array(mult, dtype=float)
        shape = tuple(int(max(abs(math.ceil((omax[a] - omin[a]) / size[a])), 1)) for a in range(3))
        R = rotation_matrix(h["bearing"], h["dip"], h["plunge"])
        corner = np.array(h["origin"]) + R @ omin
        return cls(corner, R, size, shape, bsize, mult)

    @property
    def n_panels(self) -> int:
        return int(np.prod(self.shape))

    def locate(self, xyz: np.ndarray) -> np.ndarray:
        """Panel indices (i, j, k) of points (n x 3), or -1 in the first column when outside the grid."""
        ijk = np.floor((np.asarray(xyz, dtype=float) - self.corner) @ self.axes / self.size).astype(np.int64)
        inside = ((ijk >= 0) & (ijk < np.array(self.shape))).all(axis=1)
        ijk[~inside] = -1
        return ijk

    def panel_number(self, ijk: np.ndarray) -> np.ndarray:
        """X-fastest panel number."""
        return ijk[:, 0] + self.shape[0] * (ijk[:, 1] + self.shape[1] * ijk[:, 2])

    def centre_local(self, ijk: np.ndarray) -> np.ndarray:
        return (ijk + 0.5) * self.size

    def centre_world(self, ijk: np.ndarray) -> np.ndarray:
        return self.corner + self.centre_local(ijk) @ self.axes.T

    def axis_is_world(self, a: int) -> bool:
        return all(abs(self.axes[k, a] - (1.0 if k == a else 0.0)) < 1e-9 for k in range(3))

    def axis_coordinate(self, a: int, index: np.ndarray) -> np.ndarray:
        """Coordinate shown on a swath plot: real when that axis is not rotated, else the distance along it."""
        local = (np.asarray(index) + 0.5) * self.size[a]
        return self.corner[a] + local if self.axis_is_world(a) else local


# ---------------------------------------------------------------------------------------------- reduction
def ue_key(v) -> str | None:
    """UE code as clean text: 22, "22", "22.0" and " 22 " are all "22"; empty/NA-like give None."""
    if v is None:
        return None
    if isinstance(v, (int, float, np.integer, np.floating)):
        f = float(v)
        if math.isfinite(f):
            return str(int(f)) if f == int(f) and abs(f) < 1e15 else repr(f)
        return None
    t = str(v).strip().strip('"').strip()
    try:
        f = float(t)
        if math.isfinite(f):
            return str(int(f)) if f == int(f) and abs(f) < 1e15 else repr(f)
    except ValueError:
        pass
    return None if t in _NULLS else t


def ue_sort_key(u: str):
    try:
        return (0, float(u), "")
    except ValueError:
        return (1, 0.0, u)


_QA_KEYS = ("filas", "sin_valor", "fuera_de_rango", "sin_coordenadas", "sin_ue", "ue_excluida", "peso_invalido", "fuera_de_grilla", "usadas")


def reduce_table(df: pd.DataFrame, cols: dict, role: str, grid: Grid, value_min=0.0, value_max=None, ue_filter=None):
    """Sum count, weight, weight*value and value per (UE, panel). `cols` maps roles to column names:
    value, ue, x, y, z and, for blocks, dx, dy, dz, density; for composites, length. Returns (sums, qa)."""
    n = len(df)
    qa = dict.fromkeys(_QA_KEYS, 0)
    qa["filas"] = n
    alive = np.ones(n, dtype=bool)

    def drop(mask, key):
        nonlocal alive
        hit = alive & mask
        qa[key] = int(hit.sum())
        alive &= ~mask

    def num(col):
        return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)

    v = num(cols["value"])
    drop(~np.isfinite(v), "sin_valor")
    out_of_range = np.zeros(n, dtype=bool)
    with np.errstate(invalid="ignore"):
        if value_min is not None:
            out_of_range |= ~(v >= value_min)
        if value_max is not None:
            out_of_range |= ~(v <= value_max)
    drop(out_of_range, "fuera_de_rango")
    xyz = np.c_[num(cols["x"]), num(cols["y"]), num(cols["z"])]
    drop(~np.isfinite(xyz).all(axis=1), "sin_coordenadas")
    ue = np.array([ue_key(u) for u in df[cols["ue"]].tolist()], dtype=object)
    drop(np.array([u is None for u in ue]), "sin_ue")
    if ue_filter is not None:
        keep = {str(u) for u in ue_filter}
        drop(np.array([u not in keep for u in ue]), "ue_excluida")
    w = np.ones(n)
    if role == "blocks":
        if all(cols.get(k) for k in ("dx", "dy", "dz")):
            w = num(cols["dx"]) * num(cols["dy"]) * num(cols["dz"])
        if cols.get("density"):
            w = w * num(cols["density"])
    elif cols.get("length"):
        w = num(cols["length"])
    drop(~(np.isfinite(w) & (w > 0)), "peso_invalido")
    ijk = np.full((n, 3), -1, dtype=np.int64)
    ijk[alive] = grid.locate(xyz[alive])
    drop(ijk[:, 0] < 0, "fuera_de_grilla")
    qa["usadas"] = int(alive.sum())
    d = pd.DataFrame({"ue": ue[alive], "panel": grid.panel_number(ijk[alive]), "cnt": 1.0, "w": w[alive],
                      "wv": w[alive] * v[alive], "sv": v[alive]})
    sums = d.groupby(["ue", "panel"], sort=False).sum()
    return sums, qa


# ---------------------------------------------------------------------------------------------- comparison
@dataclass
class SwathResult:
    summary: pd.DataFrame
    panels: dict[str, pd.DataFrame]
    swaths: dict[str, dict[str, pd.DataFrame]]
    qa: dict[str, dict]
    grid: Grid
    contract_version: int = CONTRACT_VERSION


def _pearson_slope(x: np.ndarray, y: np.ndarray):
    """np.corrcoef and cov(bias=True)/var, as the JavaScript engine."""
    n = len(x)
    a, b = x - x.mean(), y - y.mean()
    sxx, syy, sxy = (a * a).sum(), (b * b).sum(), (a * b).sum()
    sx, sy = math.sqrt(sxx / n), math.sqrt(syy / n)
    if n < 3 or not (sx > 0) or not (sy > 0):
        return float("nan"), float("nan")
    return sxy / math.sqrt(sxx * syy), (sxy / n) / (sxx / n)


def swath_compare(blocks: pd.DataFrame, samples: pd.DataFrame, grid: Grid, block_cols: dict, sample_cols: dict, *,
                  value_min=0.0, value_max=None, min_samples=1, min_blocks=1, ue_filter=None) -> SwathResult:
    """Compare blocks with composites in every panel and UE. See the module docstring for the rules."""
    bs, bqa = reduce_table(blocks, block_cols, "blocks", grid, value_min, value_max, ue_filter)
    ss, sqa = reduce_table(samples, sample_cols, "samples", grid, value_min, value_max, ue_filter)
    ues = sorted({u for u, _ in ss.index}, key=ue_sort_key)
    block_ues = {u for u, _ in bs.index}
    rows, panels, swaths = [], {}, {}
    for ue in ues:
        row = {"ue": ue, "estado": "OK", "paneles": 0}
        if ue not in block_ues:
            row["estado"] = "sin bloques de esta UE"
            rows.append(row)
            continue
        b, s = bs.loc[ue], ss.loc[ue]
        j = b.join(s, how="inner", lsuffix="_b", rsuffix="_s")
        j = j[(j["cnt_s"] >= min_samples) & (j["cnt_b"] >= min_blocks)].sort_index()
        if j.empty:
            row["estado"] = "sin paneles con bloques y muestras"
            rows.append(row)
            continue
        panel = j.index.to_numpy(dtype=np.int64)
        nx, ny = grid.shape[0], grid.shape[1]
        ijk = np.c_[panel % nx, (panel // nx) % ny, panel // (nx * ny)]
        cw, cl = grid.centre_world(ijk), grid.centre_local(ijk)
        lb, ls = (j["wv_b"] / j["w_b"]).to_numpy(), (j["wv_s"] / j["w_s"]).to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            dif_pct = np.where(ls > 0, 100 * (lb - ls) / ls, np.nan)
        p = pd.DataFrame({
            "panel": panel, "i": ijk[:, 0], "j": ijk[:, 1], "k": ijk[:, 2],
            "x_centro": cw[:, 0], "y_centro": cw[:, 1], "z_centro": cw[:, 2],
            "dist_x": cl[:, 0], "dist_y": cl[:, 1], "dist_z": cl[:, 2],
            "n_bloques": j["cnt_b"].to_numpy(), "n_muestras": j["cnt_s"].to_numpy(),
            "peso_bloques": j["w_b"].to_numpy(), "peso_muestras": j["w_s"].to_numpy(),
            "ley_bloques": lb, "ley_muestras": ls,
            "media_simple_bloques": (j["sv_b"] / j["cnt_b"]).to_numpy(), "media_simple_muestras": (j["sv_s"] / j["cnt_s"]).to_numpy(),
            "diferencia": lb - ls, "diferencia_pct": dif_pct})
        panels[ue] = p
        swaths[ue] = {}
        for a, ax in enumerate(AXES):
            g = j.assign(pos=ijk[:, a], paneles=1.0).groupby("pos").sum()
            swaths[ue][ax] = pd.DataFrame({
                "indice": g.index.to_numpy(), "coordenada": grid.axis_coordinate(a, g.index.to_numpy()),
                "n_bloques": g["cnt_b"].to_numpy(), "n_muestras": g["cnt_s"].to_numpy(), "paneles": g["paneles"].to_numpy(),
                "ley_bloques": (g["wv_b"] / g["w_b"]).to_numpy(), "ley_muestras": (g["wv_s"] / g["w_s"]).to_numpy(),
                "media_simple_bloques": (g["sv_b"] / g["cnt_b"]).to_numpy(), "media_simple_muestras": (g["sv_s"] / g["cnt_s"]).to_numpy()})
        mb, ms = j["wv_b"].sum() / j["w_b"].sum(), j["wv_s"].sum() / j["w_s"].sum()
        mx, my = lb.mean(), ls.mean()
        corr, slope = _pearson_slope(lb, ls)
        row.update({
            "paneles": len(j), "n_bloques": float(j["cnt_b"].sum()), "n_muestras": float(j["cnt_s"].sum()),
            "ley_bloques": mb, "ley_muestras": ms, "dif_global_pct": 100 * (mb - ms) / ms if ms > 0 else float("nan"),
            "media_paneles_bloques": mx, "media_paneles_muestras": my,
            "dif_paneles_pct": 100 * (mx - my) / my if my > 0 else float("nan"), "correlacion": corr, "pendiente": slope})
        rows.append(row)
    return SwathResult(pd.DataFrame(rows), panels, swaths, {"muestras": sqa, "bloques": bqa}, grid)
