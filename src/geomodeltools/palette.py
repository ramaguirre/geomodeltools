"""Colours for geological units: ICS / CGMW chronostratigraphic colours, lithology families and a distinguishability check.

Convention used (see docs/unit_colours.md): layered units take the colour of their age from the International Chronostratigraphic Chart
(CGMW colour code), at stage level when the unit's age fits one stage, else at epoch level; intrusive rocks take a red / pink family
(plutonic, porphyry, hypabyssal), lighter when younger; unconsolidated cover takes the Quaternary yellow; water is blue. Units that end
up too similar (CIEDE2000 below a threshold) are pushed apart in lightness, never in hue family.
"""
from __future__ import annotations

import colorsys
import math

# ICS colour code (CGMW), hex. Source: ICS chart colours as published in the Macrostrat interval table (2025 chart values).
ICS = {
    "Holocene": "FEF2E0", "Pleistocene": "FFF2AE", "Pliocene": "FFFF99", "Miocene": "FFFF00",
    "Quaternary": "F9F97F", "Neogene": "FFE619", "Paleogene": "FD9A52",
    "Oligocene": "FDC07A", "Eocene": "FDB46C", "Paleocene": "FDA75F",
    "Aquitanian": "FFFF33", "Burdigalian": "FFFF41", "Langhian": "FFFF4D", "Serravallian": "FFFF59", "Tortonian": "FFFF66",
    "Messinian": "FFFF73", "Zanclean": "FFFFB3", "Piacenzian": "FFFFBF", "Gelasian": "FFEDB3", "Calabrian": "FFF2C7",
    "Rupelian": "FED99A", "Chattian": "FEE6AA", "Priabonian": "FDCDA1",
    "Late Cretaceous": "A6D84A", "Early Cretaceous": "8CCD57", "Cretaceous": "7FC64E",
    "Maastrichtian": "F2FA8C", "Campanian": "E6F47F", "Santonian": "D9EF74", "Coniacian": "CCE968", "Turonian": "BFE35D",
    "Cenomanian": "B3DE53",
}
WATER = "73B2FF"


def hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def rgb_to_hex(rgb) -> str:
    return "{:02X}{:02X}{:02X}".format(*[int(round(min(255, max(0, c)))) for c in rgb])


def rgb_to_lab(rgb) -> tuple[float, float, float]:
    def lin(c):
        c /= 255.0
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (lin(c) for c in rgb)
    x = (0.4124564 * r + 0.3575761 * g + 0.1804375 * b) / 0.95047
    y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    z = (0.0193339 * r + 0.1191920 * g + 0.9503041 * b) / 1.08883
    f = lambda t: t ** (1 / 3) if t > 216 / 24389 else (24389 / 27 * t + 16) / 116
    fx, fy, fz = f(x), f(y), f(z)
    return 116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)


def lab_to_rgb(lab) -> tuple[int, int, int]:
    L, a, b = lab
    fy = (L + 16) / 116
    fx, fz = fy + a / 500, fy - b / 200
    finv = lambda t: t ** 3 if t ** 3 > 216 / 24389 else (116 * t - 16) / (24389 / 27)
    x, y, z = finv(fx) * 0.95047, finv(fy), finv(fz) * 1.08883
    r = 3.2404542 * x - 1.5371385 * y - 0.4985314 * z
    g = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z
    bl = 0.0556434 * x - 0.2040259 * y + 1.0572252 * z
    gam = lambda c: 12.92 * c if c <= 0.0031308 else 1.055 * max(c, 0) ** (1 / 2.4) - 0.055
    return tuple(int(round(min(255, max(0, 255 * gam(c))))) for c in (r, g, bl))


def delta_e2000(lab1, lab2) -> float:
    """CIEDE2000 (Sharma, Wu, Dalal 2005)."""
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2
    C1, C2 = math.hypot(a1, b1), math.hypot(a2, b2)
    Cm = (C1 + C2) / 2
    G = 0.5 * (1 - math.sqrt(Cm ** 7 / (Cm ** 7 + 25 ** 7)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = math.hypot(a1p, b1), math.hypot(a2p, b2)
    h1p = math.degrees(math.atan2(b1, a1p)) % 360
    h2p = math.degrees(math.atan2(b2, a2p)) % 360
    dLp, dCp = L2 - L1, C2p - C1p
    if C1p * C2p == 0:
        dhp = 0.0
    else:
        dhp = h2p - h1p
        dhp += -360 if dhp > 180 else 360 if dhp < -180 else 0
    dHp = 2 * math.sqrt(C1p * C2p) * math.sin(math.radians(dhp / 2))
    Lpm, Cpm = (L1 + L2) / 2, (C1p + C2p) / 2
    if C1p * C2p == 0:
        hpm = h1p + h2p
    elif abs(h1p - h2p) <= 180:
        hpm = (h1p + h2p) / 2
    else:
        hpm = (h1p + h2p + 360) / 2 if h1p + h2p < 360 else (h1p + h2p - 360) / 2
    T = (1 - 0.17 * math.cos(math.radians(hpm - 30)) + 0.24 * math.cos(math.radians(2 * hpm))
         + 0.32 * math.cos(math.radians(3 * hpm + 6)) - 0.20 * math.cos(math.radians(4 * hpm - 63)))
    dth = 30 * math.exp(-(((hpm - 275) / 25) ** 2))
    Rc = 2 * math.sqrt(Cpm ** 7 / (Cpm ** 7 + 25 ** 7))
    Sl = 1 + 0.015 * (Lpm - 50) ** 2 / math.sqrt(20 + (Lpm - 50) ** 2)
    Sc, Sh = 1 + 0.045 * Cpm, 1 + 0.015 * Cpm * T
    Rt = -math.sin(math.radians(2 * dth)) * Rc
    return math.sqrt((dLp / Sl) ** 2 + (dCp / Sc) ** 2 + (dHp / Sh) ** 2 + Rt * (dCp / Sc) * (dHp / Sh))


def lch_to_rgb(L: float, C: float, h_deg: float) -> tuple[int, int, int]:
    return lab_to_rgb((L, C * math.cos(math.radians(h_deg)), C * math.sin(math.radians(h_deg))))


def family_ramp(hue_deg: float, n: int, l_young: float = 72.0, l_old: float = 40.0, chroma: float = 55.0,
                hue_jitter: float = 6.0) -> list[tuple[int, int, int]]:
    """n colours of one hue family, lightest first (youngest), darkest last (oldest); neighbours alternate a few degrees of hue
    so that adjacent steps differ in more than lightness."""
    out = []
    for i in range(n):
        t = i / max(n - 1, 1)
        L = l_young + (l_old - l_young) * t
        out.append(lch_to_rgb(L, chroma, hue_deg + (hue_jitter if i % 2 else -hue_jitter)))
    return out


def min_pairwise_de(colours: dict) -> tuple[float, tuple[str, str]]:
    labs = {k: rgb_to_lab(v) for k, v in colours.items()}
    best, who = 1e9, ("", "")
    keys = list(labs)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            d = delta_e2000(labs[a], labs[b])
            if d < best:
                best, who = d, (a, b)
    return best, who


def spread_lightness(colours: dict, group: list[str], min_de: float = 10.0, step: float = 2.0, max_iter: int = 60) -> dict:
    """Push the members of `group` apart in lightness (keeping a and b) until every pair in the group is at least `min_de` apart.
    Members keep their order by lightness."""
    out = dict(colours)
    for _ in range(max_iter):
        worst, who = min_pairwise_de({k: out[k] for k in group})
        if worst >= min_de:
            break
        order = sorted(group, key=lambda k: rgb_to_lab(out[k])[0])
        a, b = who
        lo, hi = (a, b) if rgb_to_lab(out[a])[0] <= rgb_to_lab(out[b])[0] else (b, a)
        L, A, B = rgb_to_lab(out[lo]); out[lo] = lab_to_rgb((max(L - step, 5), A, B))
        L, A, B = rgb_to_lab(out[hi]); out[hi] = lab_to_rgb((min(L + step, 98), A, B))
    return out


def ramp_about(base_rgb, n: int, span: float = 24.0) -> list[tuple[int, int, int]]:
    """n colours with the hue and chroma of `base_rgb`, lightness spread evenly over `span` L units centred on the base and
    ordered lightest first (youngest) to darkest (oldest). For several units that share one ICS stage colour."""
    L, a, b = rgb_to_lab(base_rgb)
    if n == 1:
        return [tuple(base_rgb)]
    top = min(L + span / 2, 97.0)
    return [lab_to_rgb((top - span * i / (n - 1), a, b)) for i in range(n)]
