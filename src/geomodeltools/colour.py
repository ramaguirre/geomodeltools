from __future__ import annotations

import colorsys
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.dom import minidom

import numpy as np

_HEX_RE = re.compile(r"^#?[0-9A-Fa-f]{6}$")


def _is_hex_colour(value) -> bool:
    return isinstance(value, str) and bool(_HEX_RE.match(value.strip()))


def _hex_to_rgb(value: str) -> tuple[float, float, float]:
    value = value.strip().lstrip("#")
    r, g, b = (int(value[i : i + 2], 16) for i in (0, 2, 4))
    return (r / 255, g / 255, b / 255)


def _normalise_colour(value) -> tuple[float, float, float]:
    """Return an (r, g, b) tuple of floats in the 0-1 range.

    Accepts a hex string (``'#RRGGBB'`` or ``'RRGGBB'``) or an RGB triple,
    with the triple's components read either in the 0-1 range or the 0-255
    range (auto-detected from whether any component exceeds 1).
    """
    if _is_hex_colour(value):
        return _hex_to_rgb(value)

    rgb = tuple(float(c) for c in value)
    if len(rgb) != 3:
        raise ValueError(f"Expected a hex string or an (r, g, b) triple, got {value!r}")
    if any(c > 1 for c in rgb):
        rgb = tuple(c / 255 for c in rgb)
    if any(c < 0 or c > 1 for c in rgb):
        raise ValueError(f"RGB values must fall within 0-1 (or 0-255), got {value!r}")
    return rgb


def leapfrog_colour_palette(code_rgb_dict: dict, output_name: str | Path) -> Path:
    """
    Generate a Leapfrog Colour Palette (.lfc) XML file based on the provided colours.

    Parameters:
    - code_rgb_dict (dict): A dictionary mapping codes to colours. Each colour may be
                        given as an RGB triple (components in 0-1 or 0-255) or as a
                        hex string (e.g. '#E62619' or 'E62619'); the format is
                        auto-detected per entry.
    - output_name (str | Path): The name of the output file (without the file extension).
                        If the provided name ends with '.lfc', the extension is ignored.

    Returns:
    Path: The path of the written .lfc file.

    Example:
    >>> code_rgb_dict = {'A': (0.9, 0.15, 0.12), 'B': '#1A80FF', 'C': (138, 33, 0)}
    >>> leapfrog_colour_palette(code_rgb_dict, 'my_palette')

    This example will generate an XML file named 'my_palette.lfc' with entries for codes 'A', 'B', and 'C'
    and their respective RGB values.

    Note: The generated XML file will have the structure of a Leapfrog Colour Palette (.lfc) file.

    """
    root = ET.Element("LeapfrogColourPalette", version="1.0", type="legend")
    for code, colour in code_rgb_dict.items():
        r, g, b = _normalise_colour(colour)
        entry = ET.SubElement(root, "Entry")
        ET.SubElement(entry, "Code").text = str(code)
        ET.SubElement(entry, "Colour").text = f"{r} {g} {b}"

    name = str(output_name)
    if name.lower().endswith(".lfc"):
        name = name[:-4]
    output_path = Path(name + ".lfc")

    pretty_xml = minidom.parseString(ET.tostring(root, encoding="UTF-8")).toprettyxml(
        indent="  ", encoding="UTF-8"
    )
    output_path.write_bytes(pretty_xml)

    print(f"Palette saved to {output_path}")
    return output_path


def leapfrog_colour2dictionary(lfc_file_path: str | Path) -> dict[str, np.ndarray]:
    """
    Convert an .lfc file containing Leapfrog color palette data into a dictionary.

    The XML file should have a structure with <Entry> elements, each containing a <Code> and a <Colour>.
    The <Code> represents the category, and the <Colour> represents the color in RGB format.
    This function parses the XML and returns a dictionary where each key is a category code,
    and the corresponding value is a list of three floats representing the RGB color.

    Parameters:
    lfc_file_path (str | Path): The path to the XML file.

    Returns:
    dict: A dictionary with category codes as keys and RGB color values (0-1 floats) as arrays.
    """
    tree = ET.parse(lfc_file_path)
    root = tree.getroot()

    data = {}
    for entry in root.findall("Entry"):
        code = entry.find("Code").text
        colour = np.array([float(i) for i in entry.find("Colour").text.split()])
        data[code] = colour

    return data


lfc2dict = leapfrog_colour2dictionary


def cmyk_to_rgb(c: float, m: float, y: float, k: float) -> tuple[int, int, int]:
    """Convert CMYK (0-100 each) to RGB (0-255 each)."""
    r = 255 * (1 - c / 100) * (1 - k / 100)
    g = 255 * (1 - m / 100) * (1 - k / 100)
    b = 255 * (1 - y / 100) * (1 - k / 100)
    return tuple(round(v) for v in (r, g, b))


def hsv_to_rgb(h: float, s: float, v: float) -> tuple[int, int, int]:
    """Convert HSV (H: 0-360, S/V: 0-100) to RGB (0-255 each)."""
    r, g, b = colorsys.hsv_to_rgb(h / 360, s / 100, v / 100)
    return tuple(round(c * 255) for c in (r, g, b))


def lab_to_rgb(L: float, a: float, b: float) -> tuple[int, int, int]:
    """Convert CIE L*a*b* (D65 white) to sRGB (0-255 each), clipped to the gamut."""
    fy = (L + 16) / 116
    fx, fz = fy + a / 500, fy - b / 200
    f = lambda t: t ** 3 if t ** 3 > 0.008856 else (t - 16 / 116) / 7.787
    X, Y, Z = 0.95047 * f(fx), f(fy), 1.08883 * f(fz)
    linear = (3.2406 * X - 1.5372 * Y - 0.4986 * Z,
              -0.9689 * X + 1.8758 * Y + 0.0415 * Z,
              0.0557 * X - 0.2040 * Y + 1.0570 * Z)
    encode = lambda c: 12.92 * c if c <= 0.0031308 else 1.055 * max(c, 0) ** (1 / 2.4) - 0.055
    return tuple(int(round(min(255, max(0, 255 * encode(c))))) for c in linear)


def cim_color_to_rgb(color: dict | None) -> tuple[int, int, int] | None:
    """Convert a CIM color dict (CIMRGBColor, CIMCMYKColor, CIMHSVColor, CIMGrayColor or CIMLABColor) to an (r, g, b) tuple."""
    if color is None:
        return None
    values = color['values']
    if color['type'] == 'CIMRGBColor':
        return tuple(round(v) for v in values[:3])
    elif color['type'] == 'CIMCMYKColor':
        return cmyk_to_rgb(*values[:4])
    elif color['type'] == 'CIMHSVColor':
        return hsv_to_rgb(*values[:3])
    elif color['type'] == 'CIMGrayColor':
        grey = round(255 * values[0] / 100)
        return (grey, grey, grey)
    elif color['type'] == 'CIMLABColor':
        return lab_to_rgb(*values[:3])
    else:
        raise ValueError(f"Unsupported color type: {color['type']}")


def get_symbol_color(symbol_layer: dict) -> tuple[tuple[int, int, int], str] | tuple[None, None]:
    """
    Take a CIM symbol layer dict (as found in symbol_type_example_list) and
    return the most representative color as (rgb, hex).

    Handles: CIMSolidFill, CIMSolidStroke, CIMHatchFill, CIMPictureFill,
    CIMVectorMarker, CIMCharacterMarker (recursing into nested symbols).
    """
    symbol_type = symbol_layer.get('type')

    if symbol_type in ('CIMSolidFill', 'CIMSolidStroke'):
        rgb = cim_color_to_rgb(symbol_layer['color'])

    elif symbol_type == 'CIMHatchFill':
        # color of the hatch lines themselves
        line_layer = symbol_layer['lineSymbol']['symbolLayers'][0]
        return get_symbol_color(line_layer)

    elif symbol_type == 'CIMPictureFill':
        # use the substitution whose old color is black (the pattern's "ink"),
        # falling back to the first substitution if none matches
        subs = symbol_layer.get('colorSubstitutions', [])
        target = next(
            (s for s in subs if cim_color_to_rgb(s['oldColor']) == (255, 255, 255)),
            subs[0] if subs else None,
        )
        if target is None:
            return None, None
        rgb = cim_color_to_rgb(target['newColor'])

    elif symbol_type in ('CIMVectorMarker',):
        # dig into the first enabled marker graphic's nested symbol layer
        graphics = [g for g in symbol_layer.get('markerGraphics', [])]
        graphic = graphics[0] if graphics else None
        if graphic is None:
            return None, None
        nested_layer = graphic['symbol']['symbolLayers'][0]
        return get_symbol_color(nested_layer)

    elif symbol_type == 'CIMCharacterMarker':
        # prefer the first *enabled* nested layer if this is a list context,
        # otherwise use its own nested polygon symbol fill color
        nested_layer = symbol_layer['symbol']['symbolLayers'][0]
        return get_symbol_color(nested_layer)

    else:
        raise ValueError(f"Unsupported symbol type: {symbol_type}")

    if rgb is None:
        return None, None

    hex_color = '#{:02x}{:02x}{:02x}'.format(*rgb)
    return rgb, hex_color


def _safe_rgb(color):
    try:
        return cim_color_to_rgb(color)
    except (ValueError, KeyError, IndexError):
        return None


def _enabled(layers):
    return (l for l in layers if l.get('enable', True))


def _stroke_colour(layers):
    for lyr in _enabled(layers):
        if lyr.get('type') == 'CIMSolidStroke' and lyr.get('width', 1) > 0 and lyr.get('color'):
            return _safe_rgb(lyr['color'])


def _first_colour(node):
    """First colour found anywhere inside an enabled symbol part (markers on a polygon or along a line)."""
    if isinstance(node, dict):
        if node.get('enable', True) is False:
            return None
        if str(node.get('color', {}).get('type', '')).endswith('Color'):
            return _safe_rgb(node['color'])
        node = list(node.values())
    for child in node if isinstance(node, list) else []:
        rgb = _first_colour(child)
        if rgb:
            return rgb
    return None


def cim_symbol_color(symbol: dict, kind: str) -> tuple[int, int, int] | None:
    """The colour a person sees for a whole CIM symbol (all its layers), as RGB, or None.

    ``kind`` is ``'polygon'`` or ``'line'``. Polygons: the solid fill, else the hatch line, else
    the colour that replaces a pattern fill's white background. Then (and for lines) the outline
    stroke, else the first colour inside a marker. Unlike `get_symbol_color`, which reads one
    symbol layer, this looks at the whole symbol and never raises on unsupported parts.
    """
    layers = symbol.get('symbol', symbol).get('symbolLayers', [])
    if kind == 'polygon':
        for lyr in _enabled(layers):
            if lyr.get('type') == 'CIMSolidFill' and lyr.get('color'):
                return _safe_rgb(lyr['color'])
        for lyr in _enabled(layers):
            if lyr.get('type') == 'CIMHatchFill':
                rgb = _stroke_colour(lyr.get('lineSymbol', {}).get('symbolLayers', []))
                if rgb:
                    return rgb
        for lyr in _enabled(layers):
            if lyr.get('type') == 'CIMPictureFill':
                subs = {tuple(s['oldColor']['values'][:3]): s['newColor'] for s in lyr.get('colorSubstitutions', [])}
                if (255, 255, 255) in subs:
                    return _safe_rgb(subs[(255, 255, 255)])
    return _stroke_colour(layers) or _first_colour(layers)


def read_mapx_symbology(mapx: str | Path, dataset: str, kind: str) -> dict | None:
    """Colours of the map layer drawn from `dataset` in an ArcGIS .mapx (JSON map definition).

    Returns ``{'fields': [...], 'classes': {value tuple: rgb}, 'default': rgb, 'layer': name}``,
    or None when no layer draws that dataset. A dataset drawn twice (say a plain contact line and
    an interpreted one) takes the unique-value layer, which carries the categories. ``kind`` is
    ``'polygon'`` or ``'line'``; see `cim_symbol_color`.
    """
    layers = [l for l in json.loads(Path(mapx).read_text(encoding='utf-8'))['layerDefinitions']
              if l.get('featureTable', {}).get('dataConnection', {}).get('dataset') == dataset and 'renderer' in l]
    layers.sort(key=lambda l: l['renderer']['type'] != 'CIMUniqueValueRenderer')
    if not layers:
        return None
    rend = layers[0]['renderer']
    if rend['type'] == 'CIMSimpleRenderer':
        return {'fields': [], 'classes': {}, 'default': cim_symbol_color(rend['symbol'], kind), 'layer': layers[0]['name']}
    classes = {}
    for grp in rend.get('groups', []):
        for cls in grp['classes']:
            for uv in cls['values']:
                classes[tuple(str(x) for x in uv['fieldValues'])] = cim_symbol_color(cls['symbol'], kind)
    default = cim_symbol_color(rend['defaultSymbol'], kind) if rend.get('defaultSymbol') else None
    return {'fields': rend['fields'], 'classes': classes, 'default': default, 'layer': layers[0]['name']}


def symbology_colors(df, symbology: dict | None) -> list[tuple[int, int, int] | None]:
    """RGB per row of `df`: its class colour in the map, else the layer default, else None.

    `df` holds the layer's attribute columns with the values the map drew (blank and missing
    values match a blank class). Returns an empty list when `symbology` is None.
    """
    if not symbology:
        return []
    if not symbology['fields']:
        return [symbology['default']] * len(df)
    vals = df[symbology['fields']].astype('string').fillna('').apply(lambda c: c.str.strip())
    return [symbology['classes'].get(tuple(row), symbology['default']) for row in vals.itertuples(index=False, name=None)]


def arcgis_lyr_to_leapfrog_lfc(lyr_file: str | Path, output_dir: str | Path, suffix: str = '') -> list[Path]:
    """
    Read an ArcGIS .lyr(x) JSON file and write one Leapfrog .lfc palette per
    unique-values renderer field found in its layer definitions.

    Parameters:
    - lyr_file (str | Path): Path to the ArcGIS layer JSON file.
    - output_dir (str | Path): Directory the .lfc files are written to (created if missing).
    - suffix (str): Optional suffix appended to each output file's field name.

    Returns:
    list[Path]: Paths of the .lfc files written, one per renderer field.
    """
    with open(lyr_file, 'r', encoding='UTF-8') as f:
        lyr_json = json.load(f)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written_paths = []
    for layerDefinition in lyr_json['layerDefinitions']:
        field_name = layerDefinition['renderer']['fields'][0]
        cod_lito_color_dict = {}
        failed_type = []
        for i in layerDefinition['renderer']['groups']:
            for j in i['classes']:
                cod_lito = j['values'][0]['fieldValues'][0]
                symbolLayers = j['symbol']['symbol']['symbolLayers']
                sym = symbolLayers[0]
                for symbolLayer in symbolLayers:
                    if symbolLayer['type'] == 'CIMSolidFill':
                        sym = symbolLayer
                        break
                try:
                    rgb, hex_color = get_symbol_color(sym)
                    if hex_color is None:
                        raise ValueError("no resolvable colour for this symbol")
                    print(f"{cod_lito:<10} {sym['type']:<20} rgb={rgb}  hex={hex_color}")
                    cod_lito_color_dict[cod_lito] = hex_color
                except Exception as e:
                    failed_type.append(sym)
                    print(f"Failed to get color for {sym['type']}: {e}")

        if suffix:
            field_name = f"{field_name}_{suffix}"
        written_paths.append(leapfrog_colour_palette(cod_lito_color_dict, output_dir / f"{field_name}.lfc"))

    return written_paths

