from __future__ import annotations

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
