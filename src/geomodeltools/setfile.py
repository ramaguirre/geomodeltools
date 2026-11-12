"""Drillhole set files: a `SetSpec` saved as YAML (``.dhset.yaml``) or JSON (``.dhset.json``) next to the tables it describes.

It holds the tables' file names (relative to the set file, so the folder can move), how each is read, and what each column is: never
data, so it is small, safe to keep beside the data in a shared folder, and can be edited by hand. `DrillholeSet.save` / `open` use it.
"""
from __future__ import annotations

import json
import pathlib

from .schema import SetSpec

_HEADER = ("# Drillhole set (geomodeltools). What each table is and what its columns are; no data. Paths are relative to this file.\n"
           "# Open it with geomodeltools.DrillholeSet.open(path), or load it in geoview (Load, or Viewer.open_drillhole_set).\n")


def _is_json(path: pathlib.Path) -> bool:
    return path.suffix.lower() == ".json"


def write_setfile(path, spec: SetSpec) -> pathlib.Path:
    """Write ``spec`` to ``path`` (YAML unless the name ends in .json). The file is written beside its final name first, so a failed
    write never leaves half a file."""
    path = pathlib.Path(path)
    data = spec.to_dict()
    if _is_json(path):
        text = json.dumps(data, indent=1, ensure_ascii=False)
    else:
        import yaml

        text = _HEADER + yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=120)
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
    return path


def read_setfile(path) -> SetSpec:
    """The `SetSpec` in a set file (YAML or JSON)."""
    path = pathlib.Path(path)
    text = path.read_text(encoding="utf-8")
    if _is_json(path):
        data = json.loads(text)
    else:
        import yaml

        data = yaml.safe_load(text)
    if not isinstance(data, dict) or "collars" not in data:
        raise ValueError(f"{path.name} is not a drillhole set file (no collars table in it)")
    return SetSpec.from_dict(data)
