"""`import geomodeltools` is light, and every public name still resolves (cachorro_avance and others import them from the top level)."""
import subprocess
import sys

import pytest

import geomodeltools


@pytest.mark.parametrize("name", geomodeltools.__all__)
def test_every_public_name_resolves(name):
    assert getattr(geomodeltools, name) is not None


def test_unknown_name_is_an_attribute_error():
    with pytest.raises(AttributeError):
        geomodeltools.no_such_function


def test_desurvey_and_swath_import_without_the_heavy_packages():
    code = (
        "import sys\n"
        "for m in ('geopandas', 'shapely', 'rasterio', 'requests', 'scipy', 'matplotlib'):\n"
        "    sys.modules[m] = None            # an import of any of them now fails\n"
        "from geomodeltools import desurvey_intervals, Desurveyed\n"
        "import geomodeltools.swath, geomodeltools.palette, geomodeltools.contacts, geomodeltools.columns\n"
        "print('ok')\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr


def test_a_missing_optional_package_names_its_extra():
    code = (
        "import sys\n"
        "sys.modules['geopandas'] = None\n"
        "import geomodeltools\n"
        "try:\n"
        "    geomodeltools.bufferize_2d_polygons\n"
        "except ImportError as e:\n"
        "    print(e)\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert "geomodeltools[geo]" in r.stdout, (r.stdout, r.stderr)
