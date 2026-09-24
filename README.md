# geomodeltools

Reusable helpers for 3D geological modeling workflows.

## What it includes

- Geometry densification and point extraction for line and polygon features.
- Inward polygon buffering with optional PyVista point cloud output.
- DEM-based Z assignment using OpenTopography rasters.

## Install

From GitHub (a fixed version, no Git needed):

```bash
pip install https://github.com/ramaguirre/geomodeltools/archive/refs/tags/v0.1.0.zip
```

For development, from a clone of this repo:

```bash
pip install -e .
```

## OpenTopography API key

Downloading DEMs needs a free OpenTopography API key. No key is bundled with the
package; each user registers their own:

1. Sign in (or create an account) at https://portal.opentopography.org, open your
   **MyOpenTopo** dashboard, click **Get an API Key**, then **Request API key**, and copy it.
2. Save it as the user environment variable `OPENTOPOGRAPHY_API_KEY`:
   - Windows: Start menu > type "environment variables" > **Edit environment variables
     for your account** > **New** > Name `OPENTOPOGRAPHY_API_KEY`, Value: your key.
     Or in PowerShell:
     `[Environment]::SetEnvironmentVariable("OPENTOPOGRAPHY_API_KEY", "<your key>", "User")`
   - macOS / Linux: add `export OPENTOPOGRAPHY_API_KEY="<your key>"` to `~/.bashrc` or `~/.zshrc`.
3. Close and reopen your editor so it picks up the variable.

You can also pass `api_key="..."` to `add_z_from_opentopography` /
`download_opentopography_dem`. If `out_tiff_path` already exists, it is reused and
no key is needed.

## Quick example

```python
from pathlib import Path
import geopandas as gpd
from geomodeltools import bufferize_2d_polygons, add_z_from_opentopography

gdf = gpd.read_file("my_polygons.shp")
pts = bufferize_2d_polygons(gdf, feature_cols=["unit"], return_polydata=False)
pts_z, dem_path = add_z_from_opentopography(
    pts,
    out_tiff_path=Path("data/dem.tif"),
    margin_m=0,
)
```

By default, `add_z_from_opentopography` assumes PSAD56 / UTM zone 19S
(EPSG:24879) for inputs with no CRS set, and writes the DEM GeoTIFF in that
same CRS. Pass `crs=...` to target a different UTM zone or datum.
