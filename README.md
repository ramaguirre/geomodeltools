# geomodeltools

Reusable helpers for 3D geological modeling workflows.

## What it includes

- Geometry densification and point extraction for line and polygon features.
- Inward polygon buffering with optional PyVista point cloud output, with spacings
  derived from the map scale and an adaptive mode for narrow polygons.
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

## Sample data

Public SERNAGEOMIN geological map GIS packages (source and copyright: SERNAGEOMIN)
are attached to the
[`sample-data-sernageomin`](https://github.com/ramaguirre/geomodeltools/releases/tag/sample-data-sernageomin)
release, not committed to the repo. Download them into `sample_data/sernageomin/`
(git-ignored):

```bash
gh release download sample-data-sernageomin -R ramaguirre/geomodeltools -D sample_data/sernageomin
```

### Example notebook

[`notebooks/example_sernageomin_m201_points.ipynb`](notebooks/example_sernageomin_m201_points.ipynb)
runs the full workflow on the M201 *Carrizalillo – El Tofo* 1:100,000 map. It downloads the
release zip itself, clips the geological units to an area of interest of about 37 × 55 km,
reprojects to PSAD56 / UTM 19S, buffers the polygons into about 207k labelled points with
`map_scale=100_000, adaptive=True`, optionally adds DEM elevations, and exports GeoPackage,
CSV and VTK files to `notebooks/outputs/m201_example/`. It also compares fixed rings with
the adaptive mode.

The notebook also shows a datum pitfall. Converting SIRGAS 1995 (the CRS of the SERNAGEOMIN
data, EPSG:31994) directly to EPSG:24879 with `to_crs` uses PROJ's *ballpark* transformation,
which applies no datum shift and leaves every point about 460 m out. Go through WGS 84 instead:
`gdf.to_crs(4326).to_crs(24879)`. That route uses EPSG *PSAD56 to WGS 84 (16)*, which is
accurate to about 17 m.

## Quick example

```python
from pathlib import Path
import geopandas as gpd
from geomodeltools import bufferize_2d_polygons, add_z_from_opentopography

gdf = gpd.read_file("my_polygons.shp")
# map_scale sets simplification, point spacing and buffer distances from the map's
# drawing precision; adaptive=True gives narrow units (dykes, thin beds) a centreline
# so no polygon is left without points. See the bufferize_2d_polygons docstring.
pts = bufferize_2d_polygons(
    gdf, feature_cols=["unit"], map_scale=100_000, adaptive=True, return_polydata=False
)
pts_z, dem_path = add_z_from_opentopography(
    pts,
    out_tiff_path=Path("data/dem.tif"),
    margin_m=0,
)
```

By default, `add_z_from_opentopography` assumes PSAD56 / UTM zone 19S
(EPSG:24879) for inputs with no CRS set, and writes the DEM GeoTIFF in that
same CRS. Pass `crs=...` to target a different UTM zone or datum.
