# geomodeltools

Reusable helpers for 3D geological modeling workflows.

## What it includes

- Geometry densification and point extraction for line and polygon features.
- Inward polygon buffering with optional PyVista point cloud output.
- DEM-based Z assignment using OpenTopography rasters.

## Install

```bash
pip install -e .
```

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
    bounds_utm=True,
    margin=0,
)
```
