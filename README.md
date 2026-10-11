# geomodeltools

Reusable helpers for 3D geological modeling workflows.

## What it includes

- Geometry densification and point extraction for line and polygon features.
- Inward polygon buffering with optional PyVista point cloud output, with spacings
  derived from the map scale and an adaptive mode for narrow polygons.
- DEM-based Z assignment using OpenTopography rasters.
- Quick EDA and contact plots for numeric and categorical unit attributes (`quick_eda`,
  `categorical_eda`, `cat_vs_cat`, `contact_plot`), also from the command line.
- A self-contained, interactive HTML report of the same plots (`write_html`): open it in
  any browser, no server, no Node, nothing to install.

## Install

From GitHub (a fixed version, no Git needed):

```bash
pip install https://github.com/ramaguirre/geomodeltools/archive/refs/tags/v0.3.0.zip
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

### Lines (faults, contacts, traces)

Lines need no buffering: densify them, drape them on the DEM, and export one point per
vertex. `densify_geometries` adds vertices so none is more than `spacing` apart (the
original vertices are kept), `add_z_from_opentopography` samples the DEM at every vertex,
and `geometries_to_points` writes x/y/z points that keep the line's attributes plus
`part_id` (for multi-part lines) and `vertex_id` (order along the line).

```python
from geomodeltools import densify_geometries, add_z_from_opentopography, geometries_to_points

faults = gpd.read_file("my_faults.shp")                  # LineString / MultiLineString
dense = densify_geometries(faults, spacing=25)           # a vertex at least every 25 m
draped, dem_path = add_z_from_opentopography(dense, out_tiff_path=Path("data/dem.tif"))
pts = geometries_to_points(draped)                       # x/y/z + attributes, per vertex
pts[["geometry_x", "geometry_y", "z", "name", "part_id", "vertex_id"]].to_csv("faults_xyz.csv", index=False)
```

The DEM value is the cell the vertex falls in (no interpolation), so pick `spacing`
near the DEM resolution (about 30 m for the 1 arc-second products); a finer spacing only
repeats values.

By default, `add_z_from_opentopography` assumes PSAD56 / UTM zone 19S
(EPSG:24879) for inputs with no CRS set, and writes the DEM GeoTIFF in that
same CRS. Pass `crs=...` to target a different UTM zone or datum.

## Quick EDA and contact plots

Exploratory plots for a drillhole/model table: how a numeric or categorical attribute
behaves overall, by geological unit, or across the boundary between two units. Every
function returns a matplotlib `Figure` (nothing is saved unless you call `fig.savefig(...)`
yourself), so they work the same in a notebook or a script.

```python
from geomodeltools import quick_eda, categorical_eda, cat_vs_cat, contact_plot

# Numeric variable, split by geological unit: probability plot, violin, CoV by unit
fig = quick_eda(df, value="cu_pct", category="lithology", log=True)

# A categorical column alone: frequency bar chart
fig = categorical_eda(df, category="alteration")

# Two categorical columns: alteration composition within each lithology
fig = cat_vs_cat(df, category_x="lithology", category_y="alteration")

# Contact plot: how cu_pct (or, with category=..., a categorical column) changes with
# distance from the boundary between two units. kind="profile" (default) draws a single
# mean/median line; kind="boxplot" draws a literal box per distance bin instead
# (Funciones_RAguirre.contact_boxplots' style).
fig = contact_plot(
    df, coords=["x", "y", "z"], unit="lithology",
    inside="Sulfide", outside="Waste", value="cu_pct", kind="boxplot",
)
```

Same thing from a terminal, no Python needed:

```bash
geomodeltools-eda data.csv --value cu_pct --category lithology --log -o eda.png
geomodeltools-contact data.csv --coords x y z --unit lithology \
    --inside Sulfide --outside Waste --value cu_pct -o contact.png
```

`contact_plot` (and the CLI's `--inside`/`--outside`) accept more than one unit code per
side, and an optional `contact=`/`--contact` code for points that sit exactly on the
boundary. `weight=`/`--weight` applies to every function above. See each function's
docstring for the rest of the knobs (`bin_width`, `n_bins`, `agg`, `normalize`, ...).

This started from the EDA and contact-plot functions in the private `Funciones_RAguirre`
repo (`quick_eda`, `violin_by_cat`, `contact_interactive_cutoff`, `bar_cat_vs_cat`...),
rewritten here with no Jupyter/seaborn dependency and a `scipy.spatial.cKDTree` distance
calculation in place of the original's batched `cdist` loop. That repo is unchanged and
stays the reference implementation.

The CoV panel in `quick_eda` is a scatter of std vs. mean, one labelled point per
category — points on the same line through the origin share a coefficient of variation —
matching `Plotting_Functions_RAguirre.var_coeff`, not a bar chart of `std / mean`.

## Soft-boundary (gradational contact) matrix of box plots

For a contact known to be gradational rather than sharp: a matrix of contact box plots
(`Funciones_RAguirre.contact_boxplots`' style — a literal box per distance bin, see above),
one panel per unit pair, each spanning that pair's own "how far is the gradation" distance.

```python
import pandas as pd
from geomodeltools import plot_soft_boundary_boxplots

# matrix.loc[neighbor, target] = how far the gradation extends (m); 0/missing = no panel
matrix = pd.DataFrame(0.0, index=["Sulfide", "Waste"], columns=["Sulfide", "Waste"])
matrix.loc["Waste", "Sulfide"] = 15.0   # Sulfide's panel spans +/-15 m across the contact
matrix.loc["Sulfide", "Waste"] = 20.0   # Waste's panel spans +/-20 m

fig = plot_soft_boundary_boxplots(df, coords=["x", "y", "z"], value="cu_pct", unit="lithology",
                                   soft_boundary_matrix=matrix, n_bins=8)
```

This is also useful on its own as `contact_plot(..., kind="boxplot")` for a single pair,
without a matrix.

`soft_boundary_values` is a separate, lower-level function: the per-unit "hard"
(strict within-unit) vs. "soft" (neighbours within `soft_boundary_matrix`'s threshold
pooled in) populations, generalising `Geostats_RA.soft_boundary_samples`'s merged-column
result (reusing this package's own `distance_to_contact`, not a second distance
implementation) with "hard"/"soft"/each neighbour's own slice kept separate instead of
merged into one column.

This generalises `Funciones_RAguirre.Geostats_RA.soft_boundary_samples` (same pooling
rule, via this package's own `distance_to_contact`) with "hard" and "soft" kept separate
instead of merged into one column, and the per-neighbour slices kept separate too. No
box-plot companion to that function exists in the reference code, so the one-panel-per-
unit matrix layout here is this package's own design, not a port — flag it if it doesn't
match what you had in mind.

## HTML report (interactive, no Python needed to view)

`write_html` builds one HTML file with dropdowns for every numeric/categorical column you
give it, Plotly charts (pan/zoom/hover), and a tab per kind of plot (Numeric, Categorical,
Pairs, Contacts). It is self-contained — Plotly is inlined, not loaded from a CDN — so it
opens straight from disk and can be emailed or dropped in a shared folder.

```python
from geomodeltools import write_html

write_html(
    df, "report.html",
    numeric=["cu_pct", "au_gpt"],        # each gets an overall panel + one per categorical column
    categorical=["lithology", "alteration"],  # each gets a frequency panel; auto-paired for composition
    contacts=[{                           # each scenario gets a profile and/or composition panel
        "coords": ["x", "y", "z"], "unit": "lithology",
        "inside": "Sulfide", "outside": "Waste",
        "value": "cu_pct", "category": "alteration",
        "label": "Sulfide / Waste contact",
    }],
    weight="length",
)
```

Or from a terminal (contact scenarios go in a small JSON file — see `build_report`'s
docstring for the full set of keys each one takes):

```bash
geomodeltools-report data.csv --numeric cu_pct au_gpt --categorical lithology alteration -o report.html
```

Everything is computed once in Python (`build_report`, via the same `*_data()` functions
`quick_eda`/`categorical_eda`/`cat_vs_cat` use for their static figures) and shipped to the
browser as JSON; the JS only turns that into Plotly traces; it never recomputes a
probability position, a weighted mean or a contact-distance bin itself — so the numbers in
the report and in a `quick_eda(...)` figure never drift apart. Only histogram/violin/bar
shapes are built from raw values client-side, which is Plotly's own job, the same as
matplotlib's `ax.hist`/`ax.violinplot` do it internally.

`src/geomodeltools/static/plotly.min.js` is vendored (copied from the `swath_local` repo's
own vendoring of it); see `THIRD-PARTY-NOTICES.md`. `static/report.js` is hand-written plain
JS with no build step — edit it directly, no bundler to run.
