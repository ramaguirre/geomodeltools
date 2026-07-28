# Default OpenTopography API key bundled with the package.
# Override at runtime by setting the OPENTOPOGRAPHY_API_KEY environment variable
# or by passing api_key=... explicitly to the function.
OPENTOPOGRAPHY_API_KEY = "a4176ae09b21a2d6753b1e7a84338da6"

# Default assumed/output CRS: PSAD56 / UTM zone 19S.
# Used to fill in a missing CRS on input geometries, and as the default
# output CRS for the DEM GeoTIFF and output GeoDataFrame. Override per-call
# via the `crs=` parameter (accepts anything pyproj/geopandas/rasterio can
# parse: EPSG int, "EPSG:xxxx" string, WKT, proj4, etc.).
DEFAULT_CRS = "EPSG:24879"