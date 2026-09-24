# OpenTopography API key. No key is bundled with the package: each user sets
# their own in the OPENTOPOGRAPHY_API_KEY environment variable (see README),
# or passes api_key=... explicitly to the function.
OPENTOPOGRAPHY_API_KEY = None

# Default assumed/output CRS: PSAD56 / UTM zone 19S.
# Used to fill in a missing CRS on input geometries, and as the default
# output CRS for the DEM GeoTIFF and output GeoDataFrame. Override per-call
# via the `crs=` parameter (accepts anything pyproj/geopandas/rasterio can
# parse: EPSG int, "EPSG:xxxx" string, WKT, proj4, etc.).
DEFAULT_CRS = "EPSG:24879"