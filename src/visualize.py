"""
Loads computed metrics (one row per block group)
Ensures lat/long coordinates, with a 1-100 score
Build an interactive map with Folium, saves as an HTML
"""

from pathlib import Path
import geopandas as gpd
import numpy as np
# build interactive browser ready map
import folium
# build color ramps to show score range
from branca.colormap import linear

# GeoJSON from metrics.py
DATA_PATH = Path("data/processed/bg_metrics.geojson")
OUT_DIR = Path("maps")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_HTML = OUT_DIR / "greenview_sandiego.html"

gdf = gpd.read_file(DATA_PATH)

# must be lat/long coords
if gdf.crs is None or gdf.crs.to_epsg() != 4326:
    gdf = gdf.to_crs(4326)

# pick the green score from each column in the data, None if it DNE
CANDIDATES = ["green_score"]
score_column = next((c for c in CANDIDATES if c in gdf.columns), None)
if score_column is None:
    raise ValueError(
        f"Couldn't find any candidates in {list(gdf.columns)}.\n"
    )

# format scores (floats, NaN's, etc.)
vals = gdf[score_column].astype(float).replace([np.inf, -np.inf], np.nan)
if vals.max() <= 1.0:
    # normalize scores between 0-1
    norm = (vals - vals.min()) / (vals.max() - vals.min() + 1e-9)
    gdf["greenvalue_score_100"] = np.round(norm * 99 + 1)
else:
    # percentile rank values, convert into 1-100 scores
    ranks = vals.rank(pct = True)
    gdf["greenvalue_score_100"] = np.round(ranks * 99 + 1)

NAME_CANDIDATES = ["NAME", "name", "GEOID", "geoid", "block_group"]
name_column = next((c for c in NAME_CANDIDATES if c in gdf.columns), None)
if name_column is None:
    name_column = gdf.columns[0]
    
# get bounds of the geometries
west, south, east, north = gdf.total_bounds
center_lat = (south + north) / 2
center_lon = (west + east) / 2
# create Folium map centered on data
green_map = folium.Map(location=[center_lat, center_lon], zoom_start=10, tiles="cartodbpositron")

# yellow to green color map
color_map = linear.YlGn_09.scale(1, 100)
color_map.caption = "GreenView Score 1-100"

# style for each polygon
style_fn = lambda feat: {
    "fillColor": color_map(feat["properties"]["greenvalue_score_100"]),
    "color": "#555555",
    "weight": 0.4,
    "fillOpacity": 0.8
}


tooltip = folium.features.GeoJsonTooltip(
    # columns to show on hover
    fields = [name_column, "greenvalue_score_100"],
    aliases = ["Location:", "GreenView Score: "],
    localize=True,
    sticky=False
)

# add gdf as a geoJSON layer
folium.GeoJson(
    gdf,
    name="GreenView",
    style_function=style_fn,
    tooltip=tooltip,
    # hover visual
    highlight_function= lambda f: {'weight': 2, "color": "#000000"}
).add_to(green_map)

# color legend
color_map.add_to(green_map)

green_map.fit_bounds([[south, west], [north, east]])

green_map.save(str(OUT_HTML))
print(f"Saved map {OUT_HTML.resolve()}")
