"""
Load processed GeoJSON 
Ensure it is in the right CRS, compute and confirm score
Render and save interactive Folium
"""

from pathlib import Path
import geopandas as gpd
import pandas as pd
import numpy as np
import folium
from branca.colormap import linear

DATA_PATH = Path("data/processed/bg_metrics.geojson")
OUT_DIR = Path("maps")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_HTML = OUT_DIR / "greenview_sandiego.html"

gdf = gpd.read_file(DATA_PATH)

if gdf.crs is None or gdf.crs.to_epsg() != 4326:
    gdf = gdf.to_crs(4326)

CANDIDATES = ["green_score"]
score_column = next((c for c in CANDIDATES if c in gdf.columns), None)

if score_column is None:
    raise ValueError(
        f"Couldn't find any candidates in {list(gdf.columns)}.\n"
    )

vals = gdf[score_column].astype(float).replace([np.inf, -np.inf], np.nan)
if vals.max() <= 1.0:
    norm = (vals - vals.min()) / (vals.max() - vals.min() + 1e-9)
    gdf["greenvalue_score_100"] = np.round(norm * 99 + 1)
else:
    ranks = vals.rank(pct = True)
    gdf["greenvalue_score_100"] = np.round(ranks * 99 + 1)

NAME_CANDIDATES = ["NAME", "name", "GEOID", "geoid", "block_group"]
name_column = next((c for c in NAME_CANDIDATES if c in gdf.columns), None)
if name_column is None:
    name_column = gdf.columns[0]

"""
Compute a sensible map center without deprecated APIs. Use the dataset
bounds to find the center, and optionally fit to bounds after adding layers.
"""
west, south, east, north = gdf.total_bounds
center_lat = (south + north) / 2
center_lon = (west + east) / 2
green_map = folium.Map(location=[center_lat, center_lon], zoom_start=10, tiles="cartodbpositron")

color_map = linear.YlGn_09.scale(1, 100)
color_map.caption = "GreenView Score 1-100"

style_fn = lambda feat: {
    "fillColor": color_map(feat["properties"]["greenvalue_score_100"]),
    "color": "#555555",
    "weight": 0.4,
    "fillOpacity": 0.8
}

tooltip = folium.features.GeoJsonTooltip(
    fields = [name_column, "greenvalue_score_100", score_column],
    aliases = ["Area:", "GreenView Score: ", f"Raw Metric ({score_column})"],
    localize=True,
    sticky=False
)

folium.GeoJson(
    gdf,
    name="GreenView",
    style_function=style_fn,
    tooltip=tooltip,
    highlight_function= lambda f: {'weight': 2, "color": "#000000"}
).add_to(green_map)

color_map.add_to(green_map)
# Add layer control properly (instantiate then add)
folium.LayerControl().add_to(green_map)

# Ensure the map view includes all features
green_map.fit_bounds([[south, west], [north, east]])

green_map.save(str(OUT_HTML))
print(f"Saved map {OUT_HTML.resolve()}")
