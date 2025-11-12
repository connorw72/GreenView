"""
This module is meant to 
1. Get the San Diego city boundary
2. Pull public greenspace shapes from inside of the boundary using OpenStreetMap
Download the city boundary, download public greenspace polygons within boundary
Save both as GeoJSON files for easy viewing
GeoJSON: text file format for storing maps
EPSG:4326 : global coordinate system, standard used by GeoJSON and web maps
"""

# where do you get green space dictionary tags?
# what do you mean by rows with geometry?


# build file paths without hardcoding slashes
from pathlib import Path

import os

# cleanup data
import pandas as pd

# treat geospatial data as a table
import geopandas as gpd

# fetches from global, public map database(osm)
import osmnx as ox

# tell osm which geocode, and set output folder
HERE = Path(__file__).resolve()
ROOT = HERE.parent if HERE.parent.name != "src" else HERE.parent.parent
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

PLACE = "San Diego, California, USA"

# osm stores with key/value labels
GREEN_TAGS = {
    "leisure": ["park", "garden", "recreation_ground", "golf_course", "pitch", "playground", "dog_park"],
    "landuse": ["grass", "meadow", "village_green", "forest"],
    "natural": ["wood", "scrub", "heath", "grassland", "wetland", "water"],
    "boundary": ["protected_area"],
    "water": ["lake", "reservoir", "river", "pond", "canal"],
}

# only want public places
EXCLUDE_ACCESS = {"private", "no"}

def get_city_boundary(place: str) -> gpd.GeoDataFrame:
    """
    Place: str of city name
    Returns a GeoDataFrame with the geometry column
    """
    # get city polygon, ensure latitude/longitude w/ geometry column
    gdf = ox.geocode_to_gdf(place)
    return gdf.to_crs(epsg=4326)[["geometry"]]


def is_public(row):
    """
    Ensure non-public access rows are excluded 
    """
    access = str(row.get("access", "")).lower()
    return access == "" or access not in EXCLUDE_ACCESS

def get_osm_greenspace(boundary: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    boundary:

    Returns
    """
    # city polygon
    polygon = boundary.iloc[0].geometry
    layers = []
    # loop through each tag family and pull matching features
    for key, values in GREEN_TAGS.items():
        # featuresfrompolygon does not incluide osmid column for osmnx>=1.9
        G = ox.features_from_polygon(polygon, {key: values})
        if not G.empty:
            layers.append(G)
    
    if not layers:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    
    # combine cleanly, remove duplicates and no geometry
    gs = pd.concat(layers, ignore_index=True)
    # Prefer stable ID columns when available; fall back to geometry-only
    id_candidates = [
        "osmid",
        "osm_id",
        "@id",
        "id",
        "gnis:feature_id",
        "csp:globalid",
        "globalid",
        "global_id",
    ]
    existing_ids = [c for c in id_candidates if c in gs.columns]
    subset = existing_ids + (["geometry"] if "geometry" in gs.columns else [])
    if subset:
        gs = gs.drop_duplicates(subset=subset, keep="first")
    else:
        gs = gs.drop_duplicates(keep="first")
    gs = gs.loc[~gs.geometry.isna()].copy()

    # only public access and polygons
    gs = gs[gs.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
    # robust public access filter if access missing
    access_series = gs.get("access", pd.Series("", index=gs.index)).astype(str).str.lower()
    gs = gs[~access_series.isin(EXCLUDE_ACCESS)]

    # trim everything to city boundary, cannot stick out past edge
    gs = gpd.clip(gs.to_crs(4326), boundary.to_crs(4326))
    return gs[["geometry"]]

def main():
    print("Getting city boundary...")
    boundary = get_city_boundary(PLACE)
    boundary_out = DATA_PROCESSED / "city_boundary.geojson"
    boundary.to_file(boundary_out, driver="GeoJSON")
    print(f"Saved {boundary_out}")

    print("Getting OSM greenspace... this can take a minute")
    greens = get_osm_greenspace(boundary)
    greens_out = DATA_PROCESSED / "greenspace_raw.geojson"
    greens.to_file(greens_out, driver="GeoJSON")
    print(f"Saved {greens_out}")

    print(f"Greenspace polygons: {len(greens)}")

if __name__ == "__main__":
    main()
