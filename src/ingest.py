"""
This module is meant to 
1. Get the San Diego city boundary
2. Pull public greenspace shapes from inside of the boundary using OpenStreetMap
Download the city boundary, download public greenspace polygons within boundary
Save both as GeoJSON files for easy viewing
GeoJSON: text file format for storing maps
EPSG:4326 : global coordinate system, standard used by GeoJSON and web maps
"""

# build file paths without hardcoding slashes
from pathlib import Path
# cleanup data, concatenate, de-duplicate
import pandas as pd
# use GeoDataFrame operations - clip, convert coords, saveGeoJSON
import geopandas as gpd
# fetches from global, public map database(osm)
import osmnx as ox

# absolute path to file, resolve root regardless of src/
HERE = Path(__file__).resolve()
ROOT = HERE.parent if HERE.parent.name != "src" else HERE.parent.parent
DATA_PROCESSED = ROOT / "data" / "processed"
# create folder if missing
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

PLACE = "San Diego, California, USA"

# osm stores with key/value labels
# map of osm keys to "green space"
GREEN_TAGS = {
    "leisure": ["park", "garden", "recreation_ground", "golf_course", "pitch", "playground", "dog_park"],
    "landuse": ["grass", "meadow", "village_green", "forest"],
    "natural": ["wood", "scrub", "heath", "grassland", "wetland", "water"],
    "boundary": ["protected_area"],
    "water": ["lake", "reservoir", "river", "pond", "canal"],
}

# want to remove if access is private or no
EXCLUDE_ACCESS = {"private", "no"}

def get_city_boundary(place: str) -> gpd.GeoDataFrame:
    """
    Returns polygon outline of a place
    """
    gdf = ox.geocode_to_gdf(place)
    # coordinates are lat/long, only returns geometry column
    return gdf.to_crs(epsg=4326)[["geometry"]]

def get_osm_greenspace(boundary: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Fetch all green space features inside boundary
    """
    # take first row's geometry(city outline polygon)
    polygon = boundary.iloc[0].geometry
    layers = []
    for key, values in GREEN_TAGS.items():
        # get all features from osm within polygon where tag matches
        G = ox.features_from_polygon(polygon, {key: values})
        if not G.empty:
            layers.append(G)
    
    if not layers:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    
    # stack tag-result tables into one big table
    gs = pd.concat(layers, ignore_index=True)
    # potential ways to uniquely identify a feature
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
    # use ID and geometry to drop duplicates from rows
    if subset:
        gs = gs.drop_duplicates(subset=subset, keep="first")
    else:
        gs = gs.drop_duplicates(keep="first")
    # remove rows w/o geometry
    gs = gs.loc[~gs.geometry.isna()].copy()

    #only keep area shapes (polygons)
    gs = gs[gs.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
    # read access tag and drop if it should be excluded (private, no) 
    access_series = gs.get("access", pd.Series("", index=gs.index)).astype(str).str.lower()
    gs = gs[~access_series.isin(EXCLUDE_ACCESS)]

    # convert to lat/long system, cut shapes to fit city outline
    gs = gpd.clip(gs.to_crs(4326), boundary.to_crs(4326))
    # return only the polygons
    return gs[["geometry"]]

def main():
    print("Getting city boundary")
    boundary = get_city_boundary(PLACE)
    boundary_out = DATA_PROCESSED / "city_boundary.geojson"
    # save boundary as GeoJSON file to output path
    boundary.to_file(boundary_out, driver="GeoJSON")

    print("Getting OSM greenspace...")
    greens = get_osm_greenspace(boundary)
    greens_out = DATA_PROCESSED / "greenspace_raw.geojson"
    # save green spaces as GeoJSON file to output path
    greens.to_file(greens_out, driver="GeoJSON")
    print(f"Greenspace polygons: {len(greens)}")

if __name__ == "__main__":
    main()
