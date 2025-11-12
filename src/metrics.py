"""

"""

import geopandas as gpd
import pandas as pd
import requests
from pathlib import Path
from cenpy import products

DATA_DIR = Path("data")
PROCESSED = DATA_DIR / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

def load_boundary():
    """
    
    """
    return gpd.read_file(PROCESSED / "city_boundary.geojson").to_crs(4326)

def load_greenspace():
    """
    
    """
    primary = PROCESSED / "greenspace_raw.geojson"
    fallback = PROCESSED / "osm_greenspace.geojson"
    path = primary if primary.exists() else fallback
    if not path.exists():
        raise FileNotFoundError(
            f"Greenspace file not found. Expected one of: {primary} or {fallback}. "
            "Run `python -m src.ingest` first to generate it."
        )
    return gpd.read_file(path).to_crs(4326)

def get_blockgroups_population():
    """
    
    """
    try:
        return _get_bg_population_via_census_api(2023)
    except Exception as e_api:
        print(f"2023 via Census API failed ({e_api}); trying cenpy...")
        try:
            acs = products.ACS(2023)
            gdf = acs.from_county(
                "San Diego County, California",
                level="block group",
                variables=["B01003_001E"],
                geometry=True,
            ).rename(columns={"B01003_001E": "population"}).to_crs(4326)
            return gdf
        except NotImplementedError:
            fallback_year = 2019
            print(
                f"ACS 2023 not supported by this cenpy version; falling back to {fallback_year}."
            )
            acs = products.ACS(fallback_year)
            gdf = acs.from_county(
                "San Diego County, California",
                level="block group",
                variables=["B01003_001E"],
                geometry=True,
            ).rename(columns={"B01003_001E": "population"}).to_crs(4326)
            return gdf

def _get_bg_population_via_census_api(year: int) -> gpd.GeoDataFrame:
    """
    Fetch block group population for San Diego County using the Census API
    and join to 2023 geometries from pygris.
    """
    try:
        from pygris import block_groups
    except Exception as e:
        raise RuntimeError("pygris is required for 2023 geometries. Install it from requirements.txt.") from e

    var = "B01001_001E"
    state_fips = "06"  
    county_fips = "073" 
    url = f"https://api.census.gov/data/{year}/acs/acs5"
    params = {
        "get": f"NAME,{var}",
        "for": "block group:*",
        "in": f"state:{state_fips} county:{county_fips}",
    }
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    cols, rows = data[0], data[1:]
    df = pd.DataFrame(rows, columns=cols)

    df["GEOID"] = df["state"] + df["county"] + df["tract"] + df["block group"]
    df["population"] = pd.to_numeric(df[var], errors="coerce")
    df = df[["GEOID", "population"]]

    geos = block_groups(state="CA", county="San Diego", year=year, cb=True)
    geos = geos.to_crs(4326)[["GEOID", "geometry"]]

    gdf = geos.merge(df, on="GEOID", how="left")
    return gdf

def clip_to_city(gdf, city_boundary):
    """

    """
    return gpd.clip(gdf, city_boundary)

def compute_metrics(city_bg, greens):
    """
    Compute greenspace metrics
    green_area: total green area in each block group
    blockgroup_area: total block group land area
    green_percentage: green_area / blockgroup_area
    green_area_per_person: green_area / population
    green_score: 1-100 socre normalized combining % green + per person area
    """
    city_bg_meters = city_bg.to_crs(3857)
    green_meters = greens.to_crs(3857)

    def _to_polygons_only(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        gdf = gdf[~gdf.geometry.isna()].copy()
        gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
        try:
            gdf["geometry"] = gdf.geometry.buffer(0)
        except Exception:
            pass
        try:
            gdf = gdf.explode(index_parts=True)
        except TypeError:
            gdf = gdf.explode()
        gdf = gdf.reset_index(drop=True)
        gdf = gdf[gdf.geometry.geom_type == "Polygon"].copy()
        return gdf

    city_bg_meters = _to_polygons_only(city_bg_meters)
    green_meters = _to_polygons_only(green_meters)

    intersect = gpd.overlay(green_meters, city_bg_meters[["GEOID", "geometry"]], how="intersection")

    intersect["green_area"] = intersect.geometry.area
    green_by_bg = intersect.groupby("GEOID", as_index=False)["green_area"].sum()

    city_bg_meters["blockgroup_area"] = city_bg_meters.geometry.area

    output = city_bg_meters.merge(green_by_bg, on="GEOID", how="left")
    output["green_area"] = output["green_area"].fillna(0.0)

    if "population" not in output.columns and "population" in city_bg.columns:
        output = output.merge(city_bg[["GEOID", "population"]], on="GEOID", how="left")
    
    output["green_percentage"] = (output["green_area"] / output["blockgroup_area"]).fillna(0.0)

    output["population"] = output["population"].fillna(0).clip(lower=0)
    output["green_area_per_person"] = output.apply(
        lambda row: row["green_area"] / row["population"] if row["population"] > 0 else 0.0, axis = 1
    )

    def minmax(series):
        s = series.fillna(0)
        lo, hi = s.min(), s.max()
        return (s - lo) / (hi - lo) if hi > lo else pd.Series([0.0] * len(s), index=s.index)
    
    percentage_norm = minmax(output["green_percentage"])
    per_person_norm = minmax(output["green_area_per_person"])

    output["green_score"] = (0.5 * percentage_norm + 0.5 * per_person_norm) * 99 + 1
    output["green_score"] = output["green_score"].round(1).clip(1, 100)

    return output.to_crs(4326)[
        ["GEOID", "population", "blockgroup_area", "green_area", "green_percentage", "green_area_per_person", "green_score", "geometry"]
    ]

def main():
    city = load_boundary()
    greens = load_greenspace()

    blockgroup_pop = get_blockgroups_population()
    city_blockgroup = clip_to_city(blockgroup_pop, city)

    city_score = compute_metrics(city_blockgroup, greens)

    outpath = PROCESSED / "bg_metrics.geojson"
    city_score.to_file(outpath, driver="GeoJSON")
    print(f"Saved metrics : {outpath}")

if __name__ == "__main__":
    main()
    



