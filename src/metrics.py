"""
Join population data to Census block groups and intersect with green space polygons
compute metrics per block group, such as green percentage
ultimately want to find a 1-100 green score. 
Saved as GeoJSON for easier viewing
Block group : small census area, used for neighborhood metrics
EPSG:3857 : makes projection in meters, geomtry.area gives square meters
"""
from pathlib import Path
import geopandas as gpd
import pandas as pd
# make requests to Census API, pull Census/ACS data
import requests
from cenpy import products

DATA_DIR = Path("data")
# point to data / processed
PROCESSED = DATA_DIR / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

def load_boundary():
    """
    Read saved city boundary GeoJSON, make sure lat/long coords
    """
    return gpd.read_file(PROCESSED / "city_boundary.geojson").to_crs(4326)

def load_greenspace():
    """
    Read green-space polygons from ingest file
    """
    # fallback incase primary file does not exist
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
    Get population by block group for the county chosen
    Falls back to ACS helpers if Census API route first
    """
    try:
        # hit officialAPI
        return _get_bg_population_via_census_api(2023)
    except Exception as e_api:
        print(f"2023 via Census API failed ({e_api}) - trying cenpy")
        try:
            # pull block group geometry
            acs = products.ACS(2023)
            gdf = acs.from_county(
                "San Diego County, California",
                level="block group",
                # total population variable
                variables=["B01003_001E"],
                geometry=True,
                # rename to be more readable (population)
            ).rename(columns={"B01003_001E": "population"}).to_crs(4326)
            return gdf
        except NotImplementedError:
            # If 2023 is not supported by cenpy use 2019
            # ran into this issue in initial tests
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
    use Census API to fetch population for each San Diego block group for given year
    builds GEOIDs and joins data to official block group shapes
    """
    try:
        # pygris fetches official Census boundary shapes
        from pygris import block_groups
    except Exception as e:
        raise RuntimeError("pygris is required for 2023 geometries. Install it from requirements.txt.") from e

    # population variable, FIPS codes for CA and SD county
    var = "B01001_001E"
    state_fips = "06"  
    county_fips = "073" 
    url = f"https://api.census.gov/data/{year}/acs/acs5"
    params = {
        "get": f"NAME,{var}",
        "for": "block group:*",
        "in": f"state:{state_fips} county:{county_fips}",
    }
    # web request to Census API
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    # parse data 
    data = resp.json()
    cols, rows = data[0], data[1:]
    # create pandas table using header as column names
    df = pd.DataFrame(rows, columns=cols)

    # creates concat for GEOID, keeps numeric population and GEOID
    df["GEOID"] = df["state"] + df["county"] + df["tract"] + df["block group"]
    df["population"] = pd.to_numeric(df[var], errors="coerce")
    df = df[["GEOID", "population"]]
    
    # fetches shapes for the year we use 
    geos = block_groups(state="CA", county="San Diego", year=year, cb=True)
    geos = geos.to_crs(4326)[["GEOID", "geometry"]]

    # return a GeoDataFrame, merge shapes from the population dataframe on GEOID
    gdf = geos.merge(df, on="GEOID", how="left")
    return gdf

def clip_to_city(gdf, city_boundary):
    """
    cuts geodataframe down to city boundary (only features inside city outline are valid)
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
    # convert to meters for area calculations
    city_bg_meters = city_bg.to_crs(3857)
    green_meters = greens.to_crs(3857)

    def _to_polygons_only(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        clean and standardize geometries
        """
        # keep only polygons and remove rows with missing geometry
        gdf = gdf[~gdf.geometry.isna()].copy()
        gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
        try:
            # fix shapes if possible
            gdf["geometry"] = gdf.geometry.buffer(0)
        except Exception:
            pass
        try:
            # split multi-polygons into individual
            gdf = gdf.explode(index_parts=True)
        except TypeError:
            gdf = gdf.explode()
        gdf = gdf.reset_index(drop=True)
        gdf = gdf[gdf.geometry.geom_type == "Polygon"].copy()
        return gdf

    city_bg_meters = _to_polygons_only(city_bg_meters)
    green_meters = _to_polygons_only(green_meters)
    
    # create piece of green polygon that makes up each block group - each row has GEOID and geometry piece
    intersect = gpd.overlay(green_meters, city_bg_meters[["GEOID", "geometry"]], how="intersection")

    # compute each area and sum GEOID to get green area per block group
    intersect["green_area"] = intersect.geometry.area
    green_by_bg = intersect.groupby("GEOID", as_index=False)["green_area"].sum()

    city_bg_meters["blockgroup_area"] = city_bg_meters.geometry.area

    # merge green area sums onto the table, based on GEOID
    output = city_bg_meters.merge(green_by_bg, on="GEOID", how="left")
    output["green_area"] = output["green_area"].fillna(0.0)

    #merge population back in from original city block group
    if "population" not in output.columns and "population" in city_bg.columns:
        output = output.merge(city_bg[["GEOID", "population"]], on="GEOID", how="left")
    
    output["green_percentage"] = (output["green_area"] / output["blockgroup_area"]).fillna(0.0)

    output["population"] = output["population"].fillna(0).clip(lower=0)
    output["green_area_per_person"] = output.apply(
        # calculate per row (per green area)
        lambda row: row["green_area"] / row["population"] if row["population"] > 0 else 0.0, axis = 1
    )

    def minmax(series):
        """
        normalize series to 0-1
        """
        s = series.fillna(0)
        lo, hi = s.min(), s.max()
        return (s - lo) / (hi - lo) if hi > lo else pd.Series([0.0] * len(s), index=s.index)
    
    # normalize since percentage and area per person are two different metrics/units
    percentage_norm = minmax(output["green_percentage"])
    per_person_norm = minmax(output["green_area_per_person"])
    
    # scale score to 1-100 and round
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
    



