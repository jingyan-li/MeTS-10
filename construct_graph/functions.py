import geopandas as gpd
from shapely import from_wkb
import pandas as pd


def to_geodf(file):
    road = pd.read_parquet(file)
    road["geometry"] = road["geometry"].apply(lambda x: from_wkb(x))
    road = gpd.GeoDataFrame(road, geometry=road["geometry"], crs="4326")
    return road