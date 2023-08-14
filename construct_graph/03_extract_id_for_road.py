import osmnx as ox
import osmnx.graph
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely import wkb
from pathlib import Path
import matplotlib.pyplot as plt

from data_pipeline.dp03_road_graph import gkey_hash

import os
import logging
import argparse
import sys


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="This script extracts IDs of CoRe estimator for road graph (edges)")
    parser.add_argument(
        "-d",
        "--data_folder",
        type=str,
        help="Folder containing T4c data folder structure",
        required=True,
    )
    parser.add_argument(
        "-c",
        "--city",
        type=str,
        help="City to be processed",
        required=True,
        default=None,
    )
    parser.add_argument(
        "-i",
        "--id",
        type=str,
        help="ids to be extracted",
        required=False,
        default="lanes,maxspeed,name,highway",
    )
    return parser


def _string_to_list_of_int(s):
    s = s.strip("][").split(", ")

    s_ = []
    for _ in s:
        if "|" in s:
            s_.append(_.split("|")[0])
        else:
            s_.append(_)
    s = s_
    s = [int(_.strip("'").replace(" mph","").split("|")[0]) for _ in s]
    return s


HIGHWAY_KEY_TO_NUMBER = {
"motorway":0,"trunk":1,"primary":2,"secondary":3,"tertiary":4
}

def process_id_extraction(
        root_path:Path,
        city:str,
        id_of_interest:str,
    ):
    output_city_name = city
    city = city.split("_")[0]

    # Data from MeTS
    city_road_graph_folder = root_path / "road_graph" / city
    graph_file = city_road_graph_folder / "road_graph.gpkg"
    # Road table in training CoRe
    road_filtered_folder = root_path / "adj" / output_city_name
    road_filtered_file = road_filtered_folder / "road_graph_edges_filtered.parquet"
    road = pd.read_parquet(road_filtered_file)

    # IDs of interests
    ids = id_of_interest.split(",")

    if "highway" in ids:
        print("Extracting [highway] as id...")
        # by maxspeed
        road["id_highway"] = road["highway"].apply(lambda x: HIGHWAY_KEY_TO_NUMBER[x])
        print(f"ID-highway unique groups: {road['id_highway'].unique()}")
    if "lanes" in ids:
        print("Extracting [lanes num] as id...")
        # by lanes
        road["id_lanes"] = road["lanes"].apply(lambda s: max(_string_to_list_of_int(s)) if s != "nan" else -1)
        print(f"ID-Lanes unique groups: {road['id_lanes'].unique()}")
    if "maxspeed" in ids:
        print("Extracting [maxspeed limit] as id...")
        # by maxspeed
        road["id_maxspeed"] = road["maxspeed"].apply(lambda s: max(_string_to_list_of_int(s)) if s != "nan" else -1)
        print(f"ID-Maxspeed unique groups: {road['id_maxspeed'].unique()}")
    if "name" in ids:
        print("Extracting [road name] as id...")
        # Raw road graph from OSM
        g = gpd.read_file(graph_file, layer="edges")
        # gkey generation
        # As there are some rare notorious situations, where there are two different ways between the same two nodes
        # (u,v,osmid) is not unique, so let's generate hash based on geometry to differentiate.
        g["gkey"] = [gkey_hash(wkb.dumps(geo)) for geo in g["geometry"]]
        # Fiona has problems with uint64: https://github.com/Toblerity/Fiona/issues/365
        g["gkey"] = g["gkey"].astype("int64")

        # Join with raw road graph to get the road name
        road_with_name = road.merge(g[["name", "gkey", "geometry"]], on="gkey")
        # Convert to geodf
        road_with_name = gpd.GeoDataFrame(road_with_name, geometry=road_with_name["geometry_y"], crs=g.crs)
        road_with_name.drop(columns=["geometry_y"], inplace=True)

        # Remove spaces to format road name
        road_with_name["name"] = road_with_name["name"].apply(lambda x: x.replace(" ", ""))
        # Hash road name
        road_with_name["id_name"] = road_with_name["name"].apply(lambda x: gkey_hash(x.encode('utf-8')))
        road_with_name["id_name"] = road_with_name["id_name"].astype("int64")
        # Update training road table
        road = road_with_name
        print(f"ID-Name unique groups: {road['id_name'].unique().shape[0]}")
    # Save updated road table
    save_file = road_filtered_folder/"road_filtered_with_id.parquet"
    print(f"Updated road table is saving to {save_file}")
    road.to_parquet(save_file)

def main(argv):
    logging.basicConfig(
        level=os.environ.get("LOGLEVEL", "INFO"),
        format="[%(asctime)s][%(levelname)s][%(process)d][%(filename)s:%(funcName)s:%(lineno)d] %(message)s"
    )
    parser = create_parser()
    try:
        params = parser.parse_args(argv)
        params = vars(params)
        path = Path(params["data_folder"])
        city = params["city"]
        ioi = params["id"]
        #todo: add other params
    except Exception as e:
        logging.exception(f"Could not parse args.", exc_info=e)
        parser.print_help()
        exit(1)

    process_id_extraction(
        root_path = path,
        city = city,
        id_of_interest = ioi
    )


if __name__ == "__main__":
    main(sys.argv[1:])