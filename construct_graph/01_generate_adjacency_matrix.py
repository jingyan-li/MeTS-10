import pandas as pd
import numpy as np
import argparse
import sys
from pathlib import Path
import os
import logging


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="This script constructs adjacency matrix from road graph (edges)")
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
        "-fr",
        "--filter_road",
        type=str,
        help="Filter and only preserve essential roads.",
        required=False,
        default="motorway,trunk,primary,secondary",
    )
    return parser


def process_adj_generation(
        root_path:Path,
        city:str,
        filter_road:str=None,
        filter_road_file=None,
):
    output_city_name = city
    city = city.split("_")[0]

    road_path = root_path / "road_graph" / city / "road_graph_edges.parquet"
    road = pd.read_parquet(road_path)
    print(f"The adjacency matrix has {road.shape[0]} nodes")
    # output directory
    output_dir = root_path / "adj" / output_city_name
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    # filter roads by road class
    if filter_road!=None:
        if filter_road == "from_file":
            filtered = pd.read_csv(filter_road_file, index_col=None)
            road = road.merge(filtered, on="gkey")
        else:
            road = road[road["highway"].isin(filter_road.split(","))]
        print(f"After filtering, the adjacency matrix has {road.shape[0]} nodes")

    # add primary key by index
    road["road_index"] = np.arange(road.shape[0])
    # save road table
    road.to_parquet(output_dir / "road_graph_edges_filtered.parquet")

    # adjacency matrix by node overlap
    sp_df = road[["u", "road_index", "length_meters"]].rename(columns={"u": "p"}).set_index("p")
    ep_df = road[['v', "road_index", "length_meters"]].rename(columns={"v": "p"}).set_index("p")
    # join
    intersection_spsp = run_node_join(sp_df, sp_df)
    intersection_spep = run_node_join(sp_df, ep_df)
    intersection_epep = run_node_join(ep_df, ep_df)
    intersection_epsp = run_node_join(ep_df, sp_df)

    adj = np.concatenate([
        intersection_spsp,
        intersection_spep,
        intersection_epsp,
        intersection_epep
    ])

    # Remove nan values
    adj = adj[~np.isnan(adj).any(axis=1), :].astype(int)
    # remove redundant pairs
    adj = np.unique(adj, axis=0)
    print(f"The adjacency matrix has {adj.shape[0]} valid edges")

    # save adjacency matrix:
    print(f"Saving adjacency matrix to {output_dir}")
    # linked list
    save_name = "adjacency_mat" if filter_road==None else "adjacency_mat_filtered"
    np.save(output_dir / f"{save_name}.npy", adj, allow_pickle=True) # (#edges, 3) ->(road_segment_gkey, road_segment_gkey2, edge_weight=road_length)
    # full matrix
    nodes_len = road.shape[0]
    adj_full = np.diag(np.ones(nodes_len))
    adj_full[adj[:,0], adj[:,1]] = 1  # todo: change edge weight of the graph to road_length
    adj_full[adj[:,1], adj[:,0]] = 1
    np.savetxt(output_dir / f"{save_name}.csv", adj_full, delimiter=",")


def run_node_join(df_left, df_right, primary_key = "road_index"):
    intersections = df_left.join(df_right, how="inner", lsuffix="_left")
    intersections.index = np.arange(intersections.shape[0])
    # drop self intersection
    intersections.drop(index=intersections[intersections[primary_key+"_left"]==intersections[primary_key]].index,
                       inplace=True)
    intersections["edge_weight"] = (intersections["length_meters_left"] + intersections["length_meters"])/2.0
    return intersections[[primary_key+"_left",primary_key,"edge_weight"]].to_numpy()


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
        filter_road = params["filter_road"]
        #todo: add other params
    except Exception as e:
        logging.exception(f"Could not parse args.", exc_info=e)
        parser.print_help()
        exit(1)

    filter_road_file = path / "adj" / city / "filtered_road_gkeys.csv"
    process_adj_generation(
        root_path = path,
        city = city,
        filter_road = filter_road,
        filter_road_file=filter_road_file
    )


if __name__ == "__main__":
    main(sys.argv[1:])