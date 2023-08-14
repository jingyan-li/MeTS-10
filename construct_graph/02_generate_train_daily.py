import pandas as pd
import numpy as np
import argparse
import sys
from pathlib import Path
import os
import logging
import h5py
from tqdm import tqdm

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
        "-gad",
        "--generate_all_days",
        help="Generate all days records under the data folder",
        required=False, action="store_true"
    )
    parser.add_argument(
        "-sd",
        "--start_date",
        type=str,
        help="Start date to generate train daily dataset",
        required=False,
        default="2020-06-01",
    )
    parser.add_argument(
        "-ed",
        "--end_date",
        type=str,
        help="End date to generate train daily dataset",
        required=False,
        default="2020-06-02",
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


# Fill in the empty data
def fill_zeros_with_last(arr):
    prev = np.arange(len(arr))
    prev[arr == 0] = 0
    prev = np.maximum.accumulate(prev)
    return arr[prev]


def fill_zeros_with_after(arr):
    return fill_zeros_with_last(arr[::-1])[::-1]


def fill_zeros(arr):
    fz = fill_zeros_with_last(arr)
    return fill_zeros_with_after(fz)


def get_speed_info(selected, feature="median_speed_kph"):
    y = np.zeros(96)
    y[selected["t"].values] = selected[feature]
    # fill the empty values by linear interpolation
    ma = fill_zeros(y)
    return ma


def process_train_daily(road_path:Path,
                        data_path:Path,
                        output_path:Path,
                        generate_all_days:bool,
                        start_date:str,
                        end_date:str,
                        filter_road:str="motorway,trunk",
                        filter_road_file=None,
                        ):

    print(f"Processing... \ntrain daily files will be saved to {output_path}")
    # Read all roads
    road = pd.read_parquet(road_path)

    # Filter essential roads
    if filter_road == "from_file":
        filtered = pd.read_csv(filter_road_file, index_col=None)
        road = road.merge(filtered, on="gkey")
        gkeys = road["gkey"].tolist()
    else:
        gkeys = road[road["highway"].isin(filter_road.split(","))]["gkey"].tolist()

    # total dates
    if generate_all_days:
        dates = sorted([_.stem.split("_")[-1] for _ in list(data_path.rglob("*.parquet"))])
    else:
        dates = pd.date_range(start=start_date, end=end_date).tolist()
        dates = [_.strftime("%Y-%m-%d") for _ in dates]

    # process records for each day
    for day in tqdm(dates):
        # daily speed for all roads
        df = pd.read_parquet(data_path / f"speed_classes_{day}.parquet")
        all_data_daily = []
        for gkey in gkeys:
            # generate daily speed for one road
            sroad = df[df['gkey'] == gkey]
            if sroad.shape[0] == 0:
                print(f"road {gkey} has no record today")

            # Linear interpolation
            y = np.stack([
                get_speed_info(sroad, "median_speed_kph"),
                get_speed_info(sroad, "std_speed_kph"),
                get_speed_info(sroad, "congestion_factor"),
                get_speed_info(sroad, "volume_class"),
                get_speed_info(sroad, "volume"),
            ], axis=1)
            all_data_daily.append(y)

        all_data_daily = np.stack(all_data_daily, axis=1) # (timestamps=96, #roads, #features)
        # print(all_data_daily.shape)
        # Write into h5py file
        f = h5py.File(output_path/f"speed_filtered_{day}.h5","w")
        dset = f.create_dataset("data", data=all_data_daily)

        f.close()


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
        generate_all_days = params["generate_all_days"]
        start_date = params["start_date"]
        end_date = params["end_date"]
        # todo: add other params
    except Exception as e:
        logging.exception(f"Could not parse args.", exc_info=e)
        parser.print_help()
        exit(1)

    output_city_name = city
    city = city.split("_")[0]

    output_dir = path / "speed_train_daily" / output_city_name
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    filter_road_file = path / "adj" / output_city_name / "filtered_road_gkeys.csv"
    
    process_train_daily(
        road_path=path / "road_graph" / city / "road_graph_edges.parquet",
        data_path=path / "speed_classes" / city,
        output_path=output_dir,
        generate_all_days=generate_all_days,
        start_date=start_date,
        end_date=end_date,
        filter_road=filter_road,
        filter_road_file=filter_road_file
    )


if __name__ == "__main__":
    main(sys.argv[1:])