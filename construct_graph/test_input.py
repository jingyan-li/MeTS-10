import pandas as pd
import numpy as np

#%%
# Speed data
path = "/Users/aaaje/Documents/ETH_WORK/thesis/data/t4f2022/speed_classes/melbourne/speed_classes_2020-06-01.parquet"

df = pd.read_parquet(path)

road_path = "/Users/aaaje/Documents/ETH_WORK/thesis/data/t4f2022/road_graph/melbourne/road_graph_edges.parquet"

road = pd.read_parquet(road_path)

# Check if gkey of road segment is unique
print("Unique of gkey: ", road["gkey"].unique().shape == road.shape[0])

#%%
# Filter essential roads
gkeys = road[road["highway"].isin(["motorway","trunk","primary","secondary"])]["gkey"].tolist()
df = df[df["gkey"].isin(gkeys)]

#%%
# Stats for each road segments
road_stats = df.groupby(by="gkey").agg({"t":"count","volume":"median","median_speed_kph":"median","std_speed_kph":"median"})



#%%
# constrcut road adjacency matrix
def run_node_join(df_left, df_right):
    intersections = df_left.join(df_right, how="inner", lsuffix="_left")
    intersections.index = np.arange(intersections.shape[0])
    # drop self intersection
    intersections.drop(index=intersections[intersections["gkey_left"]==intersections["gkey"]].index,
                       inplace=True)
    intersections["edge_weight"] = (intersections["length_meters_left"] + intersections["length_meters"])/2.0
    return intersections[["gkey_left","gkey","edge_weight"]].to_numpy()

# adjacency matrix by node overlap
sp_df = road[["u","gkey","length_meters"]].rename(columns={"u":"p"}).set_index("p")
ep_df = road[['v',"gkey","length_meters"]].rename(columns={"v":"p"}).set_index("p")
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
#%%

