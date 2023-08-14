# Construct data for traffic forecasting
1. Adjacency matrix

````
python construct_graph/generate_adjacency_matrix.py -d /Users/aaaje/Documents/ETH_WORK/thesis/data/t4f2022 -c melbourne -fr motorway
````

2. Training dataset

````
python construct_graph/generate_train_daily.py -d /Users/aaaje/Documents/ETH_WORK/thesis/data/t4f2022 -c melbourne -gad -fr motorway
````