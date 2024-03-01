import os
import pandas as pd

from config import PATH_DATA_DIR

def load_and_align_data(data_path, data_sources, label_key="YIELD"):
    data_dfs = {}
    # load the data
    for src in data_sources:
        index_cols = data_sources[src]["index_cols"]
        data_csv = os.path.join(data_path, data_sources[src]["filename"])
        df = pd.read_csv(data_csv, index_col=index_cols)
        data_dfs[src] = df[data_sources[src]["sel_cols"]]

    # align label data
    label_df = data_dfs["YIELD"]
    print("before merging", len(label_df.index))
    for src in data_dfs:
        if (src == label_key):
            continue

        df = data_dfs[src]
        if (len(df.index.names) == 1):
            # get intersection of locations
            src_values = set(df.index.get_level_values(0))
            label_values = set(label_df.index.get_level_values(0))
            common_values = set.intersection(label_values, src_values)
            # filter both by intersection
            df = df.loc[common_values]
            label_df = pd.concat([label_df.xs(key, level=0, drop_level=False) for key in common_values])
        else:
            label_values = label_df.index.values
            # get values for index level 0 and 1
            src_values = df.index.get_level_values()
            # intersect the tuples

        print("after merging", src, len(label_df.index))
        print(label_df.sort_index().head())

    # align other data
    for src in data_dfs:
        if (src == label_key):
            continue

        df = data_dfs[src]
        df = df.merge(df, on=df.index.names)

    return data_dfs


data_sources = {
    "YIELD" : {
        "filename" : "YIELD_COUNTY_US.csv",
        "index_cols" : ["COUNTY_ID", "FYEAR"],
        "sel_cols" : ["YIELD"]
    },
    "SOIL" : {
        "filename" : "SOIL_COUNTY_US.csv",
        "index_cols" : ["COUNTY_ID"],
        "sel_cols" : ["SM_WHC"]
    },
    "REMOTE_SENSING" : {
        "filename" : "REMOTE_SENSING_COUNTY_US.csv",
        "index_cols" : ["COUNTY_ID", "FYEAR", "DEKAD"],
        "sel_cols" : ["FAPAR"]
    }
}

data_path = os.path.join(PATH_DATA_DIR, "data_US", "county_data")
data_dfs = load_and_align_data(data_path, data_sources)