import os
import pandas as pd

from config import PATH_DATA_DIR


def load_data_csv(data_path, data_sources):
    data_dfs = {}
    for src in data_sources:
        data_csv = os.path.join(data_path, data_sources[src]["filename"])
        df = pd.read_csv(data_csv, header=0)
        data_dfs[src] = df

    return data_dfs

def align_data(data_sources, label_df, feature_dfs, label_key="YIELD"):
    # align label data
    for src in feature_dfs:
        df = feature_dfs[src]
        index_cols = data_sources[src]["index_cols"]

        # static data
        if (len(index_cols) == 1):
            index_cols = data_sources[src]["index_cols"]
        else:
            index_cols = data_sources[label_key]["index_cols"]

        label_df = label_df.merge(df[index_cols].drop_duplicates(), on=index_cols, how="inner")
        print("after merging", len(label_df.index))
        print(label_df.sort_values(by=["COUNTY_ID", "FYEAR"]).head())

    # align feature data
    for src in feature_dfs:
        df = feature_dfs[src]
        print("before merging", src, len(df.index))
        index_cols = data_sources[src]["index_cols"]

        # static data
        if (len(index_cols) == 1):
            index_cols = data_sources[src]["index_cols"]
        else:
            index_cols = data_sources[label_key]["index_cols"]

        df = df.merge(label_df[index_cols].drop_duplicates(), on=index_cols, how="inner")
        print("after merging", len(df.index))
        print(df.sort_values(by=index_cols).head())

    return label_df, feature_dfs


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
data_dfs = load_data_csv(data_path, data_sources)
label_df = data_dfs["YIELD"]
feature_dfs = {
    ft_key : data_dfs[ft_key] for ft_key in data_dfs if ft_key != "YIELD"
}

label_df, feature_dfs = align_data(data_sources, label_df, feature_dfs)