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

def merge_data(data_sources, label_df, feature_dfs, update_labels=False, label_key="YIELD"):
    for src in feature_dfs:
        df = feature_dfs[src]
        index_cols = data_sources[src]["index_cols"]

        # static data
        if (len(index_cols) == 1):
            # use location column
            index_cols = data_sources[src]["index_cols"]
        else:
            # use location and year column
            index_cols = data_sources[label_key]["index_cols"]

        # first step to update label data based on features
        if (update_labels):
            label_df = label_df.merge(df[index_cols].drop_duplicates(), on=index_cols, how="inner")
        # update feature data based on labels
        else:
            df = df.merge(label_df[index_cols].drop_duplicates(), on=index_cols, how="inner")
            feature_dfs[src] = df

    return label_df, feature_dfs

def align_data(data_sources, label_df, feature_dfs, label_key="YIELD"):
    # align label data
    label_df, feature_dfs = merge_data(data_sources, label_df, feature_dfs, update_labels=True)
    # align feature data
    label_df, feature_dfs = merge_data(data_sources, label_df, feature_dfs)

    return label_df, feature_dfs

# def validate_times_series_data(df, group_cols, time_step_col, max_time_steps):
#     df = df.groupby(group_cols).agg(NUM_TIME_STEPS=(time_step_col, "count")).reset_index()
    # df = df[df["NUM_TIME_STEPS"] == max_time_steps]
    # print(len(df[df["NUM_TIME_STEPS"] == max_time_steps].index))

    # return df

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

before_and_after = {}
for src in data_dfs:
    before_and_after[src] = [len(data_dfs[src].index)]

label_df, feature_dfs = align_data(data_sources, label_df, feature_dfs)

before_and_after["YIELD"].append(len(label_df.index))
for src in feature_dfs:
    before_and_after[src].append(len(feature_dfs[src].index))

for src in before_and_after:
    print(src, "before:", before_and_after[src][0], "after:", before_and_after[src][1])

# validate_times_series_data(rs_df, group_cols=["COUNTY_ID", "FYEAR"],
#                            time_step_col="DEKAD", max_time_steps=36)
