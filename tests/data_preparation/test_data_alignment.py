import os

from config import PATH_DATA_DIR
from config import KEY_LOC, KEY_YEAR, KEY_TARGET
from data_preparation.data_alignment import (
  load_data_csv,
  merge_data,
  set_indices
)


def test_merge_indices():
    data_sources = {
        "YIELD" : {
            "filename" : "YIELD_COUNTY_US.csv",
            "index_cols" : [KEY_LOC, KEY_YEAR],
            "sel_cols" : [KEY_TARGET]
        },
        "SOIL" : {
            "filename" : "SOIL_COUNTY_US.csv",
            "index_cols" : [KEY_LOC],
            "sel_cols" : ["sm_whc"]
        },
        "REMOTE_SENSING" : {
            "filename" : "REMOTE_SENSING_COUNTY_US.csv",
            "index_cols" : [KEY_LOC, KEY_YEAR, "dekad"],
            "sel_cols" : ["fapar"]
        }
    }

    data_path = os.path.join(PATH_DATA_DIR, "data_US", "county_data")
    data_dfs = load_data_csv(data_path, data_sources)
    label_df = data_dfs["YIELD"]

    # keep data for one state (Iowa)
    label_df = label_df[label_df[KEY_LOC].str[:5] == "US-19"]
    feature_dfs = {
        ft_key : data_dfs[ft_key] for ft_key in data_dfs if ft_key != "YIELD"
    }

    label_df, feature_dfs = merge_data(data_sources, label_df, feature_dfs)
    label_df, feature_dfs = set_indices(data_sources, label_df, feature_dfs)

    # sort the indices
    label_df.sort_index(inplace=True)
    for src in feature_dfs:
        feature_dfs[src].sort_index(inplace=True)

    # check label indices are present in all feature data
    for i, row in label_df.iterrows():
        for src in feature_dfs:
            ft_df = feature_dfs[src]
            if (len(ft_df.index.names) == 1):
                assert (i[0] in ft_df.index)
            else:
                assert (i in ft_df.index)
