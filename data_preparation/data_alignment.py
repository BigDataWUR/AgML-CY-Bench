import os
import pandas as pd

from config import PATH_DATA_DIR


def load_data_csv(data_path, data_sources):
    """Load data from csv files to pandas.

    Args:
      data_path: path to directory containing csv files.

      data_sources: a dict containing entries with "filename"

    Returns:
      A dictionary of pd.DataFrame using the same keys as in `data_sources`
    """
    data_dfs = {}
    for src in data_sources:
        assert ("filename" in data_sources[src])
        data_csv = os.path.join(data_path, data_sources[src]["filename"])
        df = pd.read_csv(data_csv, header=0)
        data_dfs[src] = df

    return data_dfs

def merge_data(data_sources, label_df, feature_dfs, label_key="YIELD"):
    """Merge label and feature data to get the same location and years.

    Args:
      data_sources: a dict containing entries with "index_cols"

      label_df: a pd.DataFrame containing label data

      feature_dfs: a dict of pd.DataFrame using the same keys as in `data_sources` except `label_key`

      label_key: key for label in `data_sources`

    Returns:
      A tuple of pd.DataFrame (label data) and a dict of pd.DataFrame (feature data) after merging
    """
    # update label data to contain shared locations and years
    for src in feature_dfs:
        df = feature_dfs[src]
        assert ("index_cols" in data_sources[src])
        index_cols = data_sources[src]["index_cols"]
        # for static data, index_cols contains location column
        if (len(index_cols) > 1):
            # for other data, use location and year column
            index_cols = data_sources[label_key]["index_cols"]

        # first step to update label data based on features
        label_df = label_df.merge(df[index_cols].drop_duplicates(),
                                  on=index_cols, how="inner")

    # update feature data
    for src in feature_dfs:
        df = feature_dfs[src]
        index_cols = data_sources[src]["index_cols"]
        # for static data, index_cols contains location column
        if (len(index_cols) > 1):
            # for other data, use location and year column
            index_cols = data_sources[label_key]["index_cols"]

        feature_dfs[src] = df.merge(label_df[index_cols].drop_duplicates(),
                                    on=index_cols, how="inner")

    return label_df, feature_dfs

def set_indices(data_sources, label_df, feature_dfs, label_key="YIELD"):
    """Set indices of label and feature data.

    Args:
      data_sources: a dict containing entries with "index_cols"

      label_df: a pd.DataFrame containing label data

      feature_dfs: a dict of pd.DataFrame using the same keys as in `data_sources` except `label_key`

      label_key: key for label in `data_sources`

    Returns:
      A tuple of pd.DataFrame (label data) and a dict of pd.DataFrame (feature data) after setting indices
    """
    label_df = label_df.set_index(data_sources[label_key]["index_cols"])
    for src in feature_dfs:
        index_cols = data_sources[src]["index_cols"]
        feature_dfs[src] = feature_dfs[src].set_index(index_cols)
    
    return label_df, feature_dfs
