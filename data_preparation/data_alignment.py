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

def merge_data(data_sources, label_df, feature_dfs,
               update_labels=False, label_key="YIELD"):
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
            label_df = label_df.merge(df[index_cols].drop_duplicates(),
                                      on=index_cols, how="inner")
        # second step to update feature data based on labels
        else:
            df = df.merge(label_df[index_cols].drop_duplicates(),
                          on=index_cols, how="inner")
            feature_dfs[src] = df

    return label_df, feature_dfs

def merge_indices(data_sources, label_df, feature_dfs, label_key="YIELD"):
    # align label data
    label_df, feature_dfs = merge_data(data_sources, label_df, feature_dfs,
                                       update_labels=True, label_key=label_key)
    # align feature data
    label_df, feature_dfs = merge_data(data_sources, label_df, feature_dfs,
                                       label_key=label_key)

    return label_df, feature_dfs

def validate_times_series_data(df, group_cols, time_step_col, max_time_steps):
    df = df.groupby(group_cols).agg(NUM_TIME_STEPS=(time_step_col, "count")).reset_index()
    df = df[df["NUM_TIME_STEPS"] == max_time_steps]
    # print(len(df[df["NUM_TIME_STEPS"] == max_time_steps].index))

    return df

data_sources = {
    "YIELD" : {
        "filename" : "YIELD_COUNTY_US.csv",
        "index_cols" : ["loc_id", "year"],
        "sel_cols" : ["yield"]
    },
    "SOIL" : {
        "filename" : "SOIL_COUNTY_US.csv",
        "index_cols" : ["loc_id"],
        "sel_cols" : ["sm_whc"]
    },
    "REMOTE_SENSING" : {
        "filename" : "REMOTE_SENSING_COUNTY_US.csv",
        "index_cols" : ["loc_id", "year", "dekad"],
        "sel_cols" : ["fapar"]
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

label_df, feature_dfs = merge_indices(data_sources, label_df, feature_dfs)

before_and_after["YIELD"].append(len(label_df.index))
for src in feature_dfs:
    before_and_after[src].append(len(feature_dfs[src].index))

for src in before_and_after:
    print(src, "before:", before_and_after[src][0], "after:", before_and_after[src][1])


def get_season_start(planting_doy, maturity_doy, max_days_before_planting=30):
    season_length = maturity_doy - planting_doy
    if (season_length > 360):
        return planting_doy
    elif (season_length > 330):
        return planting_doy - (360 - season_length)
    else:
        return planting_doy - max_days_before_planting

def align_to_crop_season(df, crop_cal_df, index_cols,
                         max_days_before_planting=30):
    """
    We have planting doy and maturity doy in crop calendar.
    If the crop season is shorter than 360 days, we can keep some days before planting.
    Put a limit of 30 days max before planting.
    """
    assert (len(index_cols) == 3)
    df = df.merge(crop_cal_df, on=["loc_id"])
    print(len(df.index))
    df["season_start"] = df.apply((lambda r: get_season_start(r["planting_doy"],
                                                              r["maturity_doy"],
                                                              max_days_before_planting)), axis=1)
    df["season_end"] = df["maturity_doy"]
    df = df.drop(columns=[ c for c in list(crop_cal_df.columns) if c != "loc_id"])
    # TODO: we need a more exact conversion from dekad to doy
    if ("dekad" in df.columns):
        df["doy"] = df["dekad"] * 10

    # rotate data
    return df

data_path = os.path.join("data", "data_US")
crop_cal_csv = os.path.join(data_path, "CROP_CALENDAR_COUNTY_US.csv")
crop_cal_df = pd.read_csv(crop_cal_csv, header=0)
print(crop_cal_df.head())

# "US-01-001" "AL_AUTAUGA"

for src in data_sources:
    index_cols = data_sources[src]["index_cols"]
    if (len(index_cols) == 3):
        src_csv = os.path.join(data_path, "county_data", src + "_COUNTY_US.csv")
        src_df = pd.read_csv(src_csv, header=0)
        src_df = align_to_crop_season(src_df, crop_cal_df,
                                    index_cols)
        print(src_df.head())