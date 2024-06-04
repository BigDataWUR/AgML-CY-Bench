import pandas as pd


import pandas as pd
import numpy as np
from datetime import timedelta
from datetime import date

from cybench.config import KEY_LOC, KEY_YEAR


def _add_cutoff_days(df, lead_time):
    if "days" in lead_time:
        df["cutoff_days"] = int(lead_time.split("-")[0])
    else:
        assert "season" in lead_time
        if lead_time == "mid-season":
            df["cutoff_days"] = df["season_length"] // 2
        elif lead_time == "quarter-of-season":
            df["cutoff_days"] = df["season_length"] // 4
        else:
            raise Exception(f'Unrecognized lead time "{lead_time}"')

    return df


def trim_to_lead_time(df, index_cols, crop_cal_df, lead_time, spinup_days=60):
    select_cols = list(df.columns)

    # Merge with crop calendar
    crop_cal_cols = [KEY_LOC, "sos", "eos"]
    crop_cal_df = crop_cal_df.astype({"sos": int, "eos": int})
    df = df.merge(crop_cal_df[crop_cal_cols], on=[KEY_LOC])
    df["sos_date"] = pd.to_datetime(df[KEY_YEAR] * 1000 + df["sos"], format="%Y%j")
    df["eos_date"] = pd.to_datetime(df[KEY_YEAR] * 1000 + df["eos"], format="%Y%j")

    # The next new year starts right after this year's harvest.
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df["new_year"] = np.where(df["date"] > df["eos_date"], df["year"] + 1, df["year"])
    df["eos_date"] = np.where(
        df["year"] != df["new_year"],
        df["new_year"].astype(str) + df["eos_date"].dt.strftime("-%m-%d"),
        df["eos_date"].astype(str),
    )
    df["eos_date"] = pd.to_datetime(df["eos_date"], format="%Y-%m-%d")

    # Compute difference with eos
    df["eos_diff"] = (df["date"] - df["eos_date"]).dt.days
    df = df.rename(
        columns={
            "date": "original_date",
            KEY_YEAR: "original_year",
            "new_year": KEY_YEAR,
        }
    )

    # update date
    # 1. Add delta to the end of the year to align eos with Dec 31.
    # 2. Add delta with eos
    df["end_of_year"] = pd.to_datetime(
        df[KEY_YEAR].astype(str) + "1231", format="%Y%m%d"
    )
    df["date"] = df["eos_date"] + pd.to_timedelta(
        (df["end_of_year"] - df["eos_date"]).dt.days + df["eos_diff"], unit="d"
    )

    df = df[
        (df["original_date"] >= (df["sos_date"] - pd.Timedelta(days=spinup_days)))
        & (df["original_date"] <= df["eos_date"])
    ]

    # Remove years that don't include the complete season
    df["season_length"] = np.where(
        df["sos"] < df["eos"], (df["eos"] - df["sos"]), (365 - df["sos"] + df["eos"])
    )
    df["min_date"] = df.groupby([KEY_LOC, KEY_YEAR])["date"].transform("min")
    df["max_date"] = df.groupby([KEY_LOC, KEY_YEAR])["date"].transform("max")
    df = df[(df["max_date"] - df["min_date"]).dt.days >= df["season_length"]]
    df = df.sort_values(by=index_cols)

    # Determine cutoff days based on lead time.
    df = _add_cutoff_days(df, lead_time)
    df["cutoff_date"] = df["end_of_year"] - pd.to_timedelta(df["cutoff_days"], unit="d")
    df = df[df["date"] < df["cutoff_date"]]

    # Keep the same number of time steps for all locations and years
    num_time_steps = df.groupby([KEY_LOC, KEY_YEAR])["date"].count().min()
    df = df.groupby([KEY_LOC, KEY_YEAR]).tail(num_time_steps).reset_index()

    # NOTE: pandas adds "-" to date
    df["date"] = df["date"].astype(str)
    df["date"] = df["date"].str.replace("-", "")
    df = df[select_cols]

    return df


def align_data(df_y: pd.DataFrame, dfs_x: tuple) -> tuple:
    # Data Alignment
    # - Filter the label data based on presence within all feature data sets
    # - Filter feature data based on label data

    # Filter label data

    index_y_selection = set(df_y.index.values)
    for df_x in dfs_x:
        if len(df_x.index.names) == 1:
            index_y_selection = {
                (loc_id, year)
                for loc_id, year in index_y_selection
                if loc_id in df_x.index.values
            }

        if len(df_x.index.names) == 2:
            index_y_selection = index_y_selection.intersection(set(df_x.index.values))

        if len(df_x.index.names) == 3:
            index_y_selection = index_y_selection.intersection(
                set([(loc_id, year) for loc_id, year, _ in df_x.index.values])
            )

    # Filter the labels
    df_y = df_y.loc[list(index_y_selection)]

    # Filter feature data
    # TODO
    index_y_location_selection = set([loc_id for loc_id, _ in index_y_selection])

    return df_y, dfs_x
