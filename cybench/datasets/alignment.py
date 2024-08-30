import pandas as pd


import pandas as pd
import numpy as np
from datetime import timedelta
from datetime import date

from cybench.config import KEY_LOC, KEY_YEAR


def _add_cutoff_days(df, lead_time):
    # For lead_time, see FORECAST_LEAD_TIME in config.py.
    if "day" in lead_time:
        df["cutoff_days"] = int(lead_time.split("-")[0])
    else:
        assert "season" in lead_time
        if lead_time == "middle-of-season":
            df["cutoff_days"] = df["season_length"] // 2
        elif lead_time == "quarter-of-season":
            df["cutoff_days"] = df["season_length"] // 4
        else:
            raise Exception(f'Unrecognized lead time "{lead_time}"')

    return df


def trim_to_lead_time(df, crop_cal_df, lead_time, spinup_days=90):
    select_cols = list(df.columns)

    # Merge with crop calendar
    crop_cal_cols = [KEY_LOC, "sos", "eos"]
    crop_cal_df = crop_cal_df.astype({"sos": int, "eos": int})

    df = df.merge(crop_cal_df[crop_cal_cols], on=[KEY_LOC])
    df = df.astype({KEY_LOC: "category"})

    df["sos_date"] = pd.to_datetime(df[KEY_YEAR] * 1000 + df["sos"], format="%Y%j")
    df["eos_date"] = pd.to_datetime(df[KEY_YEAR] * 1000 + df["eos"], format="%Y%j")

    df["new_year"] = np.where(df["date"] > df["eos_date"], df["year"] + 1, df["year"])
    # Fix sos_date for seasons crossing calendar year
    df["sos_date"] = np.where(
        (df["date"] <= df["eos_date"]) & (df["sos"] > df["eos"]),
        # select eos_date for the next year
        df["sos_date"] + pd.offsets.DateOffset(years=-1),
        df["sos_date"],
    )

    # Fix eos_date for seasons crossing calendar year
    df["eos_date"] = np.where(
        (df["date"] > df["eos_date"]),
        # select eos_date for the next year
        df["eos_date"] + pd.offsets.DateOffset(years=1),
        df["eos_date"],
    )

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
    df["season_length"] = np.where(
        (df["eos"] > df["sos"]),
        (df["eos"] - df["sos"]),
        (365 - df["sos"]) + df["eos"],
    )

    # Keep data for spinup days before the start of season.
    df["ts_length"] = np.where(
        df["season_length"] + spinup_days <= 365,
        df["season_length"] + spinup_days,
        365,
    )

    # drop years with not enough data for a season
    # NOTE: It's necessary to make sure years with incomplete data
    # don't influence ensuring same number of time steps. See below.
    df["min_date"] = df.groupby([KEY_LOC, KEY_YEAR], observed=True)["date"].transform(
        "min"
    )
    df["max_date"] = df.groupby([KEY_LOC, KEY_YEAR], observed=True)["date"].transform(
        "max"
    )
    df = df[(df["max_date"] - df["min_date"]).dt.days >= df["ts_length"]]

    # Determine cutoff days based on lead time.
    df = _add_cutoff_days(df, lead_time)
    df["cutoff_date"] = df["end_of_year"] - pd.to_timedelta(df["cutoff_days"], unit="d")

    df = df[df["date"] <= df["cutoff_date"]]

    # Keep the same number of time steps for all locations and years.
    # It's necessary because models will stack time series data in a batch.
    # NOTE: We don't want more than (ts_length - cutoff_days) days of data.
    #   We could do avg of ts_length, but max is safer.
    #   It doesn't hurt to have more data in the front.
    #   Using less may hurt performance.
    # More NOTEs:
    # 1. We take min of date by (loc, year) so that all data points have
    #   num_time_steps.
    # 2. Then we look at max of (ts_length - cutoff_days).
    #    This is the maximum number of time steps after accounting for
    #    spinup_days (ts_length) and lead time (cutoff_days).
    # We take the min of 1 and 2 to meet both criteria.
    # num_time_steps = df.groupby([KEY_LOC, KEY_YEAR])["date"].count().min()

    # Perform groupby and count
    num_time_steps = (
        df.groupby([KEY_LOC, KEY_YEAR], observed=True)["date"].count().min()
    )

    num_time_steps = min(num_time_steps, (df["ts_length"] - df["cutoff_days"]).max())
    # sort by date to make sure tail works correctly
    df = df.sort_values(by=[KEY_LOC, KEY_YEAR, "date"])

    df = (
        df.groupby([KEY_LOC, KEY_YEAR], observed=True)
        .tail(num_time_steps)
        .reset_index()
    )
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
