import pandas as pd


import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta
from datetime import date

from config import KEY_LOC, KEY_YEAR


def doy_to_date(doy, year):
    # Based on
    # https://www.geeksforgeeks.org/python-convert-day-number-to-date-in-particular-year/

    # Pad on the left with 0's.
    # See https://docs.python.org/3/library/stdtypes.html. Look for str.rjust().
    if isinstance(doy, int):
        doy = str(doy)
    if isinstance(year, int):
        year = str(year)

    doy.rjust(3 + len(doy), "0")
    date_str = datetime.strptime(year + "-" + doy, "%Y-%j").strftime("%Y%m%d")

    return date_str


def _update_harvest_year(harvest_date, year, new_year):
    if year != new_year:
        return harvest_date.replace(str(year), str(new_year))

    return harvest_date


def _update_date(harvest_date, diff_with_harvest):
    # NOTE add the difference between YYYY1231 and harvest_date
    # Then add diff_with_harvest
    end_of_year = date(harvest_date.year, 12, 31)
    end_of_year_delta = (end_of_year - harvest_date).days
    days_delta = end_of_year_delta + diff_with_harvest
    return harvest_date + timedelta(days=days_delta)


def _merge_with_crop_calendar(df, crop_cal_df):
    df = df.merge(crop_cal_df, on=[KEY_LOC])
    df["season_start"] = df.apply(
        lambda r: doy_to_date(r["eos"], r[KEY_YEAR]), axis=1
    )
    df["season_end"] = df.apply(
        lambda r: doy_to_date(r["eos"], r[KEY_YEAR]), axis=1
    )

    return df


def rotate_data_by_crop_calendar(df, crop_cal_df, spinup=90,
                                 ts_index_cols=[KEY_LOC, KEY_YEAR, "date"]):
    data_cols = [c for c in df.columns if c not in ts_index_cols]
    crop_cal_cols = [KEY_LOC, "sos", "eos"]
    crop_cal_df = crop_cal_df.astype({"sos": int, "eos": int})
    df = _merge_with_crop_calendar(df, crop_cal_df[crop_cal_cols])
    df = df.astype({"date": "str"})

    # The next new year starts right after this year's harvest.
    df["new_year"] = np.where(
        df["date"] > df["season_end"], df["year"] + 1, df["year"]
    )
    df = df.astype({"season_end": str})
    df["harvest_date"] = df.apply(
        lambda r: _update_harvest_year(r["season_end"], r["year"], r["new_year"]),
        axis=1,
    )
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df["harvest_date"] = pd.to_datetime(df["harvest_date"], format="%Y%m%d")
    df["season_start"] = pd.to_datetime(df["season_start"], format="%Y%m%d")
    df["harvest_diff"] = (df["date"] - df["harvest_date"]).dt.days
    df = df.rename(
        columns={
            "date": "original_date",
            KEY_YEAR: "original_year",
            "new_year": KEY_YEAR,
        }
    )
    # NOTE pd.to_datetime() creates a Timestamp object
    df["date"] = df.apply(
        lambda r: _update_date(r["harvest_date"].date(), r["harvest_diff"]), axis=1
    )
    df["spinup_date"] = df["season_start"] - pd.Timedelta(days=spinup)
    df = df[
        (df["original_date"] >= df["spinup_date"])
        & (df["original_date"] <= df["harvest_date"])
    ]

    df = df[ts_index_cols + data_cols]

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

