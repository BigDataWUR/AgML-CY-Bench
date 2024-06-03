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


def get_cutoff_days(crop_cal_df, lead_time):
    if ("days" in lead_time):
        crop_cal_df["cutoff_days"] = int(lead_time.split("-")[0])
    else:
        assert ("season" in lead_time)
        if (lead_time == "mid-season"):
            crop_cal_df["cutoff_days"] = np.where(crop_cal_df["sos"] < crop_cal_df["eos"],
                                                  (crop_cal_df["eos"] - crop_cal_df["sos"])//2,
                                                  (365 - crop_cal_df["sos"] + crop_cal_df["eos"])//2)             
        elif (lead_time == "quarter-of-season"):
            crop_cal_df["cutoff_days"] = np.where(crop_cal_df["sos"] < crop_cal_df["eos"],
                                                  (crop_cal_df["eos"] - crop_cal_df["sos"])//4,
                                                  (365 - crop_cal_df["sos"] + crop_cal_df["eos"])//4) 
        else:
            raise Exception(f'Unrecognized lead time "{lead_time}"')

    return crop_cal_df[[KEY_LOC, "cutoff_days"]]

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
    df["season_start"] = df.apply(lambda r: doy_to_date(r["sos"], r[KEY_YEAR]), axis=1)
    df["season_end"] = df.apply(lambda r: doy_to_date(r["eos"], r[KEY_YEAR]), axis=1)

    return df


def rotate_data_by_crop_calendar(
    df, crop_cal_df, spinup=60, ts_index_cols=[KEY_LOC, KEY_YEAR, "date"]
):
    data_cols = [c for c in df.columns if c not in ts_index_cols]
    crop_cal_cols = [KEY_LOC, "sos", "eos"]
    crop_cal_df = crop_cal_df.astype({"sos": int, "eos": int})
    df = _merge_with_crop_calendar(df, crop_cal_df[crop_cal_cols])
    df = df.astype({"date": "str"})

    # The next new year starts right after this year's harvest.
    df["new_year"] = np.where(df["date"] > df["season_end"], df["year"] + 1, df["year"])
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

def trim_to_lead_time(df, crop_cal_df, lead_time):
    # Includes number of days
    df_cutoff_dates = get_cutoff_days(crop_cal_df, lead_time)

    df = df.merge(df_cutoff_dates, on=[KEY_LOC])
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df["end_of_year"] = df.apply(lambda r: date(r[KEY_YEAR], 12, 31), axis=1)
    df["cutoff_date"] = df.apply(lambda r: r["end_of_year"] -
                                 timedelta(days=r["cutoff_days"]), axis=1)
    df = df[df["date"] < df["cutoff_date"]]

    # keep the minimum number of time steps
    num_time_steps = df.groupby([KEY_LOC, KEY_YEAR])["date"].count().min()
    df = df.groupby([KEY_LOC, KEY_YEAR]).tail(num_time_steps).reset_index()
    # NOTE: pandas adds "-" to date
    df["date"] = df["date"].astype(str)
    df["date"] = df["date"].str.replace("-", "")
    df = df.drop(columns=["cutoff_days", "cutoff_date", "end_of_year", "index"])

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
