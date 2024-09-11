import os
import numpy as np
import pandas as pd


from cybench.config import (
    PATH_DATA_DIR,
    DATASETS,
    KEY_LOC,
    KEY_YEAR,
    KEY_TARGET,
    SOIL_PROPERTIES,
    METEO_INDICATORS,
    RS_FPAR,
    RS_NDVI,
    SOIL_MOISTURE_INDICATORS,
    CROP_CALENDAR_DOYS,
    CROP_CALENDAR_DATES,
)

from cybench.datasets.alignment import (
    trim_to_lead_time,
    align_inputs_and_labels,
)


def preprocess_crop_calendar(df, min_year, max_year):
    """Calculate start of season date, end of season date and season_length.

    Args:
        df (pd.DataFrame): crop calendar data
        min_year (int): earliest year in target data
        max_year (int): latest year in target data

    returns:
        the same dataframe with start of season date, end of season date and season_length
        added
    """
    df = df[[KEY_LOC] + CROP_CALENDAR_DOYS]
    df = df.astype({k: int for k in CROP_CALENDAR_DOYS})
    df = pd.concat(
        [df.assign(**{KEY_YEAR: yr}) for yr in range(min_year, max_year + 1)],
        ignore_index=True,
    )
    df["sos_date"] = pd.to_datetime(df[KEY_YEAR] * 1000 + df["sos"], format="%Y%j")
    df["eos_date"] = pd.to_datetime(df[KEY_YEAR] * 1000 + df["eos"], format="%Y%j")

    # Fix sos_date for cases where sos > eos.
    # For example, sos_date is 20011124 and eos_date is 20010615.
    # For 2001, select sos_date for the previous year because season started
    # in the previous year.
    df["sos_date"] = np.where(
        (df["sos"] > df["eos"]),
        df["sos_date"] + pd.offsets.DateOffset(years=-1),
        df["sos_date"],
    )

    df["season_length"] = (df["eos_date"] - df["sos_date"]).dt.days
    assert df[df["season_length"] > 366].empty

    return df


def _load_and_preprocess_time_series_data(
    crop, country_code, ts_input, index_cols, ts_cols, df_crop_cal
):
    """A helper function to load and preprocess time series data.

    Args:
        crop (str): crop name
        country_code (str): 2-letter country code
        ts_input (str): time series input (used to name data file)
        index_cols (list): columns used as index
        ts_cols (list): columns with time series variables
        df_crop_cal (pd.DataFrame): crop calendar data

    Returns:
        the same dataframe after preprocessing
    """
    path_data_cn = os.path.join(PATH_DATA_DIR, crop, country_code)
    df_ts = pd.read_csv(
        os.path.join(path_data_cn, "_".join([ts_input, crop, country_code]) + ".csv"),
        header=0,
    )
    df_ts["date"] = pd.to_datetime(df_ts["date"], format="%Y%m%d")
    df_ts[KEY_YEAR] = df_ts["date"].dt.year
    df_ts = df_ts[index_cols + ts_cols]
    df_ts = trim_to_lead_time(df_ts, df_crop_cal)
    df_ts.set_index(index_cols, inplace=True)

    return df_ts


def load_dfs(crop: str, country_code: str) -> tuple:
    """Load data from CSV files for crop and country.
    Expects CSV files in PATH_DATA_DIR/<crop>/<country_code>/.

    Args:
        crop (str): crop name
        country_code (str): 2-letter country code

    Returns:
        a tuple (target DataFrame, dict of input DataFrames)
    """
    path_data_cn = os.path.join(PATH_DATA_DIR, crop, country_code)

    # targets
    df_y = pd.read_csv(
        os.path.join(path_data_cn, "_".join(["yield", crop, country_code]) + ".csv"),
        header=0,
    )
    df_y = df_y.rename(columns={"harvest_year": KEY_YEAR})
    df_y = df_y[[KEY_LOC, KEY_YEAR, KEY_TARGET]]
    df_y = df_y.dropna(axis=0)
    df_y = df_y[df_y[KEY_TARGET] > 0.0]
    df_y.set_index([KEY_LOC, KEY_YEAR], inplace=True)

    # soil
    df_x_soil = pd.read_csv(
        os.path.join(path_data_cn, "_".join(["soil", crop, country_code]) + ".csv"),
        header=0,
    )
    df_x_soil = df_x_soil[[KEY_LOC] + SOIL_PROPERTIES]
    df_x_soil.set_index([KEY_LOC], inplace=True)
    dfs_x = {"soil": df_x_soil}

    # crop calendar
    df_crop_cal = pd.read_csv(
        os.path.join(
            path_data_cn, "_".join(["crop_calendar", crop, country_code]) + ".csv"
        ),
        header=0,
    )[[KEY_LOC] + CROP_CALENDAR_DOYS]
    index_y_years = set([year for _, year in df_y.index.values])
    df_crop_cal = preprocess_crop_calendar(
        df_crop_cal, min(index_y_years), max(index_y_years)
    )

    # Time series data
    # NOTE: All time series data have to be rotated by crop calendar.
    # Set index to ts_index_cols after rotation.
    ts_index_cols = [KEY_LOC, KEY_YEAR, "date"]
    ts_inputs = {
        "meteo": METEO_INDICATORS,
        "fpar": [RS_FPAR],
        "ndvi": [RS_NDVI],
        "soil_moisture": SOIL_MOISTURE_INDICATORS,
    }

    for x, ts_cols in ts_inputs.items():
        df_ts = _load_and_preprocess_time_series_data(
            crop, country_code, x, ts_index_cols, ts_cols, df_crop_cal
        )
        dfs_x[x] = df_ts

    df_crop_cal.set_index([KEY_LOC, KEY_YEAR], inplace=True)
    df_crop_cal = df_crop_cal[CROP_CALENDAR_DATES]
    dfs_x["crop_calendar"] = df_crop_cal
    df_y, dfs_x = align_inputs_and_labels(df_y, dfs_x)

    return df_y, dfs_x


def load_dfs_crop(crop: str, countries: list = None) -> dict:
    """
    Load data for crop and one or more countries. If `countries` is None,
    data for all countries in CY-Bench is loaded.

    Args:
        crop (str): crop name
        countries (list): list of 2-letter country codes

    Returns:
        a tuple (target DataFrame, dict of input DataFrames)
    """
    assert crop in DATASETS

    if countries is None:
        countries = DATASETS[crop]

    df_y = pd.DataFrame()
    dfs_x = {}
    for cn in countries:
        try:
            df_y_cn, dfs_x_cn = load_dfs(crop, cn)
        except FileNotFoundError:
            continue

        df_y = pd.concat([df_y, df_y_cn], axis=0)
        if len(dfs_x) == 0:
            dfs_x = dfs_x_cn
        else:
            for x, df_x_cn in dfs_x_cn.items():
                dfs_x[x] = pd.concat([dfs_x[x], df_x_cn], axis=0)

    return df_y, dfs_x
