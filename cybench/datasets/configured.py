import os
import pandas as pd


from cybench.config import (
    PATH_DATA_DIR,
    DATASETS,
    KEY_LOC,
    KEY_YEAR,
    KEY_TARGET,
    KEY_CROP_SEASON,
    MIN_INPUT_YEAR,
    MAX_INPUT_YEAR,
    SOIL_PROPERTIES,
    TIME_SERIES_INPUTS,
)

from cybench.datasets.alignment import (
    compute_crop_season_window,
    align_to_crop_season_window,
    align_inputs_and_labels,
)

from cybench.util.data import get_trend_features


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
        the same DataFrame after preprocessing and aligning to crop season
    """
    path_data_cn = os.path.join(PATH_DATA_DIR, crop, country_code)
    df_ts = pd.read_csv(
        os.path.join(path_data_cn, "_".join([ts_input, crop, country_code]) + ".csv"),
        header=0,
    )
    df_ts["date"] = pd.to_datetime(df_ts["date"], format="%Y%m%d")
    df_ts[KEY_YEAR] = df_ts["date"].dt.year
    df_ts = df_ts[index_cols + ts_cols]
    df_ts = align_to_crop_season_window(df_ts, df_crop_cal)
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
    # check empty targets
    if df_y.empty:
        return df_y, {}

    # yield trend
    df_y_trend = get_trend_features(df_y, 5)
    df_y_trend.set_index([KEY_LOC, KEY_YEAR], inplace=True)

    # set index of df_y
    df_y.set_index([KEY_LOC, KEY_YEAR], inplace=True)

    # soil
    df_x_soil = pd.read_csv(
        os.path.join(path_data_cn, "_".join(["soil", crop, country_code]) + ".csv"),
        header=0,
    )
    df_x_soil = df_x_soil[[KEY_LOC] + SOIL_PROPERTIES]
    df_x_soil.set_index([KEY_LOC], inplace=True)
    dfs_x = {
        "soil": df_x_soil,
        "yield_trend" : df_y_trend
    }

    # crop calendar
    df_crop_cal = pd.read_csv(
        os.path.join(
            path_data_cn, "_".join(["crop_calendar", crop, country_code]) + ".csv"
        ),
        header=0,
    )
    df_crop_cal = compute_crop_season_window(
        df_crop_cal, MIN_INPUT_YEAR, MAX_INPUT_YEAR
    )

    # Time series data
    # NOTE: All time series data have to be aligned to crop season.
    # Set index to ts_index_cols after alignment.
    ts_index_cols = [KEY_LOC, KEY_YEAR, "date"]
    for x, ts_cols in TIME_SERIES_INPUTS.items():
        df_ts = _load_and_preprocess_time_series_data(
            crop, country_code, x, ts_index_cols, ts_cols, df_crop_cal
        )
        dfs_x[x] = df_ts

    # crop season based on SPINUP_DAYS and lead time
    dfs_x[KEY_CROP_SEASON] = df_crop_cal.set_index([KEY_LOC, KEY_YEAR])
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
