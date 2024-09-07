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
    CROP_CALENDAR_ENTRIES,
)

from cybench.datasets.alignment import (
    trim_to_lead_time,
    align_inputs_and_labels,
)


def _add_year(df: pd.DataFrame) -> pd.DataFrame:
    """Add a year column.

    Args:
        df (pd.DataFrame): time series data

    Returns:
        the same DataFrame with year column added
    """
    df["date"] = df["date"].astype(str)
    df[KEY_YEAR] = df["date"].str[:4]
    df[KEY_YEAR] = df[KEY_YEAR].astype(int)

    return df


def _preprocess_time_series_data(df, index_cols, select_cols, df_crop_cal):
    """Preprocess time series data and align to crop season.

    Args:
        df (pd.DataFrame): time series data
        index_cols (list): index columns
        select_cols (list): time series data columns
        crop_cal_df (pd.DataFrame): crop calendar data

    Returns:
        the same DataFrame preprocessed and aligned to crop season
    """
    df = _add_year(df)
    df = df[index_cols + select_cols]
    df = trim_to_lead_time(df, df_crop_cal)

    return df


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

    # crop calendar
    df_crop_cal = pd.read_csv(
        os.path.join(
            path_data_cn, "_".join(["crop_calendar", crop, country_code]) + ".csv"
        ),
        header=0,
    )[[KEY_LOC] + CROP_CALENDAR_ENTRIES]
    # Calculate season length. Handle seasons crossing calendar year.
    df_crop_cal["season_length"] = np.where(
        (df_crop_cal["eos"] > df_crop_cal["sos"]),
        (df_crop_cal["eos"] - df_crop_cal["sos"]),
        (365 - df_crop_cal["sos"]) + df_crop_cal["eos"],
    )

    # Time series data
    # NOTE: All time series data have to be rotated by crop calendar.
    # Set index to ts_index_cols after rotation.
    ts_index_cols = [KEY_LOC, KEY_YEAR, "date"]
    # meteo
    df_x_meteo = pd.read_csv(
        os.path.join(path_data_cn, "_".join(["meteo", crop, country_code]) + ".csv"),
        header=0,
    )
    df_x_meteo = _preprocess_time_series_data(
        df_x_meteo, ts_index_cols, METEO_INDICATORS, df_crop_cal
    )
    df_x_meteo.set_index(ts_index_cols, inplace=True)

    # fpar
    df_x_fpar = pd.read_csv(
        os.path.join(path_data_cn, "_".join([RS_FPAR, crop, country_code]) + ".csv"),
        header=0,
    )
    df_x_fpar = _preprocess_time_series_data(
        df_x_fpar, ts_index_cols, [RS_FPAR], df_crop_cal
    )
    df_x_fpar.set_index(ts_index_cols, inplace=True)

    # ndvi
    df_x_ndvi = pd.read_csv(
        os.path.join(path_data_cn, "_".join([RS_NDVI, crop, country_code]) + ".csv"),
        header=0,
    )
    df_x_ndvi = _preprocess_time_series_data(
        df_x_ndvi, ts_index_cols, [RS_NDVI], df_crop_cal
    )
    df_x_ndvi.set_index(ts_index_cols, inplace=True)

    # soil moisture
    df_x_soil_moisture = pd.read_csv(
        os.path.join(
            path_data_cn, "_".join(["soil_moisture", crop, country_code]) + ".csv"
        ),
        header=0,
    )
    df_x_soil_moisture = _preprocess_time_series_data(
        df_x_soil_moisture,
        ts_index_cols,
        SOIL_MOISTURE_INDICATORS,
        df_crop_cal,
    )
    df_x_soil_moisture.set_index(ts_index_cols, inplace=True)

    df_crop_cal.set_index([KEY_LOC], inplace=True)

    dfs_x = {
        "soil": df_x_soil,
        "crop_calendar": df_crop_cal,
        "meteo": df_x_meteo,
        RS_FPAR: df_x_fpar,
        RS_NDVI: df_x_ndvi,
        "soil_moisture": df_x_soil_moisture,
    }
    df_y, dfs_x = align_inputs_and_labels(df_y, dfs_x)

    return df_y, dfs_x


def load_dfs_crop(
    crop: str, countries: list = None
) -> dict:
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
