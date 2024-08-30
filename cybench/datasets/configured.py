import os
import pandas as pd
from datetime import date, timedelta


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
    FORECAST_LEAD_TIME,
)

from cybench.datasets.alignment import align_data, trim_to_lead_time


def _add_year(df: pd.DataFrame) -> pd.DataFrame:
    assert pd.api.types.is_datetime64_any_dtype(
        df["date"]
    ), f"Column 'date' must be of datetime type."
    df[KEY_YEAR] = df["date"].dt.year
    return df


def _preprocess_time_series_data(df, index_cols, select_cols, df_crop_cal, lead_time):
    df = _add_year(df)
    df = df[index_cols + select_cols]
    df = df.dropna(axis=0)
    df = trim_to_lead_time(df, df_crop_cal, lead_time)

    return df


def get_dtype_mappings():
    return {KEY_LOC: "category", "crop_name": "category"}


def optimize_datatypes(
    df: pd.DataFrame, column_mappings: dict = get_dtype_mappings()
) -> pd.DataFrame:
    valid_mappings = {
        col: dtype for col, dtype in column_mappings.items() if col in df.columns
    }
    df = df.astype(valid_mappings)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    return df


def load_dfs(
    crop: str, country_code: str, lead_time: str = FORECAST_LEAD_TIME
) -> tuple:
    path_data_cn = os.path.join(PATH_DATA_DIR, crop, country_code)

    # targets
    df_y = pd.read_csv(
        os.path.join(path_data_cn, "_".join(["yield", crop, country_code]) + ".csv"),
        header=0,
    )

    df_y = df_y.rename(columns={"harvest_year": KEY_YEAR})
    df_y[KEY_YEAR] = pd.to_datetime(df_y[KEY_YEAR], format="%Y").dt.year
    df_y = optimize_datatypes(df_y)
    df_y = df_y[[KEY_LOC, KEY_YEAR, KEY_TARGET]]
    df_y = df_y.dropna(axis=0)
    df_y = df_y[df_y[KEY_TARGET] > 0.0]

    # soil
    df_x_soil = pd.read_csv(
        os.path.join(path_data_cn, "_".join(["soil", crop, country_code]) + ".csv"),
        header=0,
    )

    df_x_soil = optimize_datatypes(df_x_soil)
    df_x_soil = df_x_soil[[KEY_LOC] + SOIL_PROPERTIES]
    # crop calendar
    df_crop_cal = pd.read_csv(
        os.path.join(
            path_data_cn, "_".join(["crop_calendar", crop, country_code]) + ".csv"
        ),
        header=0,
    )[[KEY_LOC] + CROP_CALENDAR_ENTRIES]
    df_crop_cal = optimize_datatypes(df_crop_cal)

    # Time series data
    # NOTE: All time series data have to be rotated by crop calendar.
    # Set index to ts_index_cols after rotation.

    ts_index_cols = [KEY_LOC, KEY_YEAR, "date"]
    # meteo
    df_x_meteo = pd.read_csv(
        os.path.join(path_data_cn, "_".join(["meteo", crop, country_code]) + ".csv"),
        header=0,
    )
    df_x_meteo = optimize_datatypes(df_x_meteo)
    df_x_meteo = _preprocess_time_series_data(
        df_x_meteo, ts_index_cols, METEO_INDICATORS, df_crop_cal, lead_time
    )
    df_x_meteo = df_x_meteo.set_index(ts_index_cols)

    # fpar
    df_x_fpar = pd.read_csv(
        os.path.join(path_data_cn, "_".join([RS_FPAR, crop, country_code]) + ".csv"),
        header=0,
    )
    df_x_fpar = optimize_datatypes(df_x_fpar)
    df_x_fpar = _preprocess_time_series_data(
        df_x_fpar, ts_index_cols, [RS_FPAR], df_crop_cal, lead_time
    )
    df_x_fpar = df_x_fpar.set_index(ts_index_cols)

    # ndvi
    df_x_ndvi = pd.read_csv(
        os.path.join(path_data_cn, "_".join([RS_NDVI, crop, country_code]) + ".csv"),
        header=0,
    )
    df_x_ndvi = optimize_datatypes(df_x_ndvi)

    df_x_ndvi = _preprocess_time_series_data(
        df_x_ndvi, ts_index_cols, [RS_NDVI], df_crop_cal, lead_time
    )

    df_x_ndvi = df_x_ndvi.set_index(ts_index_cols)

    # soil moisture
    df_x_soil_moisture = pd.read_csv(
        os.path.join(
            path_data_cn, "_".join(["soil_moisture", crop, country_code]) + ".csv"
        ),
        header=0,
    )
    df_x_soil_moisture = optimize_datatypes(df_x_soil_moisture)

    df_x_soil_moisture = _preprocess_time_series_data(
        df_x_soil_moisture,
        ts_index_cols,
        SOIL_MOISTURE_INDICATORS,
        df_crop_cal,
        lead_time,
    )
    df_x_soil_moisture = df_x_soil_moisture.set_index(ts_index_cols)

    df_y = df_y.set_index([KEY_LOC, KEY_YEAR])
    df_x_soil = df_x_soil.set_index([KEY_LOC])

    dfs_x = (df_x_soil, df_x_meteo, df_x_fpar, df_x_ndvi, df_x_soil_moisture)
    df_y, dfs_x = align_data(df_y, dfs_x)

    return df_y, dfs_x


def load_dfs_crop(crop: str, countries: list = None) -> tuple:
    assert crop in DATASETS

    if countries is None:
        countries = DATASETS[crop]

    df_y = pd.DataFrame()
    dfs_x = tuple()
    for cn in countries:
        if not os.path.exists(os.path.join(PATH_DATA_DIR, crop, cn)):
            continue

        df_y_cn, dfs_x_cn = load_dfs(crop, cn)
        df_y = pd.concat([df_y, df_y_cn], axis=0)
        if len(dfs_x) == 0:
            dfs_x = dfs_x_cn
        else:
            dfs_x = tuple(
                pd.concat([df_x, df_x_cn], axis=0)
                for df_x, df_x_cn in zip(dfs_x, dfs_x_cn)
            )

    new_dfs_x = tuple()
    # keep the same number of time steps for time series data
    # NOTE: At this point, each df_x contains data for all selected countries.
    for df_x in dfs_x:
        # If index is [KEY_LOC, KEY_YEAR, "date"]
        if "date" in df_x.index.names:
            index_names = df_x.index.names
            column_names = list(df_x.columns)
            df_x.reset_index(inplace=True)
            min_time_steps = (
                df_x.groupby([KEY_LOC, KEY_YEAR], observed=True)["date"].count().min()
            )
            df_x = df_x.sort_values(by=[KEY_LOC, KEY_YEAR, "date"])
            df_x = (
                df_x.groupby([KEY_LOC, KEY_YEAR], observed=True)
                .tail(min_time_steps)
                .reset_index()
            )
            df_x.set_index(index_names, inplace=True)
            df_x = df_x[column_names]
        new_dfs_x += (df_x,)

    return df_y, new_dfs_x
