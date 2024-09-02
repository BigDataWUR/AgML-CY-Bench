import os
import pandas as pd


from cybench.config import (
    PATH_DATA_DIR,
    PATH_ALIGNED_DATA_DIR,
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
    SPINUP_DAYS,
)

from cybench.datasets.alignment import (
    align_to_crop_season,
    trim_to_lead_time,
    align_inputs_and_labels,
)


def _add_year(df: pd.DataFrame) -> pd.DataFrame:
    df["date"] = df["date"].astype(str)
    df[KEY_YEAR] = df["date"].str[:4]
    df[KEY_YEAR] = df[KEY_YEAR].astype(int)

    return df


def _preprocess_time_series_data(df, index_cols, select_cols, df_crop_cal):
    df = _add_year(df)
    df = df[index_cols + select_cols]
    df = df.dropna(axis=0)
    df = align_to_crop_season(df, df_crop_cal, SPINUP_DAYS)

    return df


def load_dfs(crop: str, country_code: str) -> tuple:
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

    dfs_x = {
        "soil": df_x_soil,
        "meteo": df_x_meteo,
        RS_FPAR: df_x_fpar,
        RS_NDVI: df_x_ndvi,
        "soil_moisture": df_x_soil_moisture,
    }
    df_y, dfs_x = align_inputs_and_labels(df_y, dfs_x)

    return df_y, dfs_x


def load_aligned_dfs(crop: str, country_code: str) -> tuple:
    path_data_cn = os.path.join(PATH_ALIGNED_DATA_DIR, crop, country_code)
    # targets
    df_y = pd.read_csv(
        os.path.join(path_data_cn, "_".join(["yield", crop, country_code]) + ".csv"),
        header=0,
        index_col=[KEY_LOC, KEY_YEAR],
    )

    # soil
    df_x_soil = pd.read_csv(
        os.path.join(path_data_cn, "_".join(["soil", crop, country_code]) + ".csv"),
        header=0,
        index_col=[KEY_LOC],
    )

    # Time series data
    ts_index_cols = [KEY_LOC, KEY_YEAR, "date"]
    # meteo
    df_x_meteo = pd.read_csv(
        os.path.join(path_data_cn, "_".join(["meteo", crop, country_code]) + ".csv"),
        header=0,
        index_col=ts_index_cols,
    )

    # fpar
    df_x_fpar = pd.read_csv(
        os.path.join(path_data_cn, "_".join([RS_FPAR, crop, country_code]) + ".csv"),
        header=0,
        index_col=ts_index_cols,
    )

    # ndvi
    df_x_ndvi = pd.read_csv(
        os.path.join(path_data_cn, "_".join([RS_NDVI, crop, country_code]) + ".csv"),
        header=0,
        index_col=ts_index_cols,
    )

    # soil moisture
    df_x_soil_moisture = pd.read_csv(
        os.path.join(
            path_data_cn, "_".join(["soil_moisture", crop, country_code]) + ".csv"
        ),
        header=0,
        index_col=ts_index_cols,
    )

    dfs_x = {
        "soil": df_x_soil,
        "meteo": df_x_meteo,
        RS_FPAR: df_x_fpar,
        RS_NDVI: df_x_ndvi,
        "soil_moisture": df_x_soil_moisture,
    }

    return df_y, dfs_x


def load_dfs_crop(
    crop: str, countries: list = None, lead_time: str = FORECAST_LEAD_TIME
) -> dict:
    assert crop in DATASETS

    if countries is None:
        countries = DATASETS[crop]

    df_y = pd.DataFrame()
    dfs_x = {}
    for cn in countries:
        # load aligned data if exists
        try:
            df_y_cn, dfs_x_cn = load_aligned_dfs(crop, cn)
        except FileNotFoundError:
            try:
                df_y_cn, dfs_x_cn = load_dfs(crop, cn)
                # save aligned data
                cn_data_dir = os.path.join(PATH_ALIGNED_DATA_DIR, crop, cn)
                os.makedirs(cn_data_dir, exist_ok=True)
                df_y_cn.to_csv(
                    os.path.join(cn_data_dir, "_".join([KEY_TARGET, crop, cn]) + ".csv"),
                )
                for x, df_x in dfs_x_cn.items():
                    df_x.to_csv(
                        os.path.join(cn_data_dir, "_".join([x, crop, cn]) + ".csv"),
                    )
            except FileNotFoundError:
                continue

        df_y = pd.concat([df_y, df_y_cn], axis=0)
        if len(dfs_x) == 0:
            dfs_x = dfs_x_cn
        else:
            for x, df_x_cn in dfs_x_cn.items():
                dfs_x[x] = pd.concat([dfs_x[x], df_x_cn], axis=0)

    # Trim to lead time and ensure same number of time steps
    # for time series data.
    for x in dfs_x:
        if "date" in dfs_x[x].index.names:
            dfs_x[x] = trim_to_lead_time(dfs_x[x], lead_time, SPINUP_DAYS)

    return df_y, dfs_x
