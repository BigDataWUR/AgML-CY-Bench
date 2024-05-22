import os

import pandas as pd

import config


def load_dfs_test_maize_us() -> tuple:
    path_data_us = os.path.join(config.PATH_DATA_DIR, "data_US")

    df_y = pd.read_csv(
        os.path.join(path_data_us, "county_data", "YIELD_COUNTY_US.csv"),
        index_col=["loc_id", "year"],
    )[["yield"]]

    df_x_soil = pd.read_csv(
        os.path.join(path_data_us, "county_data", "SOIL_COUNTY_US.csv"),
        index_col=["loc_id"],
    )[["sm_whc"]]

    df_x_meteo = pd.read_csv(
        os.path.join(path_data_us, "county_data", "METEO_COUNTY_US.csv"),
        index_col=["loc_id", "year", "dekad"],
    )

    df_x_rs = pd.read_csv(
        os.path.join(path_data_us, "county_data", "REMOTE_SENSING_COUNTY_US.csv"),
        index_col=["loc_id", "year", "dekad"],
    )

    dfs_x = (
        df_x_soil,
        df_x_meteo,
        df_x_rs,
    )

    df_y, dfs_x = _align_data(df_y, dfs_x)

    return df_y, dfs_x


def load_dfs_test_maize_fr() -> tuple:
    path_data_fr = os.path.join(config.PATH_DATA_DIR, "data_FR")

    df_y = pd.read_csv(
        os.path.join(path_data_fr, "YIELD_NUTS3_FR.csv"),
        index_col=["loc_id", "year"],
    )[["yield"]]

    df_x_soil = pd.read_csv(
        os.path.join(path_data_fr, "SOIL_NUTS3_FR.csv"),
        index_col=["loc_id"],
    )[["sm_whc"]]

    df_x_meteo = pd.read_csv(
        os.path.join(path_data_fr, "METEO_NUTS3_FR.csv"),
        index_col=["loc_id", "year", "dekad"],
    )

    df_x_rs = pd.read_csv(
        os.path.join(path_data_fr, "REMOTE_SENSING_NUTS3_FR.csv"),
        index_col=["loc_id", "year", "dekad"],
    )

    dfs_x = (
        df_x_soil,
        df_x_meteo,
        df_x_rs,
    )

    df_y, dfs_x = _align_data(df_y, dfs_x)

    return df_y, dfs_x


def load_dfs_test_maize() -> tuple:

    df_y_us, dfs_x_us = load_dfs_test_maize_us()
    df_y_fr, dfs_x_fr = load_dfs_test_maize_fr()

    df_y = pd.concat(
        [
            df_y_us,
            df_y_fr,
        ],
        axis=0,
    )

    dfs_x = tuple(
        pd.concat([df_x_us, df_x_fr], axis=0)
        for df_x_us, df_x_fr in zip(dfs_x_us, dfs_x_fr)
    )

    return df_y, dfs_x


def _align_data(df_y: pd.DataFrame, dfs_x: tuple) -> tuple:
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


def load_dfs_test_softwheat_nl() -> tuple:
    path_data_nl = os.path.join(config.PATH_DATA_DIR, "data_NL")

    df_y = pd.read_csv(
        os.path.join(path_data_nl, "YIELD_NUTS2_NL.csv"),
        index_col=["loc_id", "year"],
    )

    df_y = df_y.loc[df_y["crop_name"] == "Soft wheat"][["yield"]]

    df_x_soil = pd.read_csv(
        os.path.join(path_data_nl, "SOIL_NUTS2_NL.csv"),
        index_col=["loc_id"],
    )[["sm_wp", "sm_fc", "sm_sat", "rooting_depth"]]

    df_x_meteo = pd.read_csv(
        os.path.join(path_data_nl, "METEO_DAILY_NUTS2_NL.csv"),
    )

    df_x_meteo["date"] = pd.to_datetime(df_x_meteo["date"], format="%Y%m%d").dt.date
    df_x_meteo["year"] = df_x_meteo["date"].apply(lambda date: date.year)
    df_x_meteo.set_index(["loc_id", "year", "date"], inplace=True)

    df_x_rs = pd.read_csv(
        os.path.join(path_data_nl, "REMOTE_SENSING_NUTS2_NL.csv"),
    )

    df_x_rs["date"] = pd.to_datetime(df_x_rs["date"], format="%Y%m%d").dt.date
    df_x_rs["year"] = df_x_rs["date"].apply(lambda date: date.year)
    df_x_rs.set_index(["loc_id", "year", "date"], inplace=True)

    dfs_x = (
        df_x_soil,
        df_x_meteo,
        df_x_rs,
    )

    df_y, dfs_x = _align_data(df_y, dfs_x)

    return df_y, dfs_x
