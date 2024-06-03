import os
import pandas as pd

from config import (
    PATH_DATA_DIR,
    KEY_LOC,
    KEY_YEAR,
    KEY_TARGET,
    SOIL_PROPERTIES,
    METEO_INDICATORS,
    RS_FPAR
)


def load_dfs(crop:str, country_code:str) -> tuple:
    path_data_cn = os.path.join(PATH_DATA_DIR, crop, country_code)

    df_y = pd.read_csv(
        os.path.join(path_data_cn, "_".join(["yield", crop, country_code]) + ".csv"),
        header=0,
    )
    df_y = df_y.rename(columns={"harvest_year" : KEY_YEAR})
    df_y = df_y.set_index([KEY_LOC, KEY_YEAR])[[KEY_TARGET]]

    df_x_soil = pd.read_csv(
        os.path.join(path_data_cn, "_".join(["soil", crop, country_code]) + ".csv"),
        index_col=[KEY_LOC],
    )[SOIL_PROPERTIES]

    df_x_meteo = pd.read_csv(
        os.path.join(path_data_cn, "_".join(["meteo", crop, country_code]) + ".csv"),
        header=0,
    )
    df_x_meteo["date"] = df_x_meteo["date"].astype(str)
    df_x_meteo[KEY_YEAR] = df_x_meteo["date"].str[:4]
    df_x_meteo[KEY_YEAR] = df_x_meteo[KEY_YEAR].astype(int)
    df_x_meteo = df_x_meteo.set_index([KEY_LOC, KEY_YEAR, "date"])[METEO_INDICATORS]

    df_x_rs = pd.read_csv(
        os.path.join(path_data_cn, "_".join([RS_FPAR, crop, country_code]) + ".csv"),
        header=0,
    )
    df_x_rs["date"] = df_x_rs["date"].astype(str)
    df_x_rs[KEY_YEAR] = df_x_rs["date"].str[:4]
    df_x_rs[KEY_YEAR] = df_x_rs[KEY_YEAR].astype(int)
    df_x_rs = df_x_rs.set_index([KEY_LOC, KEY_YEAR, "date"])[[RS_FPAR]]

    dfs_x = (
        df_x_soil,
        df_x_meteo,
        df_x_rs,
    )

    df_y, dfs_x = _align_data(df_y, dfs_x)

    return df_y, dfs_x

def load_dfs_maize_es() -> tuple:
    return load_dfs("maize", "ES")


def load_dfs_maize_nl() -> tuple:
    return load_dfs("maize", "NL")


def load_dfs_maize() -> tuple:
    df_y_es, dfs_x_es = load_dfs("maize", "ES")
    df_y_nl, dfs_x_nl = load_dfs("maize", "NL")

    df_y = pd.concat(
        [
            df_y_es,
            df_y_nl,
        ],
        axis=0,
    )

    dfs_x = tuple(
        pd.concat([df_x_us, df_x_fr], axis=0)
        for df_x_us, df_x_fr in zip(dfs_x_es, dfs_x_nl)
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


def load_dfs_wheat_nl() -> tuple:
    return load_dfs("wheat", "NL")
