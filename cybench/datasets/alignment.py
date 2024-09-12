import pandas as pd
import numpy as np

from cybench.config import (
    KEY_LOC,
    KEY_YEAR,
    SPINUP_DAYS,
    FORECAST_LEAD_TIME,
)


def _add_cutoff_days(df: pd.DataFrame, lead_time: str):
    """Add a column with cutoff days relative to end of season.

    Args:
        df (pd.DataFrame): time series data
        lead_time (str): lead_time option

    Returns:
        the same DataFrame with column added
    """
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


def trim_to_lead_time(
    df: pd.DataFrame, crop_cal_df: pd.DataFrame, lead_time: str = FORECAST_LEAD_TIME
):
    """Align time series data to crop season.

    Args:
        df (pd.DataFrame): time series data
        crop_cal_df (pd.DataFrame): crop calendar data
        lead_time (str): lead time option

    Returns:
        a tuple with the input DataFrame and crop calendar (sos_date, eos_date)
        NOTE: We are returning crop calendar as dynamic data (including year in it)
    """
    select_cols = list(df.columns)

    # Merge with crop calendar
    df = df.merge(crop_cal_df, on=[KEY_LOC, KEY_YEAR])

    # The next new year starts right after this year's harvest.
    df[KEY_YEAR] = np.where(df["date"] > df["eos_date"], df[KEY_YEAR] + 1, df[KEY_YEAR])

    # Fix sos_date for data that are after the eos_date.
    # NOTE: In preprocess_crop_calendar, we use sos_date from previous year
    # for the case where sos > eos. We add 1 year to make
    # sure sos_date and eos_date make sense when date > eos_date.
    df["sos_date"] = np.where(
        (df["date"] > df["eos_date"]) & (df["sos"] > df["eos"]),
        df["sos_date"] + pd.offsets.DateOffset(years=1),
        df["sos_date"],
    )

    # Validate sos_date: date - sos_date should not be more than 366 days
    assert df[(df["date"] - df["sos_date"]).dt.days > 366].empty

    # Fix eos_date for data that are after the current season's eos_date.
    # Say eos_date for maize, NL is 20010728. All data after 20010728 belong to
    # the season that ends in 2002. We change the eos_date for those data to be
    # next year's eos_date.
    # NOTE: This works only for static crop calendar.
    df["eos_date"] = np.where(
        (df["date"] > df["eos_date"]),
        # select eos_date for the next year
        df["eos_date"] + pd.offsets.DateOffset(years=1),
        df["eos_date"],
    )

    # Validate eos_date: eos_date - date should not be more than 366 days
    assert df[(df["eos_date"] - df["date"]).dt.days > 366].empty

    # Total time series length including spinup days before the start of season.
    df["ts_length"] = np.where(
        df["season_length"] + SPINUP_DAYS <= 365,
        df["season_length"] + SPINUP_DAYS,
        365,
    )

    # Drop years with not enough data for a season
    # NOTE: We cannot filter with df.groupby(...)["date"].transform("count")
    # because ndvi and fpar don't have daily values.
    df["min_date"] = df.groupby([KEY_LOC, KEY_YEAR], observed=True)["date"].transform(
        "min"
    )
    df["max_date"] = df.groupby([KEY_LOC, KEY_YEAR], observed=True)["date"].transform(
        "max"
    )
    df = df[(df["max_date"] - df["min_date"]).dt.days >= df["ts_length"]]
    # Keep spinup_days before sos, i.e. keep season_length + spinup_days
    df = df[(df["eos_date"] - df["date"]).dt.days <= df["ts_length"]]

    # Trim to lead time
    df = _add_cutoff_days(df, lead_time)
    df = df[(df["eos_date"] - df["date"]).dt.days >= df["cutoff_days"]]

    return df[select_cols]


def align_inputs_and_labels(df_y: pd.DataFrame, dfs_x: dict) -> tuple:
    """Align inputs and labels to have common indices (KEY_LOC, KEY_YEAR).
    NOTE: Input data returned may still contain more (KEY_LOC, KEY_YEAR)
    entries than label data. This is fine because the index of label data is
    is used to access input data and not the other way round.

    Args:
        df_y (pd.DataFrame): target or label data
        dfs_x (dict): key is input source and and value is pd.DataFrame

    Returns:
        the same DataFrame with dates aligned to crop season
    """
    # - Filter the label data based on presence within all feature data sets
    # - Filter feature data based on label data

    # Identify common locations and years
    index_y_selection = set(df_y.index.values)
    for df_x in dfs_x.values():
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

    # Filter input data by index_y_locations and index_y_years
    index_y_locations = set([loc_id for loc_id, _ in index_y_selection])
    index_y_years = set([year for _, year in index_y_selection])

    for x in dfs_x:
        df_x = dfs_x[x]
        if len(df_x.index.names) == 1:
            df_x = df_x.loc[list(index_y_locations)]

        if len(df_x.index.names) == 2:
            df_x = df_x.loc[list(index_y_selection)]

        if len(df_x.index.names) == 3:
            index_names = df_x.index.names
            df_x.reset_index(inplace=True)
            df_x = df_x[
                (df_x[KEY_YEAR] >= min(index_y_years))
                & (df_x[KEY_YEAR] <= max(index_y_years))
            ]
            df_x.set_index(index_names, inplace=True)

        dfs_x[x] = df_x

    return df_y, dfs_x
