import pandas as pd
import numpy as np

from cybench.config import (
    KEY_LOC,
    KEY_YEAR,
    SPINUP_DAYS,
    FORECAST_LEAD_TIME,
    CROP_CALENDAR_DOYS,
    CROP_CALENDAR_DATES,
)


def add_cutoff_days(df: pd.DataFrame, lead_time: str):
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
            df["cutoff_days"] = (df["season_length"] // 2).astype(int)
        elif lead_time == "quarter-of-season":
            df["cutoff_days"] = (df["season_length"] // 4).astype(int)
        else:
            raise Exception(f'Unrecognized lead time "{lead_time}"')

    return df


def compute_crop_season_window(df, min_year, max_year, lead_time=FORECAST_LEAD_TIME):
    """Compute crop season window used for forecasting.

    Args:
        df (pd.DataFrame): crop calendar data
        min_year (int): earliest year in target data
        max_year (int): latest year in target data
        lead_time (str): forecast lead time option

    Returns:
        the same DataFrame with crop season window information
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

    df = add_cutoff_days(df, lead_time)
    df["cutoff_date"] = df["eos_date"] - pd.to_timedelta(df["cutoff_days"], unit="d")
    df["season_window_length"] = np.where(
        df["season_length"] + SPINUP_DAYS <= 365,
        df["season_length"] + SPINUP_DAYS - df["cutoff_days"],
        365 - df["cutoff_days"],
    )

    # drop redundant information
    df.drop(columns=CROP_CALENDAR_DOYS + ["season_length", "cutoff_days"], inplace=True)

    return df


def align_to_crop_season_window(df: pd.DataFrame, crop_season_df: pd.DataFrame):
    """Align time series data to crop season window (includes lead time and spinup).

    Args:
        df (pd.DataFrame): time series data
        crop_season_df (pd.DataFrame): crop season data
        lead_time (str): forecast lead time option

    Returns:
        the input DataFrame with data aligned to crop season and trimmed to lead time
    """
    select_cols = list(df.columns)

    # Merge with crop season data
    df = df.merge(
        crop_season_df[[KEY_LOC, KEY_YEAR] + CROP_CALENDAR_DATES],
        on=[KEY_LOC, KEY_YEAR],
    )

    # The next crop season starts right after current year's harvest.
    df[KEY_YEAR] = np.where(df["date"] > df["eos_date"], df[KEY_YEAR] + 1, df[KEY_YEAR])
    df.drop(columns=CROP_CALENDAR_DATES, inplace=True)

    # merge with crop season data again because we changed KEY_YEAR
    df = df.merge(crop_season_df, on=[KEY_LOC, KEY_YEAR])

    # Validate sos_date: date - sos_date should not be more than 366 days
    assert df[(df["date"] - df["sos_date"]).dt.days > 366].empty

    # Validate eos_date: eos_date - date should not be more than 366 days
    assert df[(df["eos_date"] - df["date"]).dt.days > 366].empty

    # Drop years with not enough data for a season
    # NOTE: We cannot filter with df.groupby(...)["date"].transform("count")
    # because ndvi and fpar don't have daily values.
    df["min_date"] = df.groupby([KEY_LOC, KEY_YEAR], observed=True)["date"].transform(
        "min"
    )
    df["max_date"] = df.groupby([KEY_LOC, KEY_YEAR], observed=True)["date"].transform(
        "max"
    )
    df = df[(df["max_date"] - df["min_date"]).dt.days >= df["season_window_length"]]
    # Keep season_window_length, i.e. season_length + SPINUP_DAYS - cutoff_days
    df = df[(df["cutoff_date"] - df["date"]).dt.days <= df["season_window_length"]]

    # Trim to lead time
    df = df[df["date"] <= df["cutoff_date"]]

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
