import os
import numpy as np
import pandas as pd

from config import KEY_LOC, KEY_YEAR, KEY_DATES


def fortnight_from_date(date_str):
    """Get the fortnight number from date.

    Args:
      date_str: date string in YYYYmmdd format

    Returns:
      Fortnight number, "YYYY0101" to "YYYY0115" -> 1.
    """
    month = date_str[4:6]
    day_of_month = int(date_str[6:])
    fortnight_number = (int(month) - 1) * 2
    if day_of_month <= 15:
        return fortnight_number + 1
    else:
        return fortnight_number + 2


def dekad_from_date(date_str):
    """Get the dekad number from date.

    Args:
      date_str: date string in YYYYmmdd format

    Returns:
      Dekad number, e.g. "YYYY0101" to "YYYY010" -> 1,
                         "YYYY0111" to "YYYY0120" -> 2,
                         "YYYY0121" to "YYYY0131" -> 3
    """
    date_str = date_str.replace("-", "").replace("/", "")
    month = int(date_str[4:6])
    day_of_month = int(date_str[6:])
    dekad = (month - 1) * 3
    if day_of_month <= 10:
        dekad += 1
    elif day_of_month <= 20:
        dekad += 2
    else:
        dekad += 3

    return dekad


def add_period(df, period_length):
    """Add a period column.

    Args:
      df : pd.DataFrame

      period_length: string, which can be "month", "fortnight" or "dekad"

    Returns:
      pd.DataFrame
    """
    # NOTE expects data column in string format
    # add a period column based on time step
    if period_length == "month":
        df["period"] = df["date"].str[4:6]
    elif period_length == "fortnight":
        df["period"] = df.apply(lambda r: fortnight_from_date(r["date"]), axis=1)
    elif period_length == "dekad":
        df["period"] = df.apply(lambda r: dekad_from_date(r["date"]), axis=1)

    return df


# Period can be a month or fortnight (biweekly or two weeks)
# Period sum of TAVG, TMIN, TMAX, PREC
def aggregate_by_period(df, index_cols, period_col, aggrs, ft_cols):
    """Aggregate data into features by period.

    Args:
      df : pd.DataFrame

      index_cols: list of indices, which are location and year

      period_col: string, column added by add_period()

      aggrs: dict containing columns to aggregate (keys) and corresponding
             aggregation function (values)

      ft_cols: dict for renaming columns to feature columns

    Returns:
      pd.DataFrame with features
    """
    groupby_cols = index_cols + [period_col]
    ft_df = df.groupby(groupby_cols).agg(aggrs).reset_index()

    # rename to indicate aggregation
    ft_df = ft_df.rename(columns=ft_cols)

    # pivot to add a feature column for each period
    ft_df = (
        ft_df.pivot_table(index=index_cols, columns=period_col, values=ft_cols.values())
        .fillna(0)
        .reset_index()
    )

    # combine names of two column levels
    ft_df.columns = [first + second for first, second in ft_df.columns]

    return ft_df


# Feature 4: Growing degree days
# TODO: What is the formula?

# Feature 5: Vernalization requirement
# TODO: What is the formula


def count_threshold(
    df,
    index_cols,
    period_col,
    indicator,
    threshold_exceed=True,
    threshold=0.0,
    ft_name=None,
):
    """Aggregate data into features by period.

    Args:
      df : pd.DataFrame

      index_cols: list of indices, which are location and year

      period_col: string, column added by add_period()

      indicator: string, indicator column to aggregate

      threshold_exceed: boolean

      threshold: float

      ft_name: string name for aggregated indicator

    Returns:
      pd.DataFrame with features
    """
    groupby_cols = index_cols + [period_col]
    if threshold_exceed:
        threshold_lambda = lambda x: 1 if (x[indicator] > threshold) else 0
    else:
        threshold_lambda = lambda x: 1 if (x[indicator] < threshold) else 0

    df["meet_thresh"] = df.apply(threshold_lambda, axis=1)
    ft_df = df.groupby(groupby_cols).agg(FEATURE=("meet_thresh", "sum")).reset_index()
    # drop the column we added
    df = df.drop(columns=["meet_thresh"])

    if ft_name is not None:
        ft_df = ft_df.rename(columns={"FEATURE": ft_name})

    # pivot to add a feature column for each period
    ft_df = (
        ft_df.pivot_table(index=index_cols, columns=period_col, values=ft_name)
        .fillna(0)
        .reset_index()
    )

    # rename period cols
    period_cols = df["period"].unique()
    rename_cols = {p: ft_name + "p" + p for p in period_cols}
    ft_df = ft_df.rename(columns=rename_cols)

    return ft_df


def unpack_time_series(df, indicators):
    """Unpack time series from lists into separate rows by date.

    Args:
      df : pd.DataFrame

      indicators: list of indicators to unpack

    Returns:
      pd.DataFrame
    """
    # for a data source, dates should match across all indicators
    df["date"] = df.apply(lambda r: r[KEY_DATES][indicators[0]], axis=1)

    # explode time series columns and dates
    df = df.explode(indicators + ["date"]).drop(columns=[KEY_DATES])
    df = df.astype({"date": str})
    # pandas tends to format date as "YYYY-mm-dd", remove the hyphens
    df["date"] = df["date"].str.replace("-", "")

    return df


def design_features(weather_df, soil_df, fpar_df, ndvi_df=None, soil_moisture_df=None):
    """Design features based domain expertise.

    Args:
      weather_df : pd.DataFrame, weather variables

      soil_df: pd.DataFrame, soil properties

      fpar_df : pd.DataFrame, fraction of absorbed photosynthetically active radiation

      ndvi_df: pd.DataFrame, normalized difference vegetation index

      et0_df: pd.DataFrame, potential evapotraspiration

      soil_moisture_df: pd.DataFrame, soil moisture (surface and root zone)

    Returns:
      pd.DataFrame of features
    """
    if "drainage_class" in soil_df.columns:
        soil_features = soil_df.astype({"drainage_class": "category"})
    else:
        soil_features = soil_df

    # Feature design for time series
    # TODO: 1. add code for cumulative features
    # TODO: 2. add code for ET0, ndvi, soil moisture
    index_cols = [KEY_LOC, KEY_YEAR]
    period_length = "month"
    max_feature_cols = ["fpar"]  # ["ndvi", "fpar"]
    avg_feature_cols = ["tmin", "tmax"]  # , "tmax", "tavg", "prec", "rad"]
    count_thresh_cols = {
        "tmin": ["<", "0"],  # degrees
        "tmax": [">", "35"],  # degrees
        "prec_l": ["<", "50"],  # mm
        "prec_h": [">", "100"],  # mm (per day)
    }

    fpar_df = add_period(fpar_df, period_length)
    weather_df = add_period(weather_df, period_length)

    max_aggrs = {ind: "max" for ind in max_feature_cols}
    avg_aggrs = {ind: "mean" for ind in avg_feature_cols}

    # NOTE: If combining max and avg aggregation
    # all_aggrs = {
    #     **max_aggrs,
    #     **avg_aggrs,
    # }

    max_ft_cols = {ind: "max" + ind for ind in max_feature_cols}
    avg_ft_cols = {ind: "mean" + ind for ind in avg_feature_cols}
    rs_fts = aggregate_by_period(fpar_df, index_cols, "period", max_aggrs, max_ft_cols)
    weather_fts = aggregate_by_period(
        weather_df, index_cols, "period", avg_aggrs, avg_ft_cols
    )

    # count time steps matching threshold conditions
    for ind, thresh in count_thresh_cols.items():
        threshold_exceed = thresh[0]
        threshold = float(thresh[1])
        if "_" in ind:
            ind = ind.split("_")[0]

        ft_name = ind + "".join(thresh)
        ind_fts = count_threshold(
            weather_df,
            index_cols,
            "period",
            ind,
            threshold_exceed,
            threshold,
            ft_name,
        )

        weather_fts = weather_fts.merge(ind_fts, on=index_cols, how="left")
        weather_fts = weather_fts.fillna(0.0)

    all_fts = soil_features.merge(rs_fts, on=[KEY_LOC])
    all_fts = all_fts.merge(weather_fts, on=index_cols)
    return all_fts
