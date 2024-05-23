import os
import pandas as pd

from config import KEY_LOC, KEY_YEAR, KEY_DATES


def fortnight_from_date(date_str):
    month = date_str[4:6]
    day_of_month = int(date_str[6:])
    fortnight_number = (int(month) - 1) * 2
    if day_of_month <= 15:
        return fortnight_number + 1
    else:
        return fortnight_number + 2


def dekad_from_date(date_str):
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
    index_df,
    index_cols,
    period_col,
    indicator,
    threshold_exceed=True,
    threshold=0.0,
    ft_name=None,
):
    groupby_cols = index_cols + [period_col]
    if threshold_exceed:
        filter_condition = df[indicator] > threshold
    else:
        filter_condition = df[indicator] < threshold

    if df[filter_condition].empty:
        ft_df = index_df.copy()
    else:
        ft_df = (
            df[filter_condition]
            .groupby(groupby_cols)
            .agg(FEATURE=(indicator, "count"))
            .reset_index()
        )
        if ft_name is not None:
            ft_df = ft_df.rename(columns={"FEATURE": ft_name})

        # pivot to add a feature column for each period
        ft_df = (
            ft_df.pivot_table(index=index_cols, columns=period_col, values=ft_name)
            .fillna(0)
            .reset_index()
        )

    # fill in missing features with zeros and rename period cols
    period_cols = df["period"].unique()
    rename_cols = {p: ft_name + "p" + p for p in period_cols}
    ft_df = ft_df.rename(columns=rename_cols)
    missing_features = [ft for ft in rename_cols.values() if ft not in ft_df.columns]
    for ft in missing_features:
        ft_df[ft] = 0.0

    return ft_df


def unpack_time_series(df, indicators):
    # for a data source, dates should match across all indicators
    df["date"] = df.apply(lambda r: r[KEY_DATES][indicators[0]], axis=1)

    # explode time series columns and dates
    df = df.explode(indicators + ["date"]).drop(columns=[KEY_DATES])
    df = df.astype({"date": str})
    # pandas tends to format date as "YYYY-mm-dd", remove the hyphens
    df["date"] = df["date"].str.replace("-", "")

    return df


def design_features(
    weather_df, soil_df, fapar_df, ndvi_df=None, et0_df=None, soil_moisture_df=None
):
    # for soil, we need to comput water holding capacity
    # TODO: 1. not needed for cybench data. Remove later.
    # TODO: 2. Add code to make drainage class a categorical feature.
    if "sm_whc" not in soil_df.columns:
        soil_df["sm_whc"] = soil_df["sm_fc"] - soil_df["sm_wp"]
        soil_features = soil_df[[KEY_LOC, "sm_whc"]]
    else:
        soil_features = soil_df[[KEY_LOC, "sm_whc"]]

    # Feature design for time series
    # TODO: 1. add code for cumulative features
    # TODO: 2. add code for ET0, ndvi, soil moisture
    index_cols = [KEY_LOC, KEY_YEAR]
    period_length = "month"
    max_feature_cols = ["fapar"]  # ["ndvi", "fapar"]
    avg_feature_cols = ["tmin", "tmax"]  # , "tmax", "tavg", "prec", "rad"]
    count_thresh_cols = {
        "tmin": ["<", "0"],  # degrees
        "tmax": [">", "35"],  # degrees
        "prec_l": ["<", "50"],  # mm
        "prec_h": [">", "100"],  # mm (per day)
    }

    fapar_df = add_period(fapar_df, period_length)
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
    rs_fts = aggregate_by_period(fapar_df, index_cols, "period", max_aggrs, max_ft_cols)
    weather_fts = aggregate_by_period(
        weather_df, index_cols, "period", avg_aggrs, avg_ft_cols
    )

    # count time steps matching threshold conditions
    # NOTE: Passing index_df to count_threshold()
    #       to skip calling drop_duplicates() on weather_df multiple times.
    index_df = weather_df[index_cols].drop_duplicates()
    for ind, thresh in count_thresh_cols.items():
        threshold_exceed = thresh[0]
        threshold = float(thresh[1])
        if "_" in ind:
            ind = ind.split("_")[0]

        ft_name = ind + "".join(thresh)
        ind_fts = count_threshold(
            weather_df,
            index_df,
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
