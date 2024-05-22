import os
import pandas as pd


def fortnight_from_date(date_str):
    month = date_str[4:6]
    day_of_month = int(date_str[6:])
    fortnight_number = (int(month) -1) * 2 
    if (day_of_month <= 15):
        return fortnight_number + 1
    else:
        return fortnight_number + 2

def dekad_from_date(date_str):
    month = int(date_str[4:6])
    day_of_month = int(date_str[6:])
    dekad = (month - 1) * 3
    if (day_of_month <= 10):
        dekad += 1
    elif (day_of_month <= 20):
        dekad += 2
    else:
        dekad += 3 

    return dekad

def add_period(df, time_step):
    # NOTE expects data column in string format
    # add a period column based on time step
    if (time_step == "month"):
        df["period"] = df["date"].str[4:6]
    elif(time_step == "fortnight"):
        df["period"] = df.apply(lambda r: fortnight_from_date(r["date"]), axis=1)
    elif (time_step == "dekad"):
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
    ft_df = ft_df.pivot_table(index=index_cols, columns=period_col,
                              values=ft_cols.values()).fillna(0).reset_index()

    # combine names of two column levels
    ft_df.columns = [first + second for first, second in ft_df.columns]

    return ft_df

# Feature 4: Growing degree days
# TODO: What is the formula?

# Feature 5: Vernalization requirement
# TODO: What is the formula

def count_threshold(df, index_cols, period_col,
                    indicator, threshold_exceed=True, threshold=0.0, ft_name=None):
    groupby_cols = index_cols + [period_col]
    if (threshold_exceed):
        filter_condition = df[indicator] > threshold
    else:
        filter_condition = df[indicator] < threshold

    ft_df = df[filter_condition].groupby(groupby_cols).agg(FEATURE=(indicator, "count"))
    if (ft_name is not None):
        ft_df = ft_df.rename(columns={"FEATURE" : ft_name})

    # pivot to add a feature column for each period
    ft_df = ft_df.pivot_table(index=index_cols, columns=period_col,
                              values=ft_name).fillna(0).reset_index()

    # rename period cols
    period_cols = ft_df.columns[len(index_cols):]
    rename_cols = { p : ft_name + "p" + p for p in period_cols }
    ft_df = ft_df.rename(columns=rename_cols)

    return ft_df

from datasets.dataset import Dataset
from util.data import data_to_pandas
from config import KEY_DATES

def unpack_time_series(df, indicators):
    # for a data source, dates should match across all indicators
    df["date"] = df.apply(lambda r: r[KEY_DATES][indicators[0]], axis=1)

    # explode time series columns and dates
    df = df.explode(indicators + ["date"]).drop(columns=[KEY_DATES])
    df = df.astype({"date" : str})
    # pandas tends to format date as "YYYY-mm-dd", remove the hyphens
    df["date"] = df["date"].str.replace("-", "")

    return df

def test_feature_design():
    max_feature_cols = ["fapar"] # ["ndvi", "fapar"]
    avg_feature_cols = ["tmin", "tmax"] #, "tmax", "tavg", "prec", "rad"]
    count_thresh_cols = {
        "tmin" : ["<", "0"], # degrees
        "tmax" : [">", "35"], # degrees
        "prec_l" : ["<", "50"], # mm
        "prec_h" : [">", "100"], # mm (per day)
    }

    index_cols = ["loc_id", "year"]
    test_nl_data = Dataset.load("test_softwheat_nl")
    data_df = data_to_pandas(test_nl_data)
    rs_indicators = ["fapar"]
    meteo_indicators = ["tmin", "tmax", "tavg", "prec"]
    rs_df = data_df[index_cols + ["dates"] + rs_indicators].copy()
    meteo_df = data_df[index_cols + ["dates"] + meteo_indicators].copy()
    rs_df = unpack_time_series(rs_df, rs_indicators)
    meteo_df = unpack_time_series(meteo_df, meteo_indicators)

    time_step = "month"

    group_cols = index_cols + ["period"]

    rs_df = add_period(rs_df, time_step)
    meteo_df = add_period(meteo_df, time_step)

    max_aggrs = { ind : "max" for ind in max_feature_cols }
    avg_aggrs = { ind : "mean" for ind in avg_feature_cols }

    # NOTE: If combining max and avg aggregation
    # all_aggrs = {
    #     **max_aggrs,
    #     **avg_aggrs,
    # }

    max_ft_cols = { ind : "max" + ind for ind in max_feature_cols }
    avg_ft_cols = { ind : "mean" + ind for ind in avg_feature_cols }
    rs_fts = aggregate_by_period(rs_df, index_cols, "period",
                                max_aggrs, max_ft_cols)
    meteo_fts = aggregate_by_period(meteo_df, index_cols, "period",
                                    avg_aggrs, avg_ft_cols)

    print(rs_fts.head(5))
    print(meteo_fts.head(5))

    count_fts = meteo_fts[index_cols].copy()
    for ind, thresh in count_thresh_cols.items():
        threshold_exceed = thresh[0]
        threshold = float(thresh[1])
        if ("_" in ind):
            ind = ind.split("_")[0]

        ft_name = ind + "".join(thresh)
        ind_fts = count_threshold(meteo_df, index_cols, "period",
                                ind, threshold_exceed, threshold, ft_name)
        if (not ind_fts.empty):
            count_fts = count_fts.merge(ind_fts, on=index_cols, how="outer")
            count_fts = count_fts.fillna(0.0)

    print(count_fts.head(5))

test_feature_design()
