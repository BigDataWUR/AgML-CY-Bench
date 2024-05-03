import pandas as pd
from datetime import date
from datetime import timedelta


def aggregate_feature(df, group_cols,
                      indicators, ft_names,
                      time_col, time_start, time_end,
                      agg_fn="mean"):
    filter_condition = (df[time_col] >= time_start) & (df[time_col] <= time_end)
    agg_dict = { ind : agg_fn for ind in indicators}
    rename_cols = { ind : ft_names[i] for i, ind in enumerate(indicators)}
    ft_df = df[filter_condition].groupby(group_cols).agg(agg_dict).reset_index()
    return ft_df.rename(columns=rename_cols)

def count_threshold_feature(df, group_cols,
                            indicator, ft_name,
                            date_start, date_end,
                            threshold=0.0, threshold_exceed=True):
    filter_condition = (df["date"] >= date_start) & (df["date"] <= date_end)
    if (threshold_exceed):
        filter_condition &= df[indicator] > threshold
    else:
        filter_condition &= df[indicator] < threshold

    ft_df = df[filter_condition].groupby(group_cols).agg(FEATURE=(indicator, "count"))
    return ft_df.rename(columns={"FEATURE" : ft_name})    

def count_successive_feature(df, group_cols,
                             indicator, ft_name,
                             date_start, date_end,
                             threshold=0.0, threshold_exceed=True):
    pass

max_feature_cols = ["fapar"] # ["ndvi", "fapar"]
avg_feature_cols = [] # ["tmin", "tmax", "tavg", "prec", "rad"]
count_thresh_cols = [] # ["tmin", "tmax", "prec"]
count_sccessive_thresh_cols = [] # ["tmin", "tmax", "prec"]

df = pd.read_csv("REMOTE_SENSING_COUNTY_US.csv", header=0)
df["date"] = pd.to_datetime(df["date"], format="%Y%m%d").dt.date
df = df[df["loc_id"] == "US-19-001"]
group_cols = ["loc_id", "year"]

# NOTE for other time steps, need to create a time step column
time_steps = sorted(df["dekad"].unique())
all_fts = None
for time_step in time_steps:
    if (max_feature_cols):
        ft_names = ["max_" + mxcol + str(time_step) for mxcol in max_feature_cols]
        ft_df = aggregate_feature(df, group_cols,
                                max_feature_cols, ft_names,
                                "dekad", time_step, time_step + 1,
                                agg_fn="max")
        
    if (avg_feature_cols):
        ft_names = ["avg_" + avcol + str(time_step) for avcol in avg_feature_cols]
        ft_df = aggregate_feature(df, group_cols,
                                  avg_feature_cols, ft_names,
                                  "dekad", time_step, time_step + 1,
                                  agg_fn="mean")

    if (all_fts is None):
        all_fts = ft_df
    else:
        all_fts = pd.merge(all_fts, ft_df, on=group_cols)

print(all_fts.head(5))