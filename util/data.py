import pandas as pd
import os


def csv_to_pandas(data_path, filename, index_cols):
    path = os.path.join(data_path, filename)
    df = pd.read_csv(path, index_col=index_cols)

    return df


def trend_features(df, group_col, year_col, value_col, trend_window):
    trend_fts = df.sort_values(by=[group_col, year_col])
    for i in range(trend_window, 0, -1):
        trend_fts[year_col + "-" + str(i)] = trend_fts.groupby([group_col])[
            year_col
        ].shift(i)
    for i in range(trend_window, 0, -1):
        trend_fts[value_col + "-" + str(i)] = trend_fts.groupby([group_col])[
            value_col
        ].shift(i)

    # print(trend_fts.head(10).to_string())
    return trend_fts
