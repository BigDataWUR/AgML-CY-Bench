import pandas as pd


def data_to_pandas(data_items):
    data = []
    data_cols = None
    for item in data_items:
        if data_cols is None:
            data_cols = list(item.keys())

        data.append([item[c] for c in data_cols])

    return pd.DataFrame(data, columns=data_cols)

def get_trend_features(df: pd.DataFrame, loc_id_col: str,
                       year_col: str, value_col: str, trend_window: int):
    """Creates trend features.

    Args:
      df: a pandas DataFrame with [loc_id_col, year_col, value_col]

      loc_id_col: name of location id column (e.g. "REGION")

      year_col: name of year column (e.g. "YEAR")

      value_col: name of value column (e.g. "YIELD")

      trend_window: number of years

    Returns:
      A tuple containing a pd.DataFrame, a list of x columns and a list of y columns.
    """
    trend_fts = df.sort_values(by=[loc_id_col, year_col])
    for i in range(trend_window, 0, -1):
        trend_fts[year_col + "-" + str(i)] = trend_fts.groupby([loc_id_col])[
            year_col
        ].shift(i)
    for i in range(trend_window, 0, -1):
        trend_fts[value_col + "-" + str(i)] = trend_fts.groupby([loc_id_col])[
            value_col
        ].shift(i)

    x_cols = [year_col + "-" + str(i) for i in range(trend_window, 0, -1)]
    y_cols = [value_col + "-" + str(i) for i in range(trend_window, 0, -1)]
    trend_fts = trend_fts.drop(columns=[value_col])
    trend_fts = trend_fts.dropna(axis=0)
    # print(trend_fts.head(10).to_string())

    return trend_fts, x_cols, y_cols