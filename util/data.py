import pandas as pd
import os


def normalize_data(data, params, normalization="standard"):
    exclude_keys = [k for k in data if k not in params]
    if normalization == "standard":
        return {
            **{key: data[key] for key in exclude_keys},
            **{
                key: ((data[key] - params[key]["mean"]) / params[key]["std"])
                for key in params
            },
        }
    elif normalization == "min-max":
        return {
            **{key: data[key] for key in exclude_keys},
            **{
                key: (
                    (data[key] - params[key]["min"])
                    / (params[key]["max"] - params[key]["min"])
                )
                for key in params
            },
        }
    else:
        raise Exception(f"Unsupported normalization {normalization}")


def csv_to_pandas(data_path, filename, index_cols):
    path = os.path.join(data_path, filename)
    df = pd.read_csv(path, index_col=index_cols)

    return df


def dataset_to_pandas(dataset):
    data = []
    data_cols = None
    for i in range(len(dataset)):
        data_item = dataset[i]
        if data_cols is None:
            data_cols = list(data_item.keys())

        data.append([data_item[c] for c in data_cols])

    return pd.DataFrame(data, columns=data_cols)


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


from config import PATH_DATA_DIR
import pandas as pd

if __name__ == "__main__":
    data_path = os.path.join(PATH_DATA_DIR, "data_US", "county_data")
    yield_csv = os.path.join(data_path, "YIELD_COUNTY_US.csv")
    yield_df = pd.read_csv(yield_csv, header=0)

    all_years = list(yield_df["FYEAR"].unique())
    test_years = [2012, 2018]
    train_years = [yr for yr in all_years if yr not in test_years]
    train_df = yield_df[yield_df["FYEAR"].isin(train_years)]
    test_df = yield_df[yield_df["FYEAR"].isin(test_years)]
    train_trend = trend_features(train_df, "COUNTY_ID", "FYEAR", "YIELD", 5)
    train_trend = train_trend.dropna(axis=0)
    train_trend = train_trend[train_trend["FYEAR"].isin(train_years)]
    print("\n")
    print("Training trend features. Should not include features from test years")
    print(
        train_trend[train_trend["FYEAR"] == 2013]
        .sort_values(by=["COUNTY_ID", "FYEAR"])
        .head(5)
        .to_string()
    )

    test_trend = trend_features(
        pd.concat([train_df, test_df], axis=0), "COUNTY_ID", "FYEAR", "YIELD", 5
    )
    test_trend = test_trend.dropna(axis=0)
    test_trend = test_trend[test_trend["FYEAR"].isin(test_years)]
    print("\n")
    print("Test trend features")
    print(test_trend.sort_values(by=["COUNTY_ID", "FYEAR"]).head(5).to_string())
