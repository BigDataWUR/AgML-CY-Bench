import os
import pandas as pd
import numpy as np

from datasets.dataset import Dataset
from models.trend_models import TrendModel
from util.data import get_trend_features
from config import PATH_DATA_DIR


def test_trend_model():
    trend_window = 5
    x_cols = ["FYEAR-" + str(i) for i in range(trend_window, 0, -1)]
    y_cols = ["YIELD-" + str(i) for i in range(trend_window, 0, -1)]
    model = TrendModel(x_cols, y_cols, "FYEAR")
    data_path = os.path.join(PATH_DATA_DIR, "data_US", "county_data")
    yield_csv = os.path.join(data_path, "YIELD_COUNTY_US.csv")
    yield_df = pd.read_csv(yield_csv, header=0)
    # TODO: Create trend features before alignment
    trend_fts, x_cols, y_cols = get_trend_features(yield_df, "COUNTY_ID", "FYEAR", "YIELD", trend_window)

    # align data
    yield_df = yield_df.merge(trend_fts[["COUNTY_ID", "FYEAR"]], on=["COUNTY_ID", "FYEAR"])
    yield_df = yield_df.set_index(["COUNTY_ID", "FYEAR"])
    trend_fts = trend_fts.set_index(["COUNTY_ID", "FYEAR"])

    # create dataset
    dataset = Dataset(yield_df, feature_dfs=[trend_fts])
    model.fit(dataset)

    # dummy test data
    dummy_x_vals = list(range(2016, 2021))
    dummy_y_vals = list(range(2, 7))
    test_data = {
        "COUNTY_ID": "AL_AUTAUGA",
        "FYEAR": dummy_x_vals[-1] + 1,
    }

    for i, c in enumerate(x_cols):
        test_data[c] = dummy_x_vals[i]

    for i, c in enumerate(y_cols):
        test_data[c] = dummy_y_vals[i]

    # in dummy test data, yield increases by 1 every year
    expected_pred = dummy_y_vals[-1] + 1
    test_preds, _ = model.predict_item(test_data)
    print(test_preds, expected_pred)
    assert np.round(test_preds[0], 2) == np.round(expected_pred, 2)

test_trend_model()