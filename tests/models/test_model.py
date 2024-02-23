import os
import pandas as pd
import numpy as np

from datasets.dataset import Dataset
from models.trend_models import TrendModel
from config import PATH_DATA_DIR


def test_trend_model():
    trend_window = 5
    x_cols = ["FYEAR-" + str(i) for i in range(trend_window, 0, -1)]
    y_cols = ["YIELD-" + str(i) for i in range(trend_window, 0, -1)]
    model = TrendModel(x_cols, y_cols, "FYEAR")
    data_path = os.path.join(PATH_DATA_DIR, "data_US", "county_data")
    yield_csv = os.path.join(data_path, "YIELD_COUNTY_US.csv")
    yield_df = pd.read_csv(yield_csv, index_col=["COUNTY_ID", "FYEAR"])
    # TODO: Create trend features
    trend_fts = None
    dataset = Dataset(yield_df, feature_dfs=[trend_fts])
    expected_pred = 8.2
    model.fit(dataset)
    test_data = {
        "COUNTY_ID": "AL_AUTAUGA",
        "FYEAR": 2018,
        # TODO: add trend years and features
    }

    test_preds, _ = model.predict_item(test_data)
    assert np.round(test_preds[0], 2) == np.round(expected_pred, 2)
