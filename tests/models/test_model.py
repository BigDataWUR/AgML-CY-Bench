import os
import pandas as pd
import numpy as np

from datasets.dataset import Dataset
from models.naive_models import AverageYieldModel
from config import PATH_DATA_DIR


def get_model_predictions(model, sel_county, sel_year):
    test_data = {
        "COUNTY_ID": sel_county,
        "FYEAR": sel_year,
    }

    return model.predict_item(test_data)


def test_average_yield_model():
    model = AverageYieldModel(group_cols=["COUNTY_ID"], label_col="YIELD")
    data_path = os.path.join(PATH_DATA_DIR, "data_US", "county_data")
    yield_csv = os.path.join(data_path, "YIELD_COUNTY_US.csv")
    yield_df = pd.read_csv(yield_csv, index_col=["COUNTY_ID", "FYEAR"])
    dataset = Dataset(yield_df, feature_dfs=[])
    filtered_df = yield_df[yield_df.index.get_level_values(0) == "AL_AUTAUGA"]
    expected_pred = filtered_df["YIELD"].mean()
    model.fit(dataset)

    # test prediction for an existing item
    sel_county = "AL_AUTAUGA"
    sel_year = 2018
    assert sel_county in yield_df.index.get_level_values(0)
    filtered_df = yield_df[yield_df.index.get_level_values(0) == sel_county]
    expected_pred = filtered_df["YIELD"].mean()
    test_preds, _ = get_model_predictions(model, sel_county, sel_year)
    assert np.round(test_preds[0], 2) == np.round(expected_pred, 2)

    # test prediction for an existing item
    sel_county = "CA_SAN_MATEO"
    assert sel_county not in yield_df.index.get_level_values(0)
    expected_pred = yield_df["YIELD"].mean()
    test_preds, _ = get_model_predictions(model, sel_county, sel_year)
    assert np.round(test_preds[0], 2) == np.round(expected_pred, 2)
