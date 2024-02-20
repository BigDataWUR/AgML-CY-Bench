import os
import pandas as pd
import numpy as np

from datasets.dataset import Dataset
from models.naive_models import AverageYieldModel
from config import PATH_DATA_DIR


def test_predict_item():
    model = AverageYieldModel(group_cols=["COUNTY_ID"], label_col="YIELD")
    data_sources = {
        "YIELD": {
            "index_cols": ["COUNTY_ID", "FYEAR"],
            "sel_cols": ["YIELD"],
        }
    }

    data_path = os.path.join(PATH_DATA_DIR, "data_US", "county_data")
    yield_csv = os.path.join(data_path, "YIELD_COUNTY_US.csv")
    yield_df = pd.read_csv(yield_csv, index_col=data_sources["YIELD"]["index_cols"])
    data_dfs = {"YIELD": yield_df}
    dataset = Dataset(data_dfs, data_sources, "COUNTY_ID", "FYEAR")
    filtered_df = yield_df[yield_df.index.get_level_values(0) == "AL_AUTAUGA"]
    expected_pred = filtered_df["YIELD"].mean()
    model.fit(dataset)
    test_data = {
        "COUNTY_ID": "AL_AUTAUGA",
        "FYEAR": 2018,
        "YIELD": yield_df.loc[("AL_AUTAUGA", 2018)]["YIELD"],
    }

    test_preds, _ = model.predict_item(test_data)
    assert np.round(test_preds[0], 2) == np.round(expected_pred, 2)
