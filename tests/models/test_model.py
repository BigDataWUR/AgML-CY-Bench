import os
import pandas as pd
import numpy as np

from datasets.dataset import Dataset
from models.naive_models import AverageYieldModel
from models.sklearn_model import SklearnBaseModel
from config import PATH_DATA_DIR


def test_average_yield_model():
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

def test_sklearn_base_model():
    data_sources = {
        "YIELD": {
            "index_cols": ["COUNTY_ID", "FYEAR"],
            "sel_cols": ["YIELD"],
        }
    }

    data_path = os.path.join(PATH_DATA_DIR, "data_US", "county_features")
    combined_csv = os.path.join(data_path, "grain_maize_US.csv")
    combined_df = pd.read_csv(combined_csv, index_col=data_sources["YIELD"]["index_cols"])
    yield_df = combined_df[["YIELD"]].copy()
    feature_cols = [c for c in combined_df.columns if c != "YIELD"]
    features_df = combined_df[feature_cols].copy()
    data_sources["FEATURES"] = {
        "index_cols": ["COUNTY_ID", "FYEAR"],
        "sel_cols": feature_cols,
    }

    data_dfs = {
        "YIELD": yield_df,
        "FEATURES" : features_df
    }

    dataset = Dataset(data_dfs, data_sources, "COUNTY_ID", "FYEAR")
    train_years = list(range(2000, 2013))
    test_years = list(range(2013, 2019))
    train_dataset, test_dataset = dataset.split_on_years((train_years, test_years))
    model = SklearnBaseModel(index_cols=["COUNTY_ID", "FYEAR"],
                             feature_cols=feature_cols,
                             label_col="YIELD")
    model.fit(train_dataset)
    test_data = {
        "COUNTY_ID": "AL_AUTAUGA",
        "FYEAR": 2018,
        "YIELD": yield_df.loc[("AL_AUTAUGA", 2018)]["YIELD"],
    }

    test_preds, _ = model.predict_item(test_data)
    print(test_preds[0])
    test_preds, _ = model.predict_item(test_dataset)
    print(test_preds[0])
