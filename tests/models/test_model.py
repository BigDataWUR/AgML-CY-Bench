import os
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge

from datasets.dataset import Dataset
from models.naive_models import AverageYieldModel
from models.sklearn_model import SklearnModel
from models.nn_models import BaseNNModel, ExampleLSTM
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


def test_sklearn_model():
    data_path = os.path.join(PATH_DATA_DIR, "data_US", "county_features")
    # Training dataset
    train_csv = os.path.join(data_path, "grain_maize_US_train.csv")
    train_df = pd.read_csv(train_csv, index_col=["COUNTY_ID", "FYEAR"])
    train_yields = train_df[["YIELD"]].copy()
    feature_cols = [c for c in train_df.columns if c != "YIELD"]
    train_features = train_df[feature_cols].copy()
    train_dataset = Dataset(train_yields, [train_features])

    # Test dataset
    test_csv = os.path.join(data_path, "grain_maize_US_train.csv")
    test_df = pd.read_csv(test_csv, index_col=["COUNTY_ID", "FYEAR"])
    test_yields = test_df[["YIELD"]].copy()
    test_features = test_df[feature_cols].copy()
    test_dataset = Dataset(test_yields, [test_features])

    # Model
    ridge = Ridge(alpha=0.5)
    model = SklearnModel(
        ridge,
        index_cols=["COUNTY_ID", "FYEAR"],
        feature_cols=feature_cols,
        label_col="YIELD",
    )
    model.fit(train_dataset)
    test_preds, _ = model.predict(test_dataset)
    assert test_preds.shape[0] == len(test_dataset)

    # Model with hyperparameter optimization
    fit_params = {
        "optimize_hyperparameters": True,
        "param_space": {"estimator__alpha": [0.01, 0.1, 0.0, 1.0, 5.0, 10.0]},
    }
    model.fit(train_dataset, **fit_params)
    test_preds, _ = model.predict(test_dataset)
    assert test_preds.shape[0] == len(test_dataset)

def test_nn_model():
    #TODO Finish this test
    data_path = os.path.join(PATH_DATA_DIR, "data_US", "county_features")
    # Training dataset
    train_csv = os.path.join(data_path, "grain_maize_US_train.csv")
    train_df = pd.read_csv(train_csv, index_col=["COUNTY_ID", "FYEAR"])
    train_yields = train_df[["YIELD"]].copy()
    feature_cols = [c for c in train_df.columns if c != "YIELD"]
    train_features = train_df[feature_cols].copy()
    train_dataset = Dataset(train_yields, [train_features])

    # Test dataset
    test_csv = os.path.join(data_path, "grain_maize_US_train.csv")
    test_df = pd.read_csv(test_csv, index_col=["COUNTY_ID", "FYEAR"])
    test_yields = test_df[["YIELD"]].copy()
    test_features = test_df[feature_cols].copy()
    test_dataset = Dataset(test_yields, [test_features])

    # Model
    network = ExampleLSTM(input_size=len(feature_cols), hidden_size=64, num_layers=2)
    model = BaseNNModel(network, device="cpu")
    model.fit(train_dataset)
    test_preds, _ = model.predict(test_dataset)
    assert test_preds.shape[0] == len(test_dataset)

