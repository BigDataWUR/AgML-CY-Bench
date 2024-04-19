import os
import torch
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge

from datasets.dataset import Dataset
from datasets.dataset_torch import TorchDataset
from models.naive_models import AverageYieldModel
from models.sklearn_model import SklearnModel
from models.nn_models import ExampleLSTM
from evaluation.eval import evaluate_model

from config import PATH_DATA_DIR
from config import KEY_LOC, KEY_YEAR, KEY_TARGET
from data_preparation.data_alignment import load_data_csv, merge_data, set_indices


def get_model_predictions(model, sel_loc, sel_year):
    test_data = {
        KEY_LOC: sel_loc,
        KEY_YEAR: sel_year,
    }

    return model.predict_item(test_data)


def test_average_yield_model():
    model = AverageYieldModel(group_cols=[KEY_LOC])
    data_path = os.path.join(PATH_DATA_DIR, "data_US", "county_data")
    yield_csv = os.path.join(data_path, "YIELD_COUNTY_US.csv")
    yield_df = pd.read_csv(yield_csv, index_col=[KEY_LOC, KEY_YEAR])
    dataset = Dataset(data_target=yield_df, data_features=[])
    model.fit(dataset)

    # test prediction for an existing item
    sel_loc = "US-01-001"  # "AL_AUTAUGA"
    sel_year = 2018
    assert sel_loc in yield_df.index.get_level_values(0)
    filtered_df = yield_df[yield_df.index.get_level_values(0) == sel_loc]
    expected_pred = filtered_df[KEY_TARGET].mean()
    test_preds, _ = get_model_predictions(model, sel_loc, sel_year)
    assert np.round(test_preds[0], 2) == np.round(expected_pred, 2)

    # test prediction for a non-existent item
    sel_loc = "US-06-081"  # "CA_SAN_MATEO"
    assert sel_loc not in yield_df.index.get_level_values(0)
    expected_pred = yield_df[KEY_TARGET].mean()
    test_preds, _ = get_model_predictions(model, sel_loc, sel_year)
    assert np.round(test_preds[0], 2) == np.round(expected_pred, 2)


def test_sklearn_model():
    data_path = os.path.join(PATH_DATA_DIR, "data_US", "county_features")
    # Training dataset
    train_csv = os.path.join(data_path, "grain_maize_US_train.csv")
    train_df = pd.read_csv(train_csv, index_col=[KEY_LOC, KEY_YEAR])
    train_yields = train_df[[KEY_TARGET]].copy()
    feature_cols = [c for c in train_df.columns if c != KEY_TARGET]
    train_features = train_df[feature_cols].copy()
    train_dataset = Dataset(train_yields, [train_features])

    # Test dataset
    test_csv = os.path.join(data_path, "grain_maize_US_train.csv")
    test_df = pd.read_csv(test_csv, index_col=[KEY_LOC, KEY_YEAR])
    test_yields = test_df[[KEY_TARGET]].copy()
    test_features = test_df[feature_cols].copy()
    test_dataset = Dataset(test_yields, [test_features])

    # Model
    ridge = Ridge(alpha=0.5)
    model = SklearnModel(
        ridge,
        feature_cols=feature_cols,
    )
    model.fit(train_dataset)

    test_preds, _ = model.predict(test_dataset)

    assert test_preds.shape[0] == len(test_dataset)

    evaluation_result = evaluate_model(model, test_dataset)
    expected_values = {
        "normalized_rmse": 14.49,
        "mape": 0.14,
    }
    for metric, expected_value in expected_values.items():
        assert (
            metric in evaluation_result
        ), f"Metric '{metric}' not found in evaluation result"
        assert (
            round(evaluation_result[metric], 2) == expected_value
        ), f"Value of metric '{metric}' does not match expected value"

    # Model with hyperparameter optimization
    fit_params = {
        "optimize_hyperparameters": True,
        "param_space": {"estimator__alpha": [0.01, 0.1, 0.0, 1.0, 5.0, 10.0]},
    }
    model.fit(train_dataset, **fit_params)
    test_preds, _ = model.predict(test_dataset)
    assert test_preds.shape[0] == len(test_dataset)

def test_nn_model():

    train_dataset = Dataset.load("test_maize_us")
    test_dataset = Dataset.load("test_maize_us")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model, assumes that all features are in np.ndarray format
    n_total_features = len(train_dataset[0].keys()) - 3
    ts_features = [key for key in train_dataset[0].keys() if type(train_dataset[0][key]) == np.ndarray]
    ts_features = [key for key in ts_features if len(train_dataset[0][key].shape) == 1]
   
    model = ExampleLSTM(len(ts_features), n_total_features - len(ts_features), hidden_size=64, num_layers=2, output_size=1)

    # Train model
    model.fit(train_dataset, batch_size=3200, num_epochs=10, device=device)

    test_preds, _ = model.predict(test_dataset)
    assert test_preds.shape[0] == len(test_dataset)

    # Check if evaluation results are within expected range
    evaluation_result = evaluate_model(model, test_dataset)
    print(evaluation_result)

    min_expected_values = {"normalized_rmse": 5, "mape": 0.05,}
    for metric, expected_value in min_expected_values.items():
        assert (
            metric in evaluation_result
        ), f"Metric '{metric}' not found in evaluation result"
        assert (
            evaluation_result[metric] >= expected_value
        ), f"Value of metric '{metric}' does not match expected value"

    max_expected_values = {"normalized_rmse": 60, "mape": 0.60,}
    for metric, expected_value in max_expected_values.items():
        assert (
            metric in evaluation_result
        ), f"Metric '{metric}' not found in evaluation result"
        assert (
            evaluation_result[metric] <= expected_value
        ), f"Value of metric '{metric}' does not match expected value"

if __name__ == "__main__":
    #test_average_yield_model()
    #test_sklearn_model()
    test_nn_model()
    print("All tests passed!")

