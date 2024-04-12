import os
import torch as pt
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge

from datasets.dataset import Dataset
from datasets.dataset_torch import TorchDataset
from models.naive_models import AverageYieldModel
from models.sklearn_model import SklearnModel
from models.nn_models import ExampleLSTM
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
    sel_loc = "US-01-001" # "AL_AUTAUGA"
    sel_year = 2018
    assert sel_loc in yield_df.index.get_level_values(0)
    filtered_df = yield_df[yield_df.index.get_level_values(0) == sel_loc]
    expected_pred = filtered_df[KEY_TARGET].mean()
    test_preds, _ = get_model_predictions(model, sel_loc, sel_year)
    assert np.round(test_preds[0], 2) == np.round(expected_pred, 2)

    # test prediction for a non-existent item
    sel_loc = "US-06-081" # "CA_SAN_MATEO"
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

    # Model with hyperparameter optimization
    fit_params = {
        "optimize_hyperparameters": True,
        "param_space": {"estimator__alpha": [0.01, 0.1, 0.0, 1.0, 5.0, 10.0]},
    }
    model.fit(train_dataset, **fit_params)
    test_preds, _ = model.predict(test_dataset)
    assert test_preds.shape[0] == len(test_dataset)

def test_nn_model():
    data_sources = {
            "YIELD" : {
                "filename" : "YIELD_COUNTY_US.csv",
                "index_cols" : [KEY_LOC, KEY_YEAR],
                "sel_cols" : [KEY_TARGET]
            },
            "SOIL" : {
                "filename" : "SOIL_COUNTY_US.csv",
                "index_cols" : [KEY_LOC],
                "sel_cols" : ["sm_whc"]
            },
            "REMOTE_SENSING" : {
                "filename" : "REMOTE_SENSING_COUNTY_US.csv",
                "index_cols" : [KEY_LOC, KEY_YEAR, "dekad"],
                "sel_cols" : ["fapar"]
            }
        }
    
    device = pt.device("cuda" if pt.cuda.is_available() else "cpu")

    data_path = os.path.join(PATH_DATA_DIR, "data_US", "county_data")
    data_dfs = load_data_csv(data_path, data_sources)
    label_df = data_dfs["YIELD"]

    feature_dfs = {
        ft_key : data_dfs[ft_key] for ft_key in data_dfs if ft_key != "YIELD"
    }

    label_df, feature_dfs = merge_data(data_sources, label_df, feature_dfs)
    label_df, feature_dfs = set_indices(data_sources, label_df, feature_dfs)

    # Sort the indices
    label_df.sort_index(inplace=True)
    for src in feature_dfs:
        feature_dfs[src].sort_index(inplace=True)

    # Convert dict of dataframes to list of dataframes, throw away the keys
    feature_dfs = list(feature_dfs.values())

    train_dataset = TorchDataset(Dataset(label_df, feature_dfs))
    test_dataset = TorchDataset(Dataset(label_df, feature_dfs))

    # Initialize model
    time_series_features = [key for key in dataset[0].keys() if type(dataset[0][key]) == pt.Tensor]
    time_series_features = [key for key in time_series_features if dataset[0][key].dim() > 0]
    model = ExampleLSTM(input_size=len(time_series_features), hidden_size=64, num_layers=2, output_size=1)

    # Train model
    model.fit(dataset, batch_size=3200, num_epochs=1, device=device)

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


