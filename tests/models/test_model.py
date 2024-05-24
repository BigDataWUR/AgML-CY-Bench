import os
import torch
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge

from datasets.dataset import Dataset
from datasets.dataset_torch import TorchDataset
from models.naive_models import AverageYieldModel
from models.trend_model import TrendModel
from models.sklearn_model import SklearnModel
from models.nn_models import ExampleLSTM
from evaluation.eval import evaluate_model

from config import PATH_DATA_DIR
from config import KEY_LOC, KEY_YEAR, KEY_TARGET


def test_average_yield_model():
    model = AverageYieldModel()
    dummy_data = [
        ["US-01-001", 2000, 5.0],
        ["US-01-001", 2001, 5.5],
        ["US-01-001", 2002, 6.0],
        ["US-01-001", 2003, 5.2],
        ["US-01-002", 2000, 7.0],
        ["US-01-002", 2001, 7.5],
        ["US-01-002", 2002, 6.2],
        ["US-01-002", 2003, 5.8],
    ]
    yield_df = pd.DataFrame(dummy_data, columns=[KEY_LOC, KEY_YEAR, KEY_TARGET])
    yield_df = yield_df.set_index([KEY_LOC, KEY_YEAR])

    # test prediction for an existing item
    sel_loc = "US-01-001"
    assert sel_loc in yield_df.index.get_level_values(0)
    dataset = Dataset(data_target=yield_df, data_inputs=[])
    model.fit(dataset)
    sel_year = 2018
    filtered_df = yield_df[yield_df.index.get_level_values(0) == sel_loc]
    expected_pred = filtered_df[KEY_TARGET].mean()
    test_data = {
        KEY_LOC: sel_loc,
        KEY_YEAR: sel_year,
    }

    test_preds, _ = model.predict_item(test_data)
    assert np.round(test_preds[0], 2) == np.round(expected_pred, 2)

    # test one more location
    sel_loc = "US-01-002"
    test_data[KEY_LOC] = sel_loc
    filtered_df = yield_df[yield_df.index.get_level_values(0) == sel_loc]
    expected_pred = filtered_df[KEY_TARGET].mean()
    test_preds, _ = model.predict_item(test_data)
    assert np.round(test_preds[0], 2) == np.round(expected_pred, 2)

    # test prediction for a non-existent item
    sel_loc = "US-01-003"
    assert sel_loc not in yield_df.index.get_level_values(0)
    dataset = Dataset(data_target=yield_df, data_inputs=[])
    model.fit(dataset)
    expected_pred = yield_df[KEY_TARGET].mean()
    test_data[KEY_LOC] = sel_loc
    test_preds, _ = model.predict_item(test_data)
    assert np.round(test_preds[0], 2) == np.round(expected_pred, 2)


def test_trend_model():
    """
    NOTE: quadratic trend will be the same as linear trend
    for the dummy data.
    """
    dummy_data = [
        ["US-01-001", 2000, 5.0],
        ["US-01-001", 2001, 6.0],
        ["US-01-001", 2002, 7.0],
        ["US-01-001", 2003, 8.0],
        ["US-01-002", 2000, 5.5],
        ["US-01-002", 2001, 6.5],
        ["US-01-002", 2002, 7.5],
        ["US-01-002", 2003, 8.5],
    ]
    yield_df = pd.DataFrame(dummy_data, columns=[KEY_LOC, KEY_YEAR, KEY_TARGET])
    all_years = sorted(yield_df[KEY_YEAR].unique())

    test_indexes = [0, 2, 3]
    for idx in test_indexes:
        test_year = all_years[idx]
        train_years = [y for y in all_years if y != test_year]
        sel_loc = "US-01-001"
        train_yields = yield_df[yield_df[KEY_YEAR].isin(train_years)]
        train_yields = train_yields.set_index([KEY_LOC, KEY_YEAR])
        test_yields = yield_df[yield_df[KEY_YEAR] == test_year]
        train_dataset = Dataset(train_yields, [])

        # linear trend
        model = TrendModel(trend="linear")
        model.fit(train_dataset)
        test_data = {
            KEY_LOC: sel_loc,
            KEY_YEAR: test_year,
        }
        test_preds, _ = model.predict_item(test_data)
        expected_pred = test_yields[test_yields[KEY_LOC] == sel_loc][KEY_TARGET].values[
            0
        ]
        assert np.round(test_preds[0], 2) == np.round(expected_pred, 2)

        sel_loc = "US-01-002"
        test_data[KEY_LOC] = sel_loc
        test_preds, _ = model.predict_item(test_data)
        expected_pred = test_yields[test_yields[KEY_LOC] == sel_loc][KEY_TARGET].values[
            0
        ]
        assert np.round(test_preds[0], 2) == np.round(expected_pred, 2)

        # quadratic trend ( trend = c + a x + b x^2)
        model = TrendModel(trend="quadratic")
        model.fit(train_dataset)
        sel_loc = "US-01-001"
        test_data[KEY_LOC] = sel_loc
        test_preds, _ = model.predict_item(test_data)
        expected_pred = test_yields[test_yields[KEY_LOC] == sel_loc][KEY_TARGET].values[
            0
        ]
        assert np.round(test_preds[0], 2) == np.round(expected_pred, 2)

        sel_loc = "US-01-002"
        test_data[KEY_LOC] = sel_loc
        test_preds, _ = model.predict_item(test_data)
        expected_pred = test_yields[test_yields[KEY_LOC] == sel_loc][KEY_TARGET].values[
            0
        ]
        assert np.round(test_preds[0], 2) == np.round(expected_pred, 2)


def test_sklearn_model():
    # Test with raw data
    dataset_sw_nl = Dataset.load("test_softwheat_nl")
    all_years = list(range(2001, 2019))
    test_years = [2017, 2018]
    train_years = [yr for yr in all_years if yr not in test_years]
    train_dataset, test_dataset = dataset_sw_nl.split_on_years(
        (train_years, test_years)
    )

    # Model
    ridge = Ridge(alpha=0.5)
    model = SklearnModel(ridge)
    model.fit(train_dataset)

    test_preds, _ = model.predict(test_dataset)
    assert test_preds.shape[0] == len(test_dataset)

    # Test with predesigned features
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
    model.fit(train_dataset, **{"predesigned_features": True})

    test_preds, _ = model.predict(test_dataset)
    assert test_preds.shape[0] == len(test_dataset)

    # TODO: Uncomment after evaluate_model calls predict() with dataset.
    # TODO: Need alternative to hardcoding expected metrics.
    # evaluation_result = evaluate_model(model, test_dataset)
    # expected_values = {
    #     "normalized_rmse": 14.49,
    #     "mape": 0.14,
    # }
    # for metric, expected_value in expected_values.items():
    #     assert (
    #         metric in evaluation_result
    #     ), f"Metric '{metric}' not found in evaluation result"
    #     assert (
    #         round(evaluation_result[metric], 2) == expected_value
    #     ), f"Value of metric '{metric}' does not match expected value"

    # Model with hyperparameter optimization
    fit_params = {
        "optimize_hyperparameters": True,
        "param_space": {"estimator__alpha": [0.01, 0.1, 0.0, 1.0, 5.0, 10.0]},
        "predesigned_features": True,
    }
    model.fit(train_dataset, **fit_params)
    test_preds, _ = model.predict(test_dataset)
    assert test_preds.shape[0] == len(test_dataset)


def test_nn_model():
    train_dataset = Dataset.load("test_maize_us")
    test_dataset = Dataset.load("test_maize_us")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model, assumes that all features are in np.ndarray format
    n_total_features = len(train_dataset[0].keys()) - 4
    ts_features = [
        key
        for key in train_dataset[0].keys()
        if type(train_dataset[0][key]) == np.ndarray
    ]
    ts_features = [key for key in ts_features if len(train_dataset[0][key].shape) == 1]

    model = ExampleLSTM(
        len(ts_features),
        n_total_features - len(ts_features),
        hidden_size=64,
        num_layers=2,
        output_size=1,
    )
    scheduler_fn = torch.optim.lr_scheduler.StepLR
    scheduler_kwargs = {"step_size": 2, "gamma": 0.5}

    # Train model
    model.fit(
        train_dataset,
        batch_size=3200,
        num_epochs=10,
        device=device,
        optim_kwargs={"lr": 0.01},
        scheduler_fn=scheduler_fn,
        scheduler_kwargs=scheduler_kwargs,
    )

    test_preds, _ = model.predict(test_dataset)
    assert test_preds.shape[0] == len(test_dataset)

    # Check if evaluation results are within expected range
    evaluation_result = evaluate_model(model, test_dataset)
    print(evaluation_result)

    min_expected_values = {
        "normalized_rmse": 0,
        "mape": 0.00,
    }
    for metric, expected_value in min_expected_values.items():
        assert (
            metric in evaluation_result
        ), f"Metric '{metric}' not found in evaluation result"
        assert (
            evaluation_result[metric] >= expected_value
        ), f"Value of metric '{metric}' does not match expected value"
        # Check metric is not NaN
        assert not np.isnan(
            evaluation_result[metric]
        ), f"Value of metric '{metric}' is NaN"
