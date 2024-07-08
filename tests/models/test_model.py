import os
import torch
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge

from cybench.datasets.dataset import Dataset
from cybench.datasets.dataset_torch import TorchDataset
from cybench.models.naive_models import AverageYieldModel
from cybench.models.trend_model import TrendModel
from cybench.models.sklearn_model import SklearnModel
from cybench.models.nn_models import ExampleLSTM
from cybench.evaluation.eval import evaluate_model

from cybench.config import PATH_DATA_DIR
from cybench.config import KEY_LOC, KEY_YEAR, KEY_TARGET


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
    dataset = Dataset("maize", data_target=yield_df, data_inputs=[])
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
    dataset = Dataset("maize", data_target=yield_df, data_inputs=[])
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
        ["US-01-001", 2000, 4.1],
        ["US-01-001", 2001, 4.2],
        ["US-01-001", 2002, 4.3],
        ["US-01-001", 2003, 4.4],
        ["US-01-001", 2004, 4.5],
        ["US-01-001", 2005, 4.6],
        ["US-01-001", 2006, 4.7],
        ["US-01-001", 2007, 4.8],
        ["US-01-001", 2008, 4.9],
        ["US-01-001", 2009, 5.0],
        ["US-01-002", 2000, 5.1],
        ["US-01-002", 2001, 5.2],
        ["US-01-002", 2002, 5.3],
        ["US-01-002", 2003, 5.4],
        ["US-01-002", 2004, 5.5],
        ["US-01-002", 2005, 5.6],
        ["US-01-002", 2006, 5.7],
        ["US-01-002", 2007, 5.8],
        ["US-01-002", 2008, 5.9],
        ["US-01-002", 2009, 6.0],
        ["US-01-003", 2000, 7.0],
        ["US-01-003", 2001, 8.0],
        ["US-01-003", 2003, 9.0],
        ["US-01-004", 2000, 8.0],
        ["US-01-004", 2001, 7.0],
        ["US-01-004", 2003, 9.0],
    ]
    yield_df = pd.DataFrame(dummy_data, columns=[KEY_LOC, KEY_YEAR, KEY_TARGET])

    for sel_loc in yield_df[KEY_LOC].unique():
        yield_loc_df = yield_df[yield_df[KEY_LOC] == sel_loc]
        all_years = sorted(yield_loc_df[KEY_YEAR].unique())
    
        if (sel_loc in ["US-01-001", "US-01-002"]):
            test_indexes = [0, 2, len(all_years) - 1]
            for idx in test_indexes:
                test_year = all_years[idx]
                train_years = [y for y in all_years if y != test_year]
                train_yields = yield_loc_df[yield_loc_df[KEY_YEAR].isin(train_years)]
                train_yields = train_yields.set_index([KEY_LOC, KEY_YEAR])
                test_yields = yield_loc_df[yield_loc_df[KEY_YEAR] == test_year]
                train_dataset = Dataset("maize", train_yields, [])

                # linear trend
                model = TrendModel(trend="linear")
                model.fit(train_dataset)
                test_data = {
                    KEY_LOC: sel_loc,
                    KEY_YEAR: test_year,
                }
                test_preds, _ = model.predict_item(test_data)
                expected_pred = test_yields[KEY_TARGET].values[0]
                assert np.round(test_preds[0], 2) == np.round(expected_pred, 2)

                # quadratic trend ( trend = c + a x + b x^2)
                model = TrendModel(trend="quadratic")
                model.fit(train_dataset)
                test_preds, _ = model.predict_item(test_data)
                expected_pred = test_yields[KEY_TARGET].values[0]
                assert np.round(test_preds[0], 2) == np.round(expected_pred, 2)
        else:
            test_year = all_years[-1]
            train_years = [y for y in all_years if y != test_year]
            train_yields = yield_loc_df[yield_loc_df[KEY_YEAR].isin(train_years)]
            train_yields = train_yields.set_index([KEY_LOC, KEY_YEAR])
            train_dataset = Dataset("maize", train_yields, [])

            # Expect the average due to insufficient data or no trend
            model = TrendModel(trend="linear")
            model.fit(train_dataset)
            test_data = {
                KEY_LOC: sel_loc,
                KEY_YEAR: test_year,
            }
            test_preds, _ = model.predict_item(test_data)
            expected_pred = train_yields[KEY_TARGET].mean()
            assert np.round(test_preds[0], 2) == np.round(expected_pred, 2)


def test_sklearn_model():
    # Test 1: Test with raw data
    dataset_sw_nl = Dataset.load("wheat_NL")
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

    # Test 2: Test with predesigned features
    data_path = os.path.join(PATH_DATA_DIR, "features", "maize", "US")
    # Training dataset
    train_csv = os.path.join(data_path, "grain_maize_US_train.csv")
    train_df = pd.read_csv(train_csv, index_col=[KEY_LOC, KEY_YEAR])
    train_yields = train_df[[KEY_TARGET]].copy()
    feature_cols = [c for c in train_df.columns if c != KEY_TARGET]
    train_features = train_df[feature_cols].copy()
    train_dataset = Dataset("maize", train_yields, [train_features])

    # Test dataset
    test_csv = os.path.join(data_path, "grain_maize_US_train.csv")
    test_df = pd.read_csv(test_csv, index_col=[KEY_LOC, KEY_YEAR])
    test_yields = test_df[[KEY_TARGET]].copy()
    test_features = test_df[feature_cols].copy()
    test_dataset = Dataset("maize", test_yields, [test_features])

    # Model
    ridge = Ridge(alpha=0.5)
    model = SklearnModel(
        ridge,
        feature_cols=feature_cols,
    )
    model.fit(train_dataset, **{"predesigned_features": True})

    test_preds, _ = model.predict(test_dataset)
    assert test_preds.shape[0] == len(test_dataset)

    # TODO: Need alternative to hardcoding expected metrics.
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

    # Test 3: Test hyperparameter optimization
    fit_params = {
        "optimize_hyperparameters": True,
        "param_space": {"estimator__alpha": [0.01, 0.1, 0.0, 1.0, 5.0, 10.0]},
        "predesigned_features": True,
    }
    model.fit(train_dataset, **fit_params)
    test_preds, _ = model.predict(test_dataset)
    assert test_preds.shape[0] == len(test_dataset)


# TODO: Uncomment after TorchDataset and NN models handle
# different number of time steps for time series data.
# Number of time steps can vary between sources and within a source.
# Same goes for tests.datasets.test_transforms test_transforms()


def test_nn_model():
    train_dataset = Dataset.load("maize_ES")
    test_dataset = Dataset.load("maize_ES")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model, assumes that all features are in np.ndarray format
    model = ExampleLSTM(
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
        num_epochs=2,
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
