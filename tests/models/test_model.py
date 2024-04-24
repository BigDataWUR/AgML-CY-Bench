import os
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge

from datasets.dataset import Dataset
from models.naive_models import AverageYieldModel
from models.trend_model import TrendModel
from models.sklearn_model import SklearnModel
from util.data import get_trend_features
from evaluation.eval import evaluate_model

from config import PATH_DATA_DIR
from config import KEY_LOC, KEY_YEAR, KEY_TARGET


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
    test_data = {
        KEY_LOC: sel_loc,
        KEY_YEAR: sel_year,
    }

    test_preds, _ = model.predict_item(test_data)
    assert np.round(test_preds[0], 2) == np.round(expected_pred, 2)

    # test prediction for a non-existent item
    sel_loc = "US-06-081"  # "CA_SAN_MATEO"
    assert sel_loc not in yield_df.index.get_level_values(0)
    expected_pred = yield_df[KEY_TARGET].mean()
    test_data[KEY_LOC] = sel_loc
    test_preds, _ = test_preds, _ = model.predict_item(test_data)
    assert np.round(test_preds[0], 2) == np.round(expected_pred, 2)


def test_trend_model():
    trend_window = 5
    data_path = os.path.join(PATH_DATA_DIR, "data_US", "county_data")
    yield_csv = os.path.join(data_path, "YIELD_COUNTY_US.csv")
    yield_df = pd.read_csv(yield_csv, header=0)
    trend_fts, x_cols, y_cols = get_trend_features(
        yield_df, KEY_LOC, KEY_YEAR, KEY_TARGET, trend_window
    )
    model = TrendModel(x_cols, y_cols)

    # align data
    yield_df = yield_df.merge(trend_fts[[KEY_LOC, KEY_YEAR]], on=[KEY_LOC, KEY_YEAR])
    yield_df = yield_df.set_index([KEY_LOC, KEY_YEAR])
    trend_fts = trend_fts.set_index([KEY_LOC, KEY_YEAR])

    # create dataset
    dataset = Dataset(yield_df, [trend_fts])
    model.fit(dataset)

    # dummy test data
    dummy_x_vals = list(range(2016, 2021))
    dummy_y_vals = list(range(2, 7))
    sel_loc = "US-01-001"  # "AL_AUTAUGA"
    sel_year = dummy_x_vals[-1] + 1
    assert sel_loc in yield_df.index.get_level_values(0)

    test_data = {KEY_LOC: sel_loc, KEY_YEAR: sel_year}

    for i, c in enumerate(x_cols):
        test_data[c] = dummy_x_vals[i]

    for i, c in enumerate(y_cols):
        test_data[c] = dummy_y_vals[i]

    # in dummy test data, yield increases by 1 every year
    expected_pred = dummy_y_vals[-1] + 1
    test_preds, _ = model.predict_item(test_data)
    assert np.round(test_preds[0], 2) == np.round(expected_pred, 2)

    # Test Average Trend (average of values in trend window)
    model = TrendModel(x_cols, y_cols, trend_est="average")
    model.fit(dataset)
    # prediction should be average of dummy data
    expected_pred = np.mean(dummy_y_vals)
    test_preds, _ = model.predict_item(test_data)
    assert np.round(test_preds[0], 2) == np.round(expected_pred, 2)

    # Test Quadratic Trend
    model = TrendModel(x_cols, y_cols, trend_est="quadratic")
    model.fit(dataset)
    # NOTE: Quadratic model adds a quadratic term in addition to a linear term.
    # For our dummy data, coefficient of quadratic term must be zero.
    expected_pred = dummy_y_vals[-1] + 1
    test_preds, _ = model.predict_item(test_data)
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
