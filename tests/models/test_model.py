import os
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge

from datasets.dataset import Dataset
from models.naive_models import AverageYieldModel
from models.trend_model import TrendModel
from models.sklearn_model import SklearnModel
from evaluation.eval import evaluate_model

from config import PATH_DATA_DIR
from config import KEY_LOC, KEY_YEAR, KEY_TARGET


def test_average_yield_model():
    model = AverageYieldModel()
    dummy_data = [["US-01-001", 2000, 5.0],
                  ["US-01-001", 2001, 5.5],
                  ["US-01-001", 2002, 6.0],
                  ["US-01-001", 2003, 5.2],
                  ["US-01-002", 2000, 7.0],
                  ["US-01-002", 2001, 7.5],
                  ["US-01-002", 2002, 6.2],
                  ["US-01-002", 2003, 5.8]]
    yield_df = pd.DataFrame(dummy_data,
                            columns=[KEY_LOC, KEY_YEAR, KEY_TARGET])

    # test prediction for an existing item
    sel_loc = "US-01-001"  # "AL_AUTAUGA"
    sel_yield_df = yield_df[yield_df[KEY_LOC] == sel_loc]
    assert not sel_yield_df.empty

    # set index
    sel_yield_df = sel_yield_df.set_index([KEY_LOC, KEY_YEAR])
    dataset = Dataset(data_target=sel_yield_df, data_features=[])
    model.fit(dataset)
    sel_year = 2018
    expected_pred = sel_yield_df[KEY_TARGET].mean()
    test_data = {
        KEY_LOC: sel_loc,
        KEY_YEAR: sel_year,
    }

    test_preds, _ = model.predict_item(test_data)
    assert np.round(test_preds[0], 2) == np.round(expected_pred, 2)

    # test prediction for a non-existent item
    sel_loc = "US-06-081"  # "CA_SAN_MATEO"
    assert sel_loc not in yield_df.index.get_level_values(0)

    # set index
    yield_df = yield_df.set_index([KEY_LOC, KEY_YEAR])
    dataset = Dataset(data_target=yield_df, data_features=[])
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
    dummy_data = [["US-01-001", 2000, 5.0],
                  ["US-01-001", 2001, 6.0],
                  ["US-01-001", 2002, 7.0],
                  ["US-01-001", 2003, 8.0],
                  ["US-01-002", 2000, 5.5],
                  ["US-01-002", 2001, 6.5],
                  ["US-01-002", 2002, 7.5],
                  ["US-01-002", 2003, 8.5]]
    yield_df = pd.DataFrame(dummy_data, columns=[KEY_LOC, KEY_YEAR, KEY_TARGET])
    all_years = sorted(yield_df[KEY_YEAR].unique())

    test_indexes = [0, 2, 3]
    for idx in test_indexes:
        test_year = all_years[idx]
        train_years = [y for y in all_years if y != test_year]
        sel_loc = "US-01-001"  # "AL_AUTAUGA"
        train_yields = yield_df[(yield_df[KEY_LOC] == sel_loc) &
                                yield_df[KEY_YEAR].isin(train_years)]
        train_yields = train_yields.set_index([KEY_LOC, KEY_YEAR])
        test_yields = yield_df[(yield_df[KEY_LOC] == sel_loc) &
                               (yield_df[KEY_YEAR] == test_year)]
        expected_pred = test_yields[KEY_TARGET].values[0]
        train_dataset = Dataset(train_yields, [])

        # linear trend
        model = TrendModel(trend="linear")
        model.fit(train_dataset)
        test_data = {
            KEY_LOC: sel_loc,
            KEY_YEAR: test_year,
        }
        test_preds, _ = model.predict_item(test_data)
        assert np.round(test_preds[0], 2) == np.round(expected_pred, 2)

        # quadratic trend ( trend = c + a x + b x^2)
        model = TrendModel(trend="quadratic")
        model.fit(train_dataset)
        test_data = {
            KEY_LOC: sel_loc,
            KEY_YEAR: test_year,
        }
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
