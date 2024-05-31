import os

"""Import comet before torch to track metrics automatically"""
import comet_ml
import comet_ml.integration.pytorch
from comet_ml import Experiment

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from models.model import BaseModel
from models.sklearn_model import SklearnModel
from models.trend_model import TrendModel
from models.naive_models import AverageYieldModel
from models.nn_models import BaseNNModel

from runs.run_benchmark import run_benchmark
from datasets.dataset import Dataset
from datasets.dataset_torch import TorchDataset
from models.nn_models import ExampleLSTM
from evaluation.eval import evaluate_model
from evaluation.log_experiments import (
    comet_wrapper,
    log_to_comet_post_hoc,
    log_benchmark_to_comet,
)

from config import PATH_DATA_DIR
from config import KEY_LOC, KEY_YEAR, KEY_TARGET


def example_for_logging_torch_model(comet_experiment=None):
    """
    Example of logging metrics in Comet anonymously for torch models
    """

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

    """Wrap model and log other metrics to comet"""
    comet_experiment = comet_wrapper(model=model, comet_experiment=comet_experiment)

    fit_kwargs = {
        "batch_size": 3200,
        "num_epochs": 15,
        "device": device,
        "scheduler_fn": scheduler_fn,
        "scheduler_kwargs": scheduler_kwargs,
    }
    optim_kwargs = {"lr": 0.01}
    model.fit(train_dataset, **fit_kwargs, optim_kwargs=optim_kwargs)

    evaluation_result = evaluate_model(model, test_dataset)

    """End of training and logging of results and the model itself"""

    log_to_comet_post_hoc(
        metrics=evaluation_result,
        # combine dicts of some defined parameter kwargs
        params=fit_kwargs | optim_kwargs | scheduler_kwargs,
        comet_experiment=comet_experiment,
        model=model,
        name=f"ExampleLSTM-Model",
    )


def example_for_logging_sklearn_model(comet_experiment=None, end=False):
    """
    Example of logging metrics in Comet anonymously for sklearn models
    """
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
    fit_params = {
        "optimize_hyperparameters": True,
        "param_space": {"estimator__alpha": [0.01, 0.1, 0.0, 1.0, 5.0, 10.0]},
    }
    model.fit(train_dataset, **fit_params)

    evaluation_result = evaluate_model(model, test_dataset)

    """Convert some important params as dicts"""
    feature_cols_dict = {"feature_cols": feature_cols}

    """Log post hoc only, making sure all metrics and params are passed"""
    log_to_comet_post_hoc(
        comet_experiment=comet_experiment,
        metrics=evaluation_result,
        params=feature_cols_dict | fit_params,
        model=model,
        end=end,
    )


def example_run_benchmark(comet_experiment=None):
    """
    Example of logging metrics in Comet anonymously for a benchmark run with an LSTM model
    """
    model_init_kwargs = {
        "n_ts_features": 9,
        "n_static_features": 1,
        "hidden_size": 8,
        "num_layers": 1,
    }
    model_fit_kwargs = {"num_epochs": 3}

    model_name = "ShallowLSTM"

    run_name = "shallow_1"

    results = run_benchmark(
        run_name,
        model_name,
        ExampleLSTM,
        model_init_kwargs=model_init_kwargs,
        model_fit_kwargs=model_fit_kwargs,
    )

    log_benchmark_to_comet(
        results,
        model_name,
        run_name,
        comet_experiment=comet_experiment,
        params=model_init_kwargs | model_fit_kwargs,
        end=True,
    )


if __name__ == "__main__":
    example_run_benchmark()
