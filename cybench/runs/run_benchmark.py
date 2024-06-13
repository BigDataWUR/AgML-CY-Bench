import os
from collections import defaultdict

import pandas as pd
import torch
from datetime import datetime
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

import cybench.config
from cybench.config import DATASETS, PATH_DATA_DIR, PATH_RESULTS_DIR
from cybench.models.model import BaseModel

from cybench.datasets.dataset import Dataset

from cybench.evaluation.eval import evaluate_model, evaluate_predictions

from cybench.models.naive_models import AverageYieldModel
from cybench.models.trend_model import TrendModel
from cybench.models.sklearn_model import SklearnModel
from cybench.models.nn_models import ExampleLSTM


_BASELINE_MODEL_CONSTRUCTORS = {
    "AverageYieldModel": AverageYieldModel,
    # "LinearTrend": TrendModel, # TODO: Uncomment after fixing errors.
    "SklearnRidge": SklearnModel,
    "SklearnRF": SklearnModel,
    "LSTM": ExampleLSTM,
}

sklearn_ridge = Ridge(alpha=0.5)
sklearn_rf = RandomForestRegressor(oob_score=True, n_estimators=100, min_samples_leaf=5)
BASELINE_MODELS = list(_BASELINE_MODEL_CONSTRUCTORS.keys())

_BASELINE_MODEL_INIT_KWARGS = defaultdict(dict)
_BASELINE_MODEL_INIT_KWARGS["LinearTrend"] = {"trend": "linear"}

_BASELINE_MODEL_INIT_KWARGS["SklearnRidge"] = {"sklearn_est": sklearn_ridge}
_BASELINE_MODEL_INIT_KWARGS["SklearnRF"] = {"sklearn_est": sklearn_rf}

_BASELINE_MODEL_INIT_KWARGS["LSTM"] = {
    "hidden_size": 64,
    "num_layers": 1,
}

_BASELINE_MODEL_FIT_KWARGS = defaultdict(dict)

_BASELINE_MODEL_FIT_KWARGS["SklearnRidge"] = {
    "optimize_hyperparameters": True,
    "param_space": {"estimator__alpha": [0.01, 0.1, 0.0, 1.0, 5.0, 10.0]},
}

_BASELINE_MODEL_FIT_KWARGS["LSTM"] = {
    "batch_size": 16,
    "num_epochs": 50,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "optim_fn": torch.optim.Adam,
    "optim_kwargs": {"lr": 0.0001, "weight_decay": 0.00001},
    "scheduler_fn": torch.optim.lr_scheduler.StepLR,
    "scheduler_kwargs": {"step_size": 1, "gamma": 1},
    "val_fraction": 0.1,
    "val_split_by_year": True,
    "do_early_stopping": True,
    "optimize_hyperparameters": False,
    "param_space": {
        "optim_kwargs": {
            "lr": [0.0001, 0.00001],
            "weight_decay": [0.0001, 0.00001],
        },
    },
    "do_kfold": False,
    "kfolds": 5,
}


def run_benchmark(
    run_name: str,
    model_name: str = None,
    model_constructor: callable = None,
    model_init_kwargs: dict = None,
    model_fit_kwargs: dict = None,
    baseline_models: list = None,
    dataset_name: str = "maize_NL",
) -> dict:
    """
    Run the AgML benchmark.
    Args:
        run_name (str): The name of the run. Will be used to store log files and model results
        model_name (str): The name of the model. Will be used to store log files and model results
        model_constructor (Callable): The constructor of the model. Will be used to construct the model
        model_init_kwargs (dict): The kwargs used when constructing the model.
        model_fit_kwargs (dict): The kwargs used to fit the model.
        baseline_models (list): A list of names of baseline models to run next to the provided model.
                                If unspecified, a default list of baseline models will be used.
        dataset_name (str): The name of the dataset to load
    Returns:
        a dictionary containing the results of the benchmark
    """
    baseline_models = baseline_models or BASELINE_MODELS
    assert all([name in BASELINE_MODELS for name in baseline_models])

    model_init_kwargs = model_init_kwargs or dict()
    model_fit_kwargs = model_fit_kwargs or dict()

    # Create a directory to store model output

    path_results = os.path.join(PATH_RESULTS_DIR, run_name)
    os.makedirs(path_results, exist_ok=True)

    # Make sure model_name is not already defined
    assert (
        model_name not in BASELINE_MODELS
    ), f"Model name {model_name} already occurs in the baseline"

    model_constructors = {
        **_BASELINE_MODEL_CONSTRUCTORS,
    }

    models_init_kwargs = defaultdict(dict)
    for name, kwargs in _BASELINE_MODEL_INIT_KWARGS.items():
        models_init_kwargs[name] = kwargs

    models_fit_kwargs = defaultdict(dict)
    for name, kwargs in _BASELINE_MODEL_FIT_KWARGS.items():
        models_fit_kwargs[name] = kwargs

    if model_name is not None:
        assert model_constructor is not None
        model_constructors[model_name] = model_constructor
        models_init_kwargs[model_name] = model_init_kwargs
        models_fit_kwargs[model_name] = model_fit_kwargs

    dataset = Dataset.load(dataset_name)

    all_years = dataset.years
    for test_year in all_years:
        train_years = [y for y in all_years if y != test_year]
        test_years = [test_year]
        train_dataset, test_dataset = dataset.split_on_years((train_years, test_years))

        labels = test_dataset.targets()

        model_output = {
            cybench.config.KEY_LOC: [loc_id for loc_id, _ in test_dataset.indices()],
            cybench.config.KEY_YEAR: [year for _, year in test_dataset.indices()],
            "targets": labels,
        }

        compiled_results = {}
        for model_name, model_constructor in model_constructors.items():
            model = model_constructor(**models_init_kwargs[model_name])
            model.fit(train_dataset, **models_fit_kwargs[model_name])
            predictions, _ = model.predict(test_dataset)
            # save predictions
            results = evaluate_predictions(labels, predictions)
            compiled_results[model_name] = results

            model_output[model_name] = predictions

        df = pd.DataFrame.from_dict(model_output)
        df.set_index([cybench.config.KEY_LOC, cybench.config.KEY_YEAR], inplace=True)
        df.to_csv(os.path.join(path_results, f"{dataset_name}_year_{test_year}.csv"))

    df_metrics = _compute_evaluation_results(run_name)

    return {
        "df_metrics": df_metrics,
    }


def _compute_evaluation_results(
    run_name: str,
) -> pd.DataFrame:
    path_results = os.path.join(PATH_RESULTS_DIR, run_name)

    files = [
        f
        for f in os.listdir(path_results)
        if os.path.isfile(os.path.join(path_results, f))
    ]

    # No files, return an empty data frame
    if (not files):
        return pd.DataFrame(columns=["model", "year", "metric", "value"])

    rows = []

    for file in files:
        path = os.path.join(path_results, file)

        df = pd.read_csv(path)

        df.set_index([cybench.config.KEY_LOC, cybench.config.KEY_YEAR], inplace=True)

        years = set(df.index.get_level_values(cybench.config.KEY_YEAR))
        assert len(years) == 1  # Every fold is assumed to contain only one year
        year = list(years)[0]

        columns = df.columns

        y_true = df[[columns[0]]].values

        for model_name in columns[1:]:
            y_pred = df[[model_name]].values

            metrics = evaluate_predictions(y_true, y_pred)

            for metric_name, value in metrics.items():
                rows.append(
                    {
                        "model": model_name,
                        "year": year,
                        "metric": metric_name,
                        "value": value,
                    }
                )

    df_all = pd.DataFrame(rows)

    return df_all


def run_benchmark_on_all_data():
    for crop in DATASETS:
        for cn in DATASETS[crop]:
            if os.path.exists(os.path.join(PATH_DATA_DIR, crop, cn)):
                dataset_name = crop + "_" + cn
                run_name = datetime.now().strftime(f"{dataset_name}_%H_%M_%d_%m_%Y.run")
                run_benchmark(run_name=run_name, dataset_name=dataset_name)
