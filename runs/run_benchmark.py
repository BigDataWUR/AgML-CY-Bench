import os
from collections import defaultdict

import pandas as pd
import torch

import config
from config import PATH_RESULTS_DIR
from models.model import BaseModel

from datasets.dataset import Dataset

from evaluation.eval import evaluate_model, evaluate_predictions

from models.naive_models import AverageYieldModel
from models.sklearn_model import SklearnModel
from models.nn_models import ExampleLSTM


_BASELINE_MODEL_CONSTRUCTORS = {
    "AverageYieldModel": AverageYieldModel,
    "LSTM": ExampleLSTM,
}

BASELINE_MODELS = list(_BASELINE_MODEL_CONSTRUCTORS.keys())

_BASELINE_MODEL_INIT_KWARGS = defaultdict(dict)
_BASELINE_MODEL_INIT_KWARGS["LSTM"] = {
    "n_ts_features": 9,
    "n_static_features": 1,
    "hidden_size": 32,
    "num_layers": 1,
}

_BASELINE_MODEL_FIT_KWARGS = defaultdict(dict)
_BASELINE_MODEL_FIT_KWARGS["LSTM"] = {
    'batch_size': 32,
    'num_epochs': 5,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'optim_fn': torch.optim.Adam,
    'scheduler_fn': torch.optim.lr_scheduler.StepLR,
    'scheduler_kwargs': {"step_size": 2, "gamma": 0.8},
    'val_fraction': 0.1,

    'optimize_hyperparameters': True,
    'param_space': {
        'optim_kwargs': {
            "lr": [0.01, 0.001],
            'weight_decay': [0.0001],
        },
    },
    'do_kfold': False,
    'kfolds': 5,
}


def run_benchmark(
    run_name: str,
    model_name: str,
    model_constructor: callable,
    model_init_kwargs: dict = None,
    model_fit_kwargs: dict = None,
    baseline_models: list = None,
    dataset_name: str = "test_maize_us",
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
        model_name: model_constructor,
    }

    models_init_kwargs = defaultdict(dict)
    models_init_kwargs[model_name] = model_init_kwargs
    for name, kwargs in _BASELINE_MODEL_INIT_KWARGS.items():
        models_init_kwargs[name] = kwargs

    models_fit_kwargs = defaultdict(dict)
    models_fit_kwargs[model_name] = model_fit_kwargs
    for name, kwargs in _BASELINE_MODEL_FIT_KWARGS.items():
        models_fit_kwargs[name] = kwargs

    dataset = Dataset.load(dataset_name)

    all_years = dataset.years
    for test_year in all_years:
        train_years = [y for y in all_years if y != test_year]
        test_years = [test_year]
        train_dataset, test_dataset = dataset.split_on_years((train_years, test_years))

        labels = test_dataset.targets()

        model_output = {
            config.KEY_LOC: [loc_id for loc_id, _ in test_dataset.indices()],
            config.KEY_YEAR: [year for _, year in test_dataset.indices()],
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
        df.set_index([config.KEY_LOC, config.KEY_YEAR], inplace=True)
        df.to_csv(os.path.join(path_results, f"year_{test_year}.csv"))

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

    rows = []

    for file in files:
        path = os.path.join(path_results, file)

        df = pd.read_csv(path)

        df.set_index([config.KEY_LOC, config.KEY_YEAR], inplace=True)

        years = set(df.index.get_level_values(config.KEY_YEAR))
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
    df_all.set_index(["model", "year", "metric"], inplace=True)

    return df_all
