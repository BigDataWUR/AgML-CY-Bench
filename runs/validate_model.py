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
}

BASELINE_MODELS = list(_BASELINE_MODEL_CONSTRUCTORS.keys())

_BASELINE_MODEL_INIT_KWARGS = defaultdict(dict)
_BASELINE_MODEL_FIT_KWARGS = defaultdict(dict)

def validate_single_model(
    run_name: str,
    model_name: str,
    model_constructor: callable,
    model_init_kwargs: dict = None,
    model_fit_kwargs: dict = None,
    baseline_models: list = None,
    dataset_name: str = "test_maize_us",
) -> dict:
    """
    Run a single model on a single fold and return validation results.
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
    test_year = list(all_years)[0]
    train_years = [y for y in all_years if y != test_year]
    test_years = [test_year]
    train_dataset, test_dataset = dataset.split_on_years((train_years, test_years))

    compiled_results = {}
    for model_name, model_constructor in model_constructors.items():
        model = model_constructor(**models_init_kwargs[model_name])
        _, train_dict = model.fit(train_dataset, **models_fit_kwargs[model_name])
        
        # Add model initialization and fitting kwargs to the results
        train_dict["model_init_kwargs"] = models_init_kwargs[model_name]
        train_dict["model_fit_kwargs"] = models_fit_kwargs[model_name]
        
        compiled_results[model_name] = train_dict

    df = pd.DataFrame.from_dict(compiled_results)
    return df

