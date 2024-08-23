import os
from collections import defaultdict

import pandas as pd
import torch

import cybench.config
from cybench.config import PATH_RESULTS_DIR
from cybench.models.model import BaseModel

from cybench.datasets.dataset import Dataset

from cybench.evaluation.eval import evaluate_model, evaluate_predictions

from cybench.models.naive_models import AverageYieldModel
from cybench.models.sklearn_models import SklearnRidge
from cybench.models.nn_models import BaseLSTM


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
    test_years_to_leave_out: list = None,
) -> dict:
    """
    Run a single model on a single outer fold and return validation results.
    Test is is left out completely and not used for training or validation.
    Not used for benchmarking. Use run_benchmark instead. Hyperparameters should be optimized in each outer fold in the benchmark.
    This function should only be used for exploration of initial hyperparameter settings.

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

    if test_years_to_leave_out is None:
        test_years = [list(all_years)[0]]
    else:
        assert all([y in all_years for y in test_years_to_leave_out])
        test_years = test_years_to_leave_out

    train_years = [y for y in all_years if y not in test_years]
    assert len(set(train_years).intersection(set(test_years))) == 0

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
