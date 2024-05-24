import os
import comet_ml
import comet_ml.integration.pytorch
import pandas as pd
from comet_ml import Experiment

from models.model import BaseModel
from models.sklearn_model import SklearnModel
from models.trend_model import TrendModel
from models.naive_models import AverageYieldModel
from models.nn_models import BaseNNModel
from runs.run_benchmark import _compute_evaluation_results

from config import PATH_RESULTS_DIR

# get paths
evaluation_path = os.path.dirname(os.path.realpath(__file__))
root_path = evaluation_path[
            :-10
            ]  # remove "evaluation" in the string to get the root path

# Configure Comet experiment instance
_project_name = "AgML-Crop-yield-forecasting"


# get api key
def get_comet_api_key(file=None) -> str:
    """
    Function that returns api_key.

    Store the api key under evaluation/api_keys/...

    :param: file: file name that contains one line of the comet api key.
    """

    api_key = None

    check_api_key_dir = os.path.isdir(os.path.join(evaluation_path, 'api_keys'))
    file_name = 'comet_ml' if not file else file
    check_api_key_file = os.path.isdir(
        os.path.join(evaluation_path, 'api_keys', file_name)
    )

    if check_api_key_dir and check_api_key_file:
        with open(os.path.join(evaluation_path, "api_keys", "comet_ml"), "r") as f:
            api_key = f.readline()
    return api_key


def existing_comet(
        comet_experiment: Experiment,
        comet_api_key: str = None,
        project_name: str = None,
        workspace: str = None
) -> Experiment:
    """
    If api_key is not defined, comet_ml will start in anonymous mode and the logs can be accessed
        through the URL provided in the console

    :return: comet Experiment object
    """

    if comet_api_key is None:
        comet_api_key = get_comet_api_key()

    if comet_experiment is None:
        if comet_api_key is not None:
            experiment = Experiment(
                api_key=comet_api_key,
                project_name=_project_name if project_name is None else project_name,
                workspace=workspace,
                log_env_gpu=True,
                auto_log_co2=True,
                auto_metric_logging=True,
                auto_param_logging=True,
                auto_histogram_gradient_logging=True,
            )
        else:
            comet_ml.init(anonymous=True)
            experiment = Experiment(
                project_name=_project_name if project_name is None else project_name,
                workspace=workspace,
                log_env_gpu=True,
                auto_log_co2=True,
                auto_metric_logging=True,
                auto_param_logging=True,
                auto_histogram_gradient_logging=True,
            )
    else:
        experiment = comet_experiment

    return experiment


def log_to_comet_post_hoc(
        metrics: dict,
        params: dict,
        comet_experiment: Experiment = None,
        comet_api_key: str = None,
        name: str = None,
        model: BaseModel = None,
        asset_path: str = None,
        end: bool = False
) -> None:
    """
    Log metrics, params, asset and model to Comet_ml

    :param comet_experiment: comet experiment object
    :param metrics: dict of metrics
    :param params: dict of params/hyperparams
    :param model: name of the saved model
    :param name: name of the comet experiment
    :param asset_path: path to asset. To log a custom asset (e.g., config file etc.) to Comet.
    :return: None
    """

    experiment = existing_comet(comet_experiment=comet_experiment, comet_api_key=comet_api_key)
    assert isinstance(metrics, dict)
    experiment.log_parameters(params)
    experiment.log_metrics(metrics)
    experiment.log_code(folder=root_path)

    if model is not None:
        # TODO determine where models are saved
        if os.path.exists(os.path.join(root_path, "output", f"{model}.pth")):
            model_path = os.path.join(root_path, "output", f"{model}.pth")
            experiment.log_model(f"{model}", model_path)

    if asset_path is not None:
        asset_name = os.path.basename(asset_path)
        experiment.log_asset(file_data=asset_path,
                             file_name=asset_name.split('.')[0])

    if name:
        experiment.set_name(name)
    else:
        experiment.set_name(f"CYF-model")

    if isinstance(model, BaseNNModel):
        # automatically log Torch model with Comet;
        # Comet uses the native torch.save and saves the model into the assets in Comet
        comet_ml.integration.pytorch.log_model(comet_experiment, model=model, model_name=name if name else "CYF-model")

    if end:
        experiment.end()


def comet_wrapper(
        model: BaseModel,
        comet_experiment: Experiment = None,
        comet_api_key: str = None
) -> Experiment:
    """
    Wrap model before training with a Comet experiment instance

    :return: Comet_ml experiment instance
    """

    experiment = existing_comet(
        comet_experiment=comet_experiment, comet_api_key=comet_api_key
    )

    experiment.log_code(folder=root_path)

    if isinstance(model, BaseNNModel):
        comet_ml.integration.pytorch.watch(model, 1)
        experiment.log_parameters(model._modules)

    return experiment


def log_benchmark_to_comet(
        results_dict: dict,
        model_name: str,
        run_name: str,
        params: dict = None,
        comet_experiment: Experiment = None,
        comet_api_key: str = None,
        end: bool = True
) -> None:
    """
    Function to log benchmark results to comet
    """

    assert 'df_metrics' in results_dict, f'Wrong dict passed. Please pass dictionary from benchmark run'

    experiment = existing_comet(comet_experiment=comet_experiment, comet_api_key=comet_api_key)

    experiment.set_name(f'Benchmark: {run_name} {model_name}')
    experiment.log_code(folder=root_path)
    experiment.add_tag('benchmark')

    # Log each year as a table in comet
    path_results = os.path.join(PATH_RESULTS_DIR, run_name)
    files = [f for f in os.listdir(path_results) if os.path.isfile(os.path.join(path_results, f))]

    for f in files:
        experiment.log_table(filename=f"{os.path.join(path_results, f)}")

    if params is not None:
        experiment.log_parameters(params)

    # Log Comet metrics from benchmark
    df = results_dict['df_metrics']

    experiment.log_table(filename=f"{run_name}_computed_metrics.csv",
                         tabular_data=df)

    expanded_df = df.to_dict('index')

    metrics_data = {'normalized_rmse': {}, 'mape': {}}

    for (model, year, metric), value_dict in expanded_df.items():
        if metric not in metrics_data:
            metrics_data[metric] = {}
        if year not in metrics_data[metric]:
            metrics_data[metric][year] = {}
        metrics_data[metric][year][model] = value_dict['value']

    # Log metrics for each year
    for metric, years_data in metrics_data.items():
        for year, models_data in years_data.items():
            log_data = {f"{model}_{year}_{metric}": value for model, value in models_data.items()}
            experiment.log_metrics(log_data, epoch=year)

    # Log metrics to create scatter plots
    for (model, year, metric), value_dict in expanded_df.items():
        value = value_dict['value']
        if metric == 'normalized_rmse':
            experiment.log_metric(f'{model}_normalized_rmse', value, step=year)
        elif metric == 'mape':
            experiment.log_metric(f'{model}_mape', value, step=year)

    print("Go to Comet UI to show charts")

    if end:
        experiment.end()


def log_benchmark_to_comet_post_hoc(
        run_name: str,
        model_name: str,
        params: dict = None,
        comet_experiment: Experiment = None,
        comet_api_key: str = None,
        end: bool = True
) -> None:
    """
    Log benchmark results to comet after the training run from saved files
    """

    assert os.path.isdir(os.path.join(PATH_RESULTS_DIR, run_name)), f'{run_name} does not exist!'
    path_results = os.path.join(PATH_RESULTS_DIR, run_name)
    assert os.listdir(path_results), f'Directory empty...'

    results = _compute_evaluation_results(run_name)
    results = {'df_metrics': results}

    log_benchmark_to_comet(results,
                           model_name,
                           run_name,
                           params,
                           comet_experiment,
                           comet_api_key,
                           end)
