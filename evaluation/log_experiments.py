import os
import comet_ml
import comet_ml.integration.pytorch
from comet_ml import Experiment

from models.model import BaseModel
from models.sklearn_model import SklearnModel
from models.trend_model import TrendModel
from models.naive_models import AverageYieldModel
from models.nn_models import BaseNNModel

# get paths
evaluation_path = os.path.dirname(os.path.realpath(__file__))
root_path = evaluation_path[:-10]  # remove "evaluation" in the string to get the root path

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
    check_api_key_file = os.path.isdir(os.path.join(evaluation_path, 'api_keys', file_name))

    if check_api_key_dir and check_api_key_file:
        with open(os.path.join(evaluation_path, 'api_keys', 'comet_ml'), 'r') as f:
            api_key = f.readline()
    return api_key


def existing_comet(comet_experiment: Experiment,
                   comet_api_key: str = None,
                   project_name: str = None,
                   workspace: str = None) -> Experiment:
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
                auto_histogram_gradient_logging=True
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
                auto_histogram_gradient_logging=True
            )
    else:
        experiment = comet_experiment

    return experiment


def log_to_comet_post_hoc(metrics: dict,
                          params: dict,
                          comet_experiment: Experiment = None,
                          comet_api_key: str = None,
                          name: str = None,
                          model: BaseModel = None,
                          asset_path: str = None,
                          end: bool = False) -> None:
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


def comet_wrapper(model: BaseModel,
                  comet_experiment: Experiment = None,
                  comet_api_key: str = None) -> Experiment:
    """
    Wrap model before training with a Comet experiment instance

    :return: Comet_ml experiment instance
    """

    experiment = existing_comet(comet_experiment=comet_experiment, comet_api_key=comet_api_key)

    experiment.log_code(folder=root_path)

    if isinstance(model, BaseNNModel):
        comet_ml.integration.pytorch.watch(model, 1)
        experiment.log_parameters(model._modules)

    return experiment
