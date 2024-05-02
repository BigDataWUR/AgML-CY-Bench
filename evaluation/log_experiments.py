import os

from comet_ml import Experiment
import torch
import sklearn
from models.sklearn_model import SklearnModel
from models.trend_model import TrendModel
from models.naive_models import AverageYieldModel

# get paths
evaluation_path = os.path.dirname(os.path.realpath(__file__))
root_path = evaluation_path[:-10]

# TODO set path
with open(os.path.join(evaluation_path, 'api_keys', 'comet_ml'), 'r') as f:
    api_key = f.readline()


# Configure Comet experiment instance

comet_api_key = api_key
project_name = "Crop-yield-forecasting"
workspace = "AgML"


def log_to_comet(metrics: dict,
                 params: dict,
                 name: str = None,
                 model: str=None,
                 asset_path: str=None):
    """
    Log metrics, params, asset and model to Comet_ml

    :param metrics: dict of metrics
    :param params: dict of params/hyperparams
    :param model: name of the saved model
    :param name: name of the comet experiment
    :param asset_path: path to asset. To log a custom asset (e.g., config file etc.) to Comet.
    :return:
    """
    experiment = Experiment(
        api_key=comet_api_key,
        project_name=project_name,
        workspace=workspace
    )
    experiment.log_parameters(params)
    experiment.log_metrics(metrics)
    experiment.log_code(folder=root_path)

    if model is not None:
        # TODO determine where models are saved
        model_path = os.path.join(root_path, "output", f"{model}.pth")
        experiment.log_model(f"{model}", model_path)

    if asset_path is not None:
        asset_name = os.path.basename(asset_path)
        experiment.log_asset(file_data=asset_path,
                             file_name=asset_name.split('.')[0])

    if name:
        experiment.set_name(name)
    elif name and model:
        experiment.set_name(model)
    else:
        experiment.set_name(f"CYM-model")

    experiment.end()
