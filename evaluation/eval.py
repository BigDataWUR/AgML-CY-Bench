import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

from config import KEY_TARGET
from models.model import BaseModel
from datasets.dataset import Dataset


implemented_metrics = {}


def metric(func):
    """Decorator to mark functions as metrics"""
    implemented_metrics[func.__name__] = func
    return func


def get_default_metrics():
    return ("normalized_rmse", "mape")


def evaluate_model(model: BaseModel, dataset: Dataset, metrics=get_default_metrics()):
    """
    Evaluate the performance of a model using specified metrics.

    Args:
      model: The trained model to be evaluated.
      dataset: Dataset.
      metrics: List of metrics to calculate.

    Returns:
      A dictionary containing the calculated metrics.
    """

    y_true = dataset.targets()
    y_pred, _ = model.predict(dataset)
    results = evaluate_predictions(y_true, y_pred, metrics)

    return results


def evaluate_predictions(
    y_true: np.ndarray, y_pred: np.ndarray, metrics=get_default_metrics()
):
    """
    Evaluate predictions using specified metrics.

    Args:
      y_true (numpy.ndarray): True labels for evaluation.
      y_pred (numpy.ndarray): Predicted values.
      metrics: List of metrics to calculate.

    Returns:
      A dictionary containing the calculated metrics.
    """

    results = {}
    for metric_name in metrics:
        metric_function = implemented_metrics.get(metric_name)
        if metric_function:
            result = metric_function(y_true, y_pred)
            results[metric_name] = result
        else:
            raise ValueError(f"Metric function '{metric_name}' not implemented.")

    return results


@metric
def normalized_rmse(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Calculate the normalized Root Mean Squared Error (RMSE) between true and predicted values.

    Args:
      y_true (numpy.ndarray): True values.
      y_pred (numpy.ndarray): Predicted values.

    Returns:
      float: Normalized RMSE value as a percentage.
    """

    mse = mean_squared_error(y_true, y_pred)
    mean_y_true = np.mean(y_true)
    return 100 * np.sqrt(mse) / mean_y_true


@metric
def mape(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Calculate Mean Absolute Percentage Error (MAPE).
    Note that in the provided implementation using scikit-learn, there is an absence of multiplication by 100

    Args:
    - y_true (numpy.ndarray): True values.
    - y_pred (numpy.ndarray): Predicted values.

    Returns:
    - float: Mean Absolute Percentage Error.
    """

    return mean_absolute_percentage_error(y_true, y_pred)
