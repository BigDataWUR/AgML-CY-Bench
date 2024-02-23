import pickle
import numpy as np

from models.model import BaseModel
from datasets.dataset import Dataset


class TrendModel(BaseModel):
    def __init__(self, x_cols, y_cols, year_col):
        self._x_cols = x_cols
        self._y_cols = y_cols
        self._year_col = year_col

    def fit(self, dataset: Dataset, **fit_params) -> tuple:
        """Fit or train the model.

        Args:
          dataset: Dataset

          **fit_params: Additional parameters.

        Returns:
          A tuple containing the fitted model and a dict with additional information.
        """
        # No training required.
        return self, {}

    def predict_batch(self, X: list):
        """Run fitted model on batched data items.

        Args:
          X: a list of data items, each of which is a dict

        Returns:
          A tuple containing a np.ndarray and a dict with additional information.
        """
        predictions = np.zeros((len(X), 1))
        for i, item in enumerate(X):
            trend_x = [item[y] for y in self._x_cols]
            trend_y = [item[c] for c in self._y_cols]
            predictions[i] = self._get_trend(trend_x, trend_y, item[self._year_col])

        return predictions, {}

    def _get_trend(trend_x, trend_y, pred_y):
        """
        Now this implements a linear trend.
        """
        slope, coeff = np.polyfit(trend_x, trend_y)
        return (pred_y * slope + coeff)

    def save(self, model_name):
        """Save model, e.g. using pickle.

        Args:
          model_name: Filename that will be used to save the model.
        """
        with open(model_name, "wb") as f:
            pickle.dump(self, f)

    def load(cls, model_name):
        """Deserialize a saved model.

        Args:
          model_name: Filename that was used to save the model.

        Returns:
          The deserialized model.
        """
        with open(model_name, "rb") as f:
            saved_model = pickle.load(f)

        return saved_model
