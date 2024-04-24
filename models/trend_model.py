import pickle
import numpy as np
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

from models.model import BaseModel
from datasets.dataset import Dataset

from config import KEY_YEAR


class TrendModel(BaseModel):
    def __init__(self, x_cols, y_cols, trend_est="linear"):
        self._x_cols = x_cols
        self._y_cols = y_cols

        # Trend estimator function
        if (trend_est == "average"):
            self._trend_fn = self._get_average_trend
        elif (trend_est == "quadratic"):
            self._trend_fn = self._get_quadratic_trend
        # Default
        else:
            self._trend_fn = self._get_linear_trend

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

            predictions[i] = self._trend_fn(trend_x, trend_y, item[KEY_YEAR])

        print(predictions)
        return predictions, {}

    def _get_average_trend(self, trend_x, trend_y, pred_x):
        """Implements an average trend.
        Args:
          trend_x: a list of years.
          trend_y: a list of values (e.g. yields)
          pred_x: year for which to predict trend
        Returns:
          The trend based on average of trend_y values
        """
        return np.mean(trend_y)

    def _get_linear_trend(self, trend_x, trend_y, pred_x):
        """Implements a linear trend.
        Args:
          trend_x: a list of years.
          trend_y: a list of values (e.g. yields)
          pred_x: year for which to predict trend
        Returns:
          The trend based on linear trend of years and values
        """
        slope, coeff = np.polyfit(trend_x, trend_y, 1)
        return pred_x * slope + coeff

    def _get_quadratic_trend(self, trend_x, trend_y, pred_x):
        """Implements a quadratic trend. Contributed by @ritviksahajpal
        Args:
          trend_x: a list of years.
          trend_y: a list of values (e.g. yields)
          pred_x: year for which to predict trend
        Returns:
          The trend based on quadratic trend of years and values
        """
        trend_x = np.reshape(np.array(trend_x), (len(trend_x), 1))
        trend_y = np.reshape(np.array(trend_y), (len(trend_y), 1))
        quad_x = add_constant(np.column_stack((trend_x, trend_x ** 2)))
        quad_model = OLS(trend_y, quad_x).fit()
        pred_x = np.reshape(np.array([pred_x]), (1, 1))
        pred_x = add_constant(np.column_stack((pred_x, pred_x ** 2)), has_constant='add')
        return quad_model.predict(pred_x)

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
