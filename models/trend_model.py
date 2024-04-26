import pickle
import numpy as np
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

from models.model import BaseModel
from datasets.dataset import Dataset
from util.data import data_to_pandas

from config import KEY_YEAR, KEY_TARGET


class TrendModel(BaseModel):
    """Default trend estimator.

    Trend is estimated using years as features.
    If the data includes multiple locations or admin regions,
    it's better to estimate per-region trend. If data for a country or multiple
    regions is passed, TrendModel will compute the overall trend.
    """
    def __init__(self, trend="linear"):
        self._trend = trend
        self._trend_est = None

    def _linear_trend_estimator(self, trend_x, trend_y):
        """Implements a linear trend.
        Args:
          trend_x: a list of years.
          trend_y: a list of values (e.g. yields)
          pred_x: year for which to predict trend
        Returns:
          A linear trend estimator
        """
        trend_x = add_constant(trend_x)
        linear_trend_est = OLS(trend_y, trend_x).fit()

        return linear_trend_est

    def _quadratic_trend_estimator(self, trend_x, trend_y):
        """Implements a quadratic trend. Suggested by @ritviksahajpal.
        Args:
          trend_x: a np.ndarray of years.
          trend_y: a np.ndarray of values (e.g. yields)
        Returns:
          A quadratic trend estimator (with an additive quadratic term)
        """
        quad_x = add_constant(np.column_stack((trend_x, trend_x ** 2)), has_constant="add")
        quad_est = OLS(trend_y, quad_x).fit()

        return quad_est

    def fit(self, dataset: Dataset, **fit_params) -> tuple:
        """Fit or train the model.
        Args:
          dataset: Dataset
          **fit_params: Additional parameters.
        Returns:
          A tuple containing the fitted model and a dict with additional information.
        """
        train_df = data_to_pandas(dataset)
        trend_x = train_df[KEY_YEAR].values
        trend_y = train_df[KEY_TARGET].values
        # NOTE: trend can be "linear" or "quadratic". We could implement LOESS.
        if (self._trend == "quadratic"):
            self._trend_est = self._quadratic_trend_estimator(trend_x, trend_y)
        else:
            self._trend_est = self._linear_trend_estimator(trend_x, trend_y)

        return self, {}

    def predict_batch(self, X: list):
        """Run fitted model on batched data items.
        Args:
          X: a list of data items, each of which is a dict
        Returns:
          A tuple containing a np.ndarray and a dict with additional information.
        """

        test_df = data_to_pandas(X)
        trend_x = test_df[KEY_YEAR].values
        if (self._trend == "quadratic"):
            trend_x = add_constant(np.column_stack((trend_x, trend_x ** 2)), has_constant='add')
        else:
            trend_x = add_constant(trend_x, has_constant='add')

        predictions = self._trend_est.predict(trend_x)

        return predictions, {}

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
