import pickle
import numpy as np
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
import pymannkendall as trend_mk

from cybench.models.model import BaseModel
from cybench.datasets.dataset import Dataset
from cybench.util.data import data_to_pandas

from cybench.config import KEY_LOC, KEY_YEAR, KEY_TARGET


class TrendModel(BaseModel):
    """Default trend estimator.

    Trend is estimated using years as features.
    """

    def __init__(self, trend="linear"):
        self._trend = trend
        self._trend_estimators = {}

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
        quad_x = add_constant(
            np.column_stack((trend_x, trend_x**2)), has_constant="add"
        )
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
        sel_cols = [KEY_LOC, KEY_YEAR, KEY_TARGET]
        train_df = data_to_pandas(dataset, data_cols=sel_cols)
        loc_ids = train_df[KEY_LOC].unique()
        for loc in loc_ids:
            loc_df = train_df[train_df[KEY_LOC] == loc]
            trend_x = loc_df[KEY_YEAR].values
            trend_y = loc_df[KEY_TARGET].values

            result = trend_mk.original_test(trend_y)
            # NOTE Changing this condition may require an update to
            # test_trend_model in tests/models/test_model.py.
            if (trend_x.shape[0] < 4) or not result.h:
                self._trend_estimators[loc] = {
                    "estimator" : None,
                    "mean" : np.mean(trend_y)
                }
            else:
              # NOTE: trend can be "linear" or "quadratic". We could implement LOESS.
              if self._trend == "quadratic":
                  trend_est = self._quadratic_trend_estimator(trend_x, trend_y)
              else:
                  trend_est = self._linear_trend_estimator(trend_x, trend_y)

              self._trend_estimators[loc] = {
                  "estimator" : trend_est,
                  "mean" : None,
              }

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
            loc_id = item[KEY_LOC]
            year = item[KEY_YEAR]
            trend_est = self._trend_estimators[loc_id]["estimator"]
            if (trend_est is None):
                predictions[i] = self._trend_estimators[loc_id]["mean"]
            else:
                trend_x = np.array([year]).reshape((1, 1))
                if self._trend == "quadratic":
                    trend_x = add_constant(
                        np.column_stack((trend_x, trend_x**2)), has_constant="add"
                    )
                else:
                    trend_x = add_constant(trend_x, has_constant="add")

                predictions[i] = trend_est.predict(trend_x)

        return predictions.flatten(), {}

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
