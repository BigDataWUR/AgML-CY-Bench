import pickle
import numpy as np

from models.model import BaseModel
from datasets.dataset import Dataset
from util.data import data_to_pandas


class AverageYieldModel(BaseModel):
    def __init__(self, group_cols, label_col):
        self._train_df = None
        self._averages = None
        self._group_cols = group_cols
        self._label_col = label_col

    def fit(self, dataset: Dataset, **fit_params) -> tuple:
        """Fit or train the model.

        Args:
          dataset: Dataset

          **fit_params: Additional parameters.

        Returns:
          A tuple containing the fitted model and a dict with additional information.
        """
        self._train_df = data_to_pandas(dataset)
        self._averages = (
            self._train_df.groupby(self._group_cols)
            .agg(GROUP_AVG=(self._label_col, "mean"))
            .reset_index()
        )

        return self

    def predict_batch(self, X: list):
        """Run fitted model on batched data items.

        Args:
          X: a list of data items, each of which is a dict

        Returns:
          A tuple containing a np.ndarray and a dict with additional information.
        """
        predictions = np.zeros((len(X), 1))
        for i, item in enumerate(X):
            filter_condition = None
            for g in self._group_cols:
                if filter_condition is None:
                    filter_condition = self._averages[g] == item[g]
                else:
                    filter_condition &= self._averages[g] == item[g]

            filtered = self._averages[filter_condition]
            # If there is no matching group in training data,
            # predict the global average
            if filtered.empty:
                y_pred = self._train_df[self._label_col].mean()
            else:
                y_pred = filtered["GROUP_AVG"].values[0]

            predictions[i] = y_pred

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
