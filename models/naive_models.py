import pickle
import numpy as np

from models.model import BaseModel
from datasets.dataset import Dataset
from util.data import data_to_pandas
from config import KEY_TARGET


class AverageYieldModel(BaseModel):
    """A naive yield prediction model.

    Predicts the average of the training set.
    If the data includes multiple locations or admin regions,
    it's better to estimate per-region average. If data for a country or multiple
    regions is passed, AverageYieldModel will compute the overall average.
    """
    def __init__(self):
        self._train_df = None

    def fit(self, dataset: Dataset, **fit_params) -> tuple:
        """Fit or train the model.

        Args:
          dataset: Dataset

          **fit_params: Additional parameters.

        Returns:
          A tuple containing the fitted model and a dict with additional information.
        """
        self._train_df = data_to_pandas(dataset)
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
            predictions[i] = self._train_df[KEY_TARGET].mean()

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
