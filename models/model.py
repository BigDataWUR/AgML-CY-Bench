"""Model base class
"""

from abc import ABC, abstractmethod

from datasets.dataset import Dataset


class BaseModel(ABC):
    @abstractmethod
    def fit(self, dataset: Dataset, **fit_params) -> tuple:
        """Fit or train the model.

        Args:
          dataset: Dataset

          **fit_params: Additional parameters.

        Returns:
          A tuple containing the fitted model and a dict with additional information.
        """
        raise NotImplementedError

    def predict(self, dataset: Dataset, **predict_params) -> tuple:
        """Run fitted model on data.

        Args:
          dataset: Dataset
          **predict_params: Additional parameters.

        Returns:
          A tuple containing a np.ndarray and a dict with additional information.
        """
        batch = [d for d in dataset]
        return self.predict_batch(batch, **predict_params)

    @abstractmethod
    def predict_batch(self, X: list, **predict_params):
        """Run fitted model on batched data items.

        Args:
          X: a list of data items, each of which is a dict
          **predict_params: Additional parameters.

        Returns:
          A tuple containing a np.ndarray and a dict with additional information.
        """
        raise NotImplementedError

    def predict_item(self, X: dict, **predict_params):
        """Run fitted model on one data item.

        Args:
          X: a data item
          **predict_params: Additional parameters.

        Returns:
          A tuple containing a np.ndarray and a dict with additional information.
        """
        batch = [X]
        return self.predict_batch(batch, **predict_params)

    @abstractmethod
    def save(self, model_name):
        """Save model, e.g. using pickle.

        Args:
          model_name: Filename that will be used to save the model.
        """
        raise NotImplementedError

    @abstractmethod
    def load(cls, model_name):
        """Deserialize a saved model.

        Args:
          model_name: Filename that was used to save the model.

        Returns:
          The deserialized model.
        """
        raise NotImplementedError
