from abc import ABC, abstractmethod
from typing import overload

class AgMLBaseModel(ABC):
    @abstractmethod
    def fit(self, train_dataset):
        """
        Fit or train the model.
        Args:
          train_dataset: The training dataset.
        """
        raise NotImplementedError

    @abstractmethod
    @overload
    def predict(self, test_dataset):
        """
        Run the model on the entire test set.
        Args:
          test_dataset: The test dataset.
        Returns:
          An numpy ndarray of predictions.
        """
        raise NotImplementedError

    @abstractmethod
    @overload
    def predict(self, data):
        """
        Run the model on selected data items.
        Args:
          items: Data items that include predictors and labels.
        Returns:
          An numpy ndarray of predictions.
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, model_name):
        """
        Saves model, e.g. using pickle.
        Args:
          model_name: Name of the file that will be used to pickle the model.
        """
        raise NotImplementedError

    @abstractmethod
    def load(cls, model_name):
        """
        Deserializes or unpickles a model saved using pickle.
        Args:
          model_name: The name that was used to save the model.
        Returns:
          The unpickled model.
        """
