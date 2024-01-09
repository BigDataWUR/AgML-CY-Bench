from abc import ABC, abstractmethod

from datasets.dataset import Dataset


class BaseModel(ABC):
    @abstractmethod
    def fit(self, dataset: Dataset):
        """
        Fit or train the model.

        Args:
          train_dataset: The training dataset.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, data: dict) -> tuple:
        """
        Perform inference on data. The data may be batched.

        Args:
          data: Predictors for one or more data items. May include labels.

        Returns:
          A tuple including predictions.
        """
        raise NotImplementedError

    def _set_training(self, training: bool = True):
        """
        Set training or evaluation mode.

        Args:
          training: bool (default=True)
          Whether to set the training mode (True) or evaluation model (False).
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, save_path):
        """
        Saves model, e.g. using pickle.

        Args:
          save_path: File path that will be used to pickle the model.
        """
        raise NotImplementedError

    @abstractmethod
    def load(cls, load_path):
        """
        Deserializes or unpickles a model saved using pickle.

        Args:
          load_path: File path that was used to save the model.

        Returns:
          The unpickled model.
        """
        raise NotImplementedError
