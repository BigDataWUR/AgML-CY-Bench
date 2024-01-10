"""Model base class

The API takes some ideas from skorch (https://github.com/skorch-dev/skorch/).
"""

from abc import ABC, abstractmethod


class BaseModel(ABC):
    @abstractmethod
    def fit(self, X, y=None, epochs=None, **fit_params):
        """Fit or train the model.

        Args:
          X: Input data, which can be
            * numpy array
            * torch tensor
            * a dictionary of numpy array or torch tensor
            * pandas DataFrame
            * Dataset

          y: Target data. Supported data types are the same as for ``X``.
            If ``X`` is a dictionary or Dataset that contains the target, ``y`` may be None.

          epochs: int or None (default=None)
            If not None, train for this many epochs.

          **fit_params: Additional parameters.

          Returns:
            self: Fitted model.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, X):
        """Run fitted model on data.

        Args:
          X: Input data, which can be
            * numpy array
            * torch tensor
            * a dictionary of numpy array or torch tensor
            * pandas DataFrame
            * Dataset

        Returns:
          Predictions, which can be
            * numpy array
            * torch tensor
            * pandas DataFrame
        """
        raise NotImplementedError

    @abstractmethod
    def _set_training(self, training: bool = True):
        """Set training or evaluation mode.

        Args:
          training: bool (default=True)
          Whether to set the training mode (True) or evaluation model (False).
        """
        raise NotImplementedError

    @abstractmethod
    def _get_data_splits(self, X, y=None, **split_params):
        """Get training and validation splits for internal validation.

        Args:
          X: Input data, which can be
            * numpy array
            * torch tensor
            * a dictionary of numpy array or torch tensor
            * pandas DataFrame
            * Dataset

          y: Target data. Supported data types are the same as for ``X``.
            If ``X`` is a dictionary or Dataset that contains the target, ``y`` may be None.

          **split_params: Additional parameters for splitting data.
        """
        raise NotImplementedError

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
