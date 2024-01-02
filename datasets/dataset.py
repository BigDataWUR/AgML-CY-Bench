from abc import ABC, abstractmethod

class AgMLBaseDataset(ABC):
    @abstractmethod
    def __len__(self) -> int:
        """
        Return the size of dataset.
        Returns:
          An int representing the number of data items in the dataset.
        """
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, index):
        """
        Run the model on the entire test set.
        Args:
          index: It is either an integer or a tuple.
        Returns:
          A dict with column names as keys and data values for given index as values.
        """
        raise NotImplementedError
