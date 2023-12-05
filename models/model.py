import abc
# from datasets.dataset import AGMLDataset

class AgMLBaseModel(abc.ABC):

  @abc.abstractmethod
  def fit(self, AgMLDataset):
    raise NotImplementedError

  @abc.abstractmethod
  def predict(self, X):
    raise NotImplementedError

  @abc.abstractmethod
  def save(self, model_name):
    raise NotImplementedError

  @abc.abstractclassmethod
  def load(cls, model_name):
    raise NotImplementedError