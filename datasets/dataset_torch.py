import torch
import torch.utils.data

from config import KEY_LOC, KEY_YEAR, KEY_TARGET

from datasets.dataset import Dataset
from util.torch import batch_tensors


class TorchDataset(torch.utils.data.Dataset):

    def __init__(self, dataset: Dataset):
        """
        PyTorch Dataset wrapper for compatibility with torch DataLoader objects
        :param dataset:
        """
        self._dataset = dataset

    def __getitem__(self, index) -> dict:
        """
        Get a sample from the dataset
        Data is cast to PyTorch tensors where required
        :param index: index that is passed to the dataset
        :return: a dict containing the data sample
        """
        return self._cast_to_tensor(
            self._dataset[index],
        )

    def __len__(self):
        return len(self._dataset)

    @classmethod
    def _cast_to_tensor(cls, sample: dict) -> dict:
        """
        Create a sample with all data cast to torch tensors
        :param sample: the sample to convert
        :return: the converted data sample
        """
        return {
            KEY_LOC: sample[KEY_LOC],
            KEY_YEAR: sample[KEY_YEAR],
            KEY_TARGET: torch.tensor(sample[KEY_TARGET]),
            **{
                key: torch.tensor(sample[key]) for key in sample.keys() if key not in [KEY_LOC, KEY_YEAR, KEY_TARGET]
              }  # TODO -- support nonnumeric data?
        }

    @classmethod
    def collate_fn(cls, samples: list) -> dict:
        """
        Function that takes a list of data samples (as dicts, containing torch tensors) and converts it to a dict of
        batched torch tensors
        :param samples: a list of data samples
        :return: a dict with batched data
        """
        assert len(samples) > 0

        feature_names = set.intersection(*[set(sample.keys()) for sample in samples])
        feature_names.remove(KEY_TARGET)
        feature_names.remove(KEY_LOC)
        feature_names.remove(KEY_YEAR)

        batched_samples = {
            KEY_TARGET: batch_tensors(*[sample[KEY_TARGET] for sample in samples]),
            KEY_LOC: [sample[KEY_LOC] for sample in samples],
            KEY_YEAR: [sample[KEY_YEAR] for sample in samples],
            **{key: batch_tensors(*[sample[key] for sample in samples]) for key in feature_names}
        }

        return batched_samples

