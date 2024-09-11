import torch
import torch.utils.data

from cybench.config import (
    KEY_LOC,
    KEY_YEAR,
    KEY_TARGET,
    KEY_DATES,
    CROP_CALENDAR_DATES,
    ALL_PREDICTORS,
)

from cybench.datasets.dataset import Dataset
from cybench.util.torch import batch_tensors


class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: Dataset):
        """
        PyTorch Dataset wrapper for compatibility with torch DataLoader objects
        :param dataset:
        """
        self._dataset = dataset

        # NOTE: Crop calendar data comes with KEY_LOC, KEY_YEAR, sos_date, eos_date.
        # df_crop_cal = dataset._dfs_x["crop_calendar"]

        # NOTE:
        # data can be in different resolution
        # data can have different dates
        # some dates in one location and year, different dates in another location and year
        # data can have different number of dates and values
        # TODO interpolation
        # TODO aggregation to dekadal resolution
        # TODO Ensure number of time steps is the same for all locations, years and
        # time series data sources.

    def get_normalization_params(self, normalization="standard"):
        """
        Compute normalization parameters for input data.
        :param normalization: normalization method, default standard or z-score
        :return: a dict containing normalization parameters (e.g. mean and std)
        """
        return self._dataset.get_normalization_params(normalization=normalization)

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
        nontensors1 = {
            KEY_LOC: sample[KEY_LOC],
            KEY_YEAR: sample[KEY_YEAR],
            KEY_DATES: sample[KEY_DATES],
        }

        # crop calendar dates are datetime objects
        nontensors2 = {k: sample[k] for k in CROP_CALENDAR_DATES if k in sample}

        tensors1 = {
            KEY_TARGET: torch.tensor(sample[KEY_TARGET], dtype=torch.float32),
        }

        tensors2 = {
            key: torch.tensor(sample[key], dtype=torch.float32)
            for key in ALL_PREDICTORS
        }  # TODO -- support nonnumeric data?

        return {**nontensors1, **nontensors2, **tensors1, **tensors2}

    @classmethod
    def collate_fn(cls, samples: list) -> dict:
        """
        Function that takes a list of data samples (as dicts, containing torch tensors)
        and converts it to a dict of batched torch tensors
        :param samples: a list of data samples
        :return: a dict with batched data
        """
        assert len(samples) > 0

        nontensors1 = {
            KEY_LOC: [sample[KEY_LOC] for sample in samples],
            KEY_YEAR: [sample[KEY_YEAR] for sample in samples],
            KEY_DATES: samples[0][KEY_DATES],
        }

        # crop calendar dates are datetime objects
        nontensors2 = {
            k: [sample[k] for sample in samples if k in sample]
            for k in CROP_CALENDAR_DATES
        }

        tensors1 = {
            KEY_TARGET: batch_tensors(*[sample[KEY_TARGET] for sample in samples])
        }

        tensors2 = {
            key: batch_tensors(*[sample[key] for sample in samples])
            for key in ALL_PREDICTORS
        }

        return {**nontensors1, **nontensors2, **tensors1, **tensors2}
