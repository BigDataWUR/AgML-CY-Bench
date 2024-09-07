import torch
import torch.utils.data
import pandas as pd

from cybench.config import KEY_LOC, KEY_YEAR, KEY_TARGET, KEY_DATES

from cybench.datasets.dataset import Dataset
from cybench.util.torch import batch_tensors


class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: Dataset):
        """
        PyTorch Dataset wrapper for compatibility with torch DataLoader objects
        :param dataset:
        """
        self._dataset = dataset
        # NOTE: Crop calendar dataframe comes with the original sos and eos.
        # They are days of the year (from 1 to 366). Because they are averages,
        # the numbers may be floats. Convert to ints.
        df_crop_cal = dataset._dfs_x["crop_calendar"]

        # NOTE: Code to compute sos_date and eos_date.
        # df_ts = df_ts.merge(crop_cal_df[crop_cal_cols], on=[KEY_LOC])
        # df_ts["sos_date"] = pd.to_datetime(df_ts[KEY_YEAR] * 1000 + df_ts["sos"], format="%Y%j")
        # df_ts["eos_date"] = pd.to_datetime(df_ts[KEY_YEAR] * 1000 + df_ts["eos"], format="%Y%j")
        # 
        # Fix sos_date for data that are in a different year than sos_date.
        # Say maize, AO sos_date is 20011124 and eos_date is 20020615.
        # We want the data from 20020101 to 20020615 to have the sos_date of
        # 20011124.
        # df_ts["sos_date"] = np.where(
        #     (df_ts["date"] <= df_ts["eos_date"]) & (df_ts["sos"] > df_ts["eos"]),
        #     # select sos_date for the previous year because season started
        #     # in the previous year.
        #     df_ts["sos_date"] + pd.offsets.DateOffset(years=-1),
        #     df_ts["sos_date"],
        # )

        # # Validate sos_date: date - sos_date should not be more than 366 days
        # assert df_ts[(df_ts["date"] - df_ts["sos_date"]).dt.days > 366].empty

        # # Fix eos_date for data that are after the current season's eos_date.
        # # Say eos_date for maize, NL is 20010728. All data after 20010728 belong to
        # # the season that ends in 2002. We change the eos_date for those data to be
        # # next year's eos_date.
        # # NOTE: This works only for static crop calendar.
        # df_ts["eos_date"] = np.where(
        #     (df_ts["date"] > df_ts["eos_date"]),
        #     # select eos_date for the next year
        #     df_ts["eos_date"] + pd.offsets.DateOffset(years=1),
        #     df_ts["eos_date"],
        # )

        # # Validate eos_date: eos_date - date should not be more than 366 days
        # assert df_ts[(df_ts["eos_date"] - df_ts["date"]).dt.days > 366].empty

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
        return {
            KEY_LOC: sample[KEY_LOC],
            KEY_YEAR: sample[KEY_YEAR],
            KEY_TARGET: torch.tensor(sample[KEY_TARGET], dtype=torch.float32),
            KEY_DATES: sample[KEY_DATES],
            **{
                key: torch.tensor(sample[key], dtype=torch.float32)
                for key in sample.keys()
                if key not in [KEY_LOC, KEY_YEAR, KEY_TARGET, KEY_DATES]
            },  # TODO -- support nonnumeric data?
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
        feature_names.remove(KEY_DATES)

        batched_samples = {
            KEY_TARGET: batch_tensors(*[sample[KEY_TARGET] for sample in samples]),
            KEY_LOC: [sample[KEY_LOC] for sample in samples],
            KEY_YEAR: [sample[KEY_YEAR] for sample in samples],
            KEY_DATES: samples[0][KEY_DATES],
            **{
                key: batch_tensors(*[sample[key] for sample in samples])
                for key in feature_names
            },
        }

        return batched_samples
