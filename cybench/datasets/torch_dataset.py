import pandas as pd
import numpy as np
import torch
import torch.utils.data

from cybench.config import (
    KEY_LOC,
    KEY_YEAR,
    KEY_TARGET,
    KEY_DATES,
    KEY_CROP_SEASON,
    CROP_CALENDAR_DATES,
    TIME_SERIES_PREDICTORS,
    TIME_SERIES_AGGREGATIONS,
    ALL_PREDICTORS,
)

from cybench.datasets.dataset import Dataset
from cybench.util.torch import batch_tensors
from cybench.util.features import dekad_from_date


class TorchDataset(Dataset, torch.utils.data.Dataset):
    def __init__(
        self,
        dataset: Dataset,
        aggregate_time_series_to: str = None,
        max_season_window_length: int = None,
    ):
        """
        PyTorch Dataset wrapper for compatibility with torch DataLoader objects
        :param dataset: Dataset
        :param aggregate_time_series_to: aggregation resolution for time series data
                                         ("week" and "dekad" are supported)
        :param max_season_window_length: maximum length of time series
        """
        self._crop = dataset.crop
        self._df_y = dataset._df_y

        if max_season_window_length is None:
            self._max_season_window_length = dataset.max_season_window_length
        else:
            self._max_season_window_length = max_season_window_length

        self._aggregate_time_series_to = aggregate_time_series_to
        if aggregate_time_series_to is not None:
            if aggregate_time_series_to not in ["week", "dekad"]:
                raise Exception(
                    f"Unsupported time series aggregation resolution {aggregate_time_series_to}"
                )

            assert self._max_season_window_length is not None
            self._dfs_x = {}
            ts_inputs = [
                x for x in dataset._dfs_x if "date" in dataset._dfs_x[x].index.names
            ]
            for x in dataset._dfs_x:
                if x not in ts_inputs:
                    self._dfs_x[x] = dataset._dfs_x[x]

            # combine time series data
            df_ts = pd.concat(
                [dataset._dfs_x[x] for x in ts_inputs], join="outer", axis=1
            )

            # create a dataframe with all dates for every location and year
            df_all_dates = self._dfs_x[KEY_CROP_SEASON][["cutoff_date"]].copy()
            df_all_dates["date"] = df_all_dates.apply(
                lambda r: pd.date_range(
                    end=r["cutoff_date"],
                    periods=self._max_season_window_length,
                    freq="D",
                ),
                axis=1,
            )
            df_all_dates = df_all_dates.explode("date").drop(columns=["cutoff_date"])
            df_all_dates.reset_index(inplace=True)
            df_all_dates.set_index([KEY_LOC, KEY_YEAR, "date"], inplace=True)

            # merge to get all dates in time series data
            df_ts = df_ts.merge(
                df_all_dates, how="outer", on=[KEY_LOC, KEY_YEAR, "date"]
            )
            del df_all_dates

            # NOTE: interpolate fills data in forward direction.
            df_ts = df_ts.sort_index().interpolate(method="linear")
            # fill NAs in the front with 0.0
            df_ts.fillna(0.0, inplace=True)

            # aggregate
            df_ts.reset_index(inplace=True)
            if self._aggregate_time_series_to == "week":
                df_ts["week"] = df_ts["date"].dt.isocalendar().week
            else:
                df_ts["dekad"] = df_ts.apply(
                    lambda r: dekad_from_date(r["date"]), axis=1
                )

            ts_aggrs = {k: TIME_SERIES_AGGREGATIONS[k] for k in TIME_SERIES_PREDICTORS}
            # Primarily to avoid losing the "date" column.
            ts_aggrs["date"] = "min"
            df_ts = (
                df_ts.groupby([KEY_LOC, KEY_YEAR, self._aggregate_time_series_to])
                .agg(ts_aggrs)
                .reset_index()
            )
            df_ts.drop(columns=[self._aggregate_time_series_to], inplace=True)

            # Add time series to self._dfs_x
            self._dfs_x["combined_ts"] = df_ts.set_index(
                [KEY_LOC, KEY_YEAR, "date"]
            ).sort_index()
        else:
            self._dfs_x = dataset._dfs_x

        # Bool value that specifies whether missing data values are allowed
        # For now always set to False
        self._allow_incomplete = False

    def __getitem__(self, index) -> dict:
        """
        Get a sample from the dataset
        Data is cast to PyTorch tensors where required
        :param index: index that is passed to the dataset
        :return: a dict containing the data sample
        """
        sample = super(TorchDataset, self).__getitem__(index)
        if self._aggregate_time_series_to is not None:
            if self._aggregate_time_series_to == "week":
                num_time_steps = int(np.ceil(self._max_season_window_length / 7))
            else:
                num_time_steps = int(np.ceil(self._max_season_window_length / 10))

            # make sure time series data has the same number of time steps
            for k in TIME_SERIES_PREDICTORS:
                if sample[k].shape[0] > num_time_steps:
                    sample[k] = sample[k][-num_time_steps:]
                elif sample[k].shape[0] < num_time_steps:
                    sample[k] = np.pad(
                        sample[k],
                        (num_time_steps - sample[k].shape[0], 0),
                        mode="constant",
                        constant_values=0.0,
                    )

        return self._cast_to_tensor(sample)

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
