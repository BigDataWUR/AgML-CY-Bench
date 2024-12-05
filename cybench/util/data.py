import pandas as pd
import numpy as np


def data_to_pandas(data_items, data_cols=None):
    """Convert data items as dict to pandas DataFrame

    Args:
      data_items : list of data items, each of which is a dict
      data_cols : list of keys to include as columns

    Returns:
      pd.DataFrame
    """
    data = []
    for item in data_items:
        if data_cols is None:
            data_cols = list(item.keys())

        data.append([item[c] for c in data_cols])

    return pd.DataFrame(data, columns=data_cols)


def trim_time_series_data(sample: dict, num_time_steps: int, time_series_keys: list):
    """Trims time series data to provided number of time steps

    Args:
      sample (dict): key is name, value is np.ndarray
      num_time_steps (int): number of time steps to keep
      time_series_keys (list): keys for time series data

    Returns:
      the same sample with modified data
    """
    for k in time_series_keys:
        if sample[k].shape[0] > num_time_steps:
            sample[k] = sample[k][-num_time_steps:]
        elif sample[k].shape[0] < num_time_steps:
            sample[k] = np.pad(
                sample[k],
                (num_time_steps - sample[k].shape[0], 0),
                mode="constant",
                constant_values=0.0,
            )

    return sample
