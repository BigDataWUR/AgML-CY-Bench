import copy
import pandas as pd
from sklearn.model_selection import ParameterGrid


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


def flatten_nested_dict(d, parent_key="", sep="."):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_nested_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_nested_dict(d, sep="."):
    out = {}
    for k, v in d.items():
        keys = k.split(sep)
        if len(keys) == 1:
            out[k] = v
        else:
            new_key = keys[0]
            new_subkey = sep.join(keys[1:])
            if new_key not in out:
                out[new_key] = {}
            out[new_key][new_subkey] = v
    for k, v in out.items():
        if isinstance(v, dict):
            out[k] = unflatten_nested_dict(v, sep=sep)
    return out


def update_settings(new_settings: dict, standard_settings: dict):
    new_settings = flatten_nested_dict(new_settings)
    standard_settings = copy.deepcopy(standard_settings)
    standard_settings = flatten_nested_dict(standard_settings)
    standard_settings.update(new_settings)
    standard_settings = unflatten_nested_dict(standard_settings)
    return standard_settings


def generate_settings(param_space: dict, standard_settings: dict):
    settings = []
    param_space = flatten_nested_dict(param_space)
    standard_settings = flatten_nested_dict(standard_settings)
    combs = list(ParameterGrid(param_space))
    for comb in combs:
        setting = copy.deepcopy(standard_settings)
        setting.update(comb)
        settings.append(setting)
    settings = [unflatten_nested_dict(setting) for setting in settings]
    return settings
