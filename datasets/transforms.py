import torch
from datasets.dataset import Dataset
from datasets.dataset_torch import TorchDataset
from util.features import dekad_from_date
from config import KEY_DATES


def _transform_ts_input_to_dekadal(ts_key, value, dates, min_date, max_date):
    # Transform dates to dekads
    min_dekad = dekad_from_date(min_date)
    max_dekad = dekad_from_date(max_date)
    dekads = list(range(0, max_dekad - min_dekad + 1))

    datestrings = [str(date) for date in dates]
    value_dekads = torch.tensor(
        [dekad_from_date(date) for date in datestrings], device=value.device
    )
    value_dekads -= 1

    # Aggregate timeseries to dekadal resolution
    new_value = torch.full(
        (value.shape[0], len(dekads)), float("inf"), dtype=value.dtype
    )
    for d in dekads:
        # Inefficient, but scatter_min or scatter_max are not supported by torch
        mask = value_dekads == d
        if value[:, mask].shape[1] == 0:
            new_value[:, d] = 0.0
        else:
            if ts_key in ["tmax"]:  # Max aggregation
                new_value[:, d] = value[:, mask].max(dim=1).values
            elif ts_key in ["tmin"]:  # Min aggregation
                new_value[:, d] = value[:, mask].min(dim=1).values
            else:  # for all other inputs
                new_value[:, d] = torch.mean(value[:, mask], dim=1)
    return new_value


def transform_ts_inputs_to_dekadal(batch_dict, min_date, max_date):
    out_dict = {}
    dates = batch_dict[KEY_DATES]
    for key, value in batch_dict.items():
        if type(value) == torch.Tensor and len(value.shape) == 2:
            out_dict[key] = _transform_ts_input_to_dekadal(
                key, value, dates[key], min_date, max_date
            )
        else:
            out_dict[key] = value
    return out_dict


def transform_stack_ts_static_inputs(batch_dict, min_date, max_date):
    # Sort values
    ts = {}
    static = {}
    other = {}
    for key, value in batch_dict.items():
        if type(value) != torch.Tensor:
            other[key] = value
        elif len(value.shape) == 1:
            static[key] = value
        else:
            ts[key] = value

    # Stack values
    if len(ts) != 0:
        ts = torch.cat([v.unsqueeze(2) for k, v in ts.items()], dim=2)
    else:
        ts = None
    if len(static) != 0:
        static = torch.cat([v.unsqueeze(1) for k, v in static.items()], dim=1)
    else:
        static = None

    return {"ts": ts, "static": static, "other": other}
