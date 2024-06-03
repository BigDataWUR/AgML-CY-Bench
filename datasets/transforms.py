import numpy as np
import torch
from datasets.dataset import Dataset
from datasets.dataset_torch import TorchDataset

def date_to_dekad(date_str):
    date_str = date_str.replace("-", "").replace("/", "")
    month = int(date_str[4:6])
    day_of_month = int(date_str[6:])
    dekad = (month - 1) * 3
    if (day_of_month <= 10):
        dekad += 1
    elif (day_of_month <= 20):
        dekad += 2
    else:
        dekad += 3
    return dekad


def transform_single_ts_feature_to_dekadal(ts_key, value, dates):
    # Transform dates to dekads
    bs = value.shape[0]
    datestrings = [str(date) for date in dates]
    dekads = torch.tensor([date_to_dekad(date) for date in datestrings])
    dekads -= 1
    
    # Aggregate timeseries to dekadal resolution
    bs = value.shape[0]
    num_groups = dekads.max().item() + 1
    if ts_key in ["tmax"]: # Max aggregation
        new_value = torch.full((bs, num_groups), float('-inf'), dtype=value.dtype)
        for i in range(num_groups): # Inefficient, but scatter_min or scatter_max are not supported by torch
            mask = (dekads == i)
            new_value[:, i] = value[:, mask].max(dim=1).values
    elif ts_key in ["tmin"]: # Min aggregation
        new_value = torch.full((bs, num_groups), float('inf'), dtype=value.dtype)
        for i in range(num_groups): # Inefficient, but scatter_min or scatter_max are not supported by torch
            mask = (dekads == i)
            new_value[:, i] = value[:, mask].min(dim=1).values
    else: # Mean aggregation: use scatter_add to sum and count, then calculate the average
        dekads = dekads.unsqueeze(0).expand(value.shape)
        dekad_sum = torch.zeros((bs, num_groups), dtype=value.dtype)
        dekad_count = torch.zeros((bs, num_groups), dtype=torch.int32)
        dekad_sum.scatter_add_(1, dekads, value)
        dekad_count.scatter_add_(1, dekads, torch.ones_like(value, dtype=torch.int32))
        new_value = dekad_sum / dekad_count
    return new_value


def transform_ts_features_to_dekadal(batch_dict):
    out_dict = {}
    dates = batch_dict["dates"]
    for key, value in batch_dict.items():
        if type(value) == torch.Tensor and len(value.shape) == 2:
            out_dict[key] = transform_single_ts_feature_to_dekadal(key, value, dates[key])
        else:
            out_dict[key] = value
    return out_dict


def transform_stack_ts_static_features(batch_dict):
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
    if len(ts) != 0: ts = torch.cat([v.unsqueeze(2) for k, v in ts.items()], dim=2)
    else: ts = None
    if len(static) != 0: static = torch.cat([v.unsqueeze(1) for k, v in static.items()], dim=1)
    else: static = None

    return {"ts": ts, "static": static, "other": other}

def test_transform():
    import time
    train_dataset = Dataset.load("test_softwheat_nl")
    train_dataset = TorchDataset(train_dataset)

    # Get a batch
    batch = [train_dataset[i] for i in range(16)]
    batch_0 = TorchDataset.collate_fn(batch).copy()
    start = time.time()
    n_repeats = 100
    for i in range(n_repeats):
        batch = batch_0.copy()
        batch = transform_ts_features_to_dekadal(batch)
        batch = transform_stack_ts_static_features(batch)
    print("Elapsed time:", time.time() - start)
    print(f"Time per transform: {(time.time() - start) / n_repeats :.2e} s")
    return



if __name__ == "__main__":
    test_transform()