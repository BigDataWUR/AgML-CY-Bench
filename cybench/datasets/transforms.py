import torch
from cybench.util.features import dekad_from_date
from cybench.config import (
    KEY_DATES,
    TIME_SERIES_PREDICTORS,
)


def transform_ts_inputs_to_dekadal(batch, min_date, max_date):
    min_dekad = dekad_from_date(min_date)
    max_dekad = dekad_from_date(max_date)
    dekads = list(range(0, max_dekad - min_dekad + 1))
    for key in TIME_SERIES_PREDICTORS:
        value = batch[key]
        # Transform dates to dekads
        date_strs = [str(date) for date in batch[KEY_DATES][key]]
        value_dekads = torch.tensor(
            [dekad_from_date(date) for date in date_strs], device=value.device
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
                if key in ["tmax"]:  # Max aggregation
                    new_value[:, d] = value[:, mask].max(dim=1).values
                elif key in ["tmin"]:  # Min aggregation
                    new_value[:, d] = value[:, mask].min(dim=1).values
                else:  # for all other inputs
                    new_value[:, d] = torch.mean(value[:, mask], dim=1)

        batch[key] = new_value

    return batch
