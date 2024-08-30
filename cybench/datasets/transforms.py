import torch
from cybench.util.features import dekad_from_date
from cybench.config import (
    KEY_DATES,
    TIME_SERIES_PREDICTORS,
)


def transform_ts_inputs_to_dekadal(batch, min_date, max_date):
    min_dekad = dekad_from_date(str(min_date))
    max_dekad = dekad_from_date(str(max_date))
    dekads = list(range(0, max_dekad - min_dekad + 1))
    for key in TIME_SERIES_PREDICTORS:
        value = batch[key]
        # Transform dates to dekads
        value_dekads = torch.tensor(
            [dekad_from_date(str(date)) for date in batch[KEY_DATES][key]], device=value.device
        )
        value_dekads -= 1

        # Aggregate timeseries to dekadal resolution
        new_value = torch.full(
            (value.shape[0], len(dekads)), 0.0, dtype=value.dtype, device=value.device
        )
        for d in dekads:
            mask = value_dekads == d
            if value[:, mask].shape[1] == 0:
                continue

            # Inefficient, but scatter_min or scatter_max are not supported by torch
            if key in ["tmax"]:  # Max aggregation
                new_value[:, d] = value[:, mask].max(dim=1).values
            elif key in ["tmin"]:  # Min aggregation
                new_value[:, d] = value[:, mask].min(dim=1).values
            elif key in ["prec", "cwb"]:
                new_value[:, d] = torch.sum(value[:, mask], dim=1)
            else:  # for all other inputs
                new_value[:, d] = torch.mean(value[:, mask], dim=1)

        batch[key] = new_value

    return batch
