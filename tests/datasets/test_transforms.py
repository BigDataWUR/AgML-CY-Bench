from cybench.datasets.dataset import Dataset
from cybench.datasets.dataset_torch import TorchDataset
from cybench.datasets.transforms import (
    transform_ts_inputs_to_dekadal,
)

from cybench.config import TIME_SERIES_PREDICTORS


def test_transforms():
    train_dataset = Dataset.load("maize_NL")
    min_date = train_dataset.min_date
    max_date = train_dataset.max_date
    train_dataset = TorchDataset(train_dataset)
    batch = [train_dataset[i] for i in range(16)]
    batch = TorchDataset.collate_fn(batch).copy()
    transforms = [transform_ts_inputs_to_dekadal]
    for i, transform in enumerate(transforms):
        batch = transform(batch, min_date, max_date)
        # check time series transfomations
        if i == 0:
            num_dekads = None
            for ft_key in TIME_SERIES_PREDICTORS:
                if num_dekads is None:
                    num_dekads = batch[ft_key].shape[1]
                else:
                    assert batch[ft_key].shape[1] == num_dekads
