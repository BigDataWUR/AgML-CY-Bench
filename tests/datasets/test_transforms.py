from datasets.dataset import Dataset
from datasets.dataset_torch import TorchDataset
from datasets.transforms import (
    transform_ts_features_to_dekadal,
    transform_stack_ts_static_features,
)


# TODO: Uncomment after TorchDataset handles
# different number of time steps for time series data.
# Number of time steps can vary between sources and within a source.
# Same goes for tests.models.test_models test_nn_model()

# def test_transforms():
#    train_dataset = Dataset.load("maize_NL")
#    train_dataset = TorchDataset(train_dataset)
#    batch = [train_dataset[i] for i in range(16)]
#    batch = TorchDataset.collate_fn(batch).copy()
#    transforms = [transform_ts_features_to_dekadal, transform_stack_ts_static_features]
#    for transform in transforms: batch = transform(batch)
