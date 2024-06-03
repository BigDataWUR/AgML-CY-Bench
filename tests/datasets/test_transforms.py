from datasets.dataset import Dataset
from datasets.dataset_torch import TorchDataset
from datasets.transforms import transform_ts_features_to_dekadal, transform_stack_ts_static_features

def test_transforms():
    train_dataset = Dataset.load("maize_NL")
    train_dataset = TorchDataset(train_dataset)
    batch = [train_dataset[i] for i in range(16)]
    batch = TorchDataset.collate_fn(batch).copy()
    transforms = [transform_ts_features_to_dekadal, transform_stack_ts_static_features]
    for transform in transforms: batch = transform(batch)
