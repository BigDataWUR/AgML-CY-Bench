import torch
import torch.utils.data

from datasets.dataset import AgMLDataset
from util.torch import batch_tensors


class TorchDataset(torch.utils.data.Dataset):

    def __init__(self, dataset: AgMLDataset):
        self._dataset = dataset

    def __getitem__(self, index):
        return self._cast_to_tensor(
            self._dataset[index],
        )

    def __len__(self):
        return len(self._dataset)

    @classmethod
    def _cast_to_tensor(cls, sample: dict) -> dict:
        return {
            **{key: sample[key] for key in AgMLDataset.INDEX_KEYS},
            **{key: torch.tensor(sample[key]) for key in sample.keys() if key not in AgMLDataset.INDEX_KEYS}
        }

    @classmethod
    def collate_fn(cls, samples: list) -> dict:
        key_y = AgMLDataset.TARGET_KEY

        batched_samples = {
            **{key: [sample[key] for sample in samples] for key in AgMLDataset.INDEX_KEYS},
            **{key: batch_tensors(*[sample[key] for sample in samples]) for key in AgMLDataset.FEATURE_KEYS},
            key_y: batch_tensors(*[sample[key_y] for sample in samples]),
        }

        return batched_samples

    # TODO -- define transform for normalization

    # @classmethod
    # def _normalize(cls, sample: dict, parameters: dict = None,) -> dict:
    #     raise NotImplementedError
    #
    # def get_normalization_parameters(self) -> dict:
    #     raise NotImplementedError

    pass


if __name__ == '__main__':

    _dataset = AgMLDataset()

    # print(_dataset.years)
    # print(_dataset.locations)
    print(_dataset['AL_LAWRENCE', 2000])

    _dataset_train, _dataset_test = AgMLDataset.train_test_datasets()

    _dataset_train_torch = TorchDataset(_dataset_train)

    print(_dataset_train_torch['AL_LAWRENCE', 2000])

    import torch.utils.data
    _dataloader = torch.utils.data.DataLoader(
        _dataset_train_torch,
        collate_fn=_dataset_train_torch.collate_fn,
        shuffle=True,
        batch_size=2,
    )

    for _batch in _dataloader:

        print(_batch)
        print(_batch['YIELD'].shape)
        print(_batch['TMAX'].shape)

        break
