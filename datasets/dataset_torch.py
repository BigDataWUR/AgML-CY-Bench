import torch
import torch.utils.data

from config import KEY_LOC, KEY_YEAR, KEY_TARGET

from datasets.dataset import Dataset
from util.torch import batch_tensors


class TorchDataset(torch.utils.data.Dataset):

    def __init__(self, dataset: Dataset):
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
            KEY_LOC: sample[KEY_LOC],
            KEY_YEAR: sample[KEY_YEAR],
            KEY_TARGET: torch.tensor(sample[KEY_TARGET]),
            **{
                key: torch.tensor(sample[key]) for key in sample.keys() if key not in [KEY_LOC, KEY_YEAR, KEY_TARGET]
              }  # TODO -- support nonnumeric data?
        }

    @classmethod
    def collate_fn(cls, samples: list) -> dict:
        assert len(samples) > 0

        feature_names = set.intersection(*[set(sample.keys()) for sample in samples])
        feature_names.remove(KEY_TARGET)
        feature_names.remove(KEY_LOC)
        feature_names.remove(KEY_YEAR)

        batched_samples = {
            KEY_TARGET: batch_tensors(*[sample[KEY_TARGET] for sample in samples]),
            KEY_LOC: [sample[KEY_LOC] for sample in samples],
            KEY_YEAR: [sample[KEY_YEAR] for sample in samples],
            **{key: batch_tensors(*[sample[key] for sample in samples]) for key in feature_names}
        }

        # batched_samples = {
        #     **{key: [sample[key] for sample in samples] for key in Dataset.INDEX_KEYS},
        #     **{key: batch_tensors(*[sample[key] for sample in samples]) for key in Dataset.FEATURE_KEYS},
        #     key_y: batch_tensors(*[sample[key_y] for sample in samples]),
        # }

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

    _dataset_train, _dataset_test = Dataset.get_datasets()

    _dataset_train_torch = TorchDataset(_dataset_train)

    # print(_dataset_train_torch['AL_LAWRENCE', 2000])
    print(_dataset_train_torch[1])

    import torch.utils.data
    _dataloader = torch.utils.data.DataLoader(
        _dataset_train_torch,
        collate_fn=_dataset_train_torch.collate_fn,
        shuffle=True,
        batch_size=2,
    )

    for _batch in _dataloader:

        print(_batch)
        print(_batch['yield'].shape)
        print(_batch['TMAX'].shape)

        break
