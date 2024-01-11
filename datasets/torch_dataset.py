import torch
import torch.utils.data

from datasets.dataset import CropYieldDataset
from util.torch import batch_tensors


class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: CropYieldDataset):
        self._dataset = dataset

    def __getitem__(self, index):
        return self._cast_to_tensor(self._dataset[index])

    def __len__(self):
        return len(self._dataset)

    def _cast_to_tensor(self, sample: dict) -> dict:
        index_cols = self._dataset.indexCols
        feature_cols = self._dataset.featureCols
        label_col = self._dataset.labelCol
        return {
            **{key: sample[key] for key in index_cols},
            **{key: torch.tensor(sample[key]) for key in feature_cols},
            label_col: torch.tensor(sample[label_col]),
        }

    def collate_fn(self, samples: list) -> dict:
        index_cols = self._dataset.indexCols
        feature_cols = self._dataset.featureCols
        label_col = self._dataset.labelCol
        batched_samples = {
            **{key: [sample[key] for sample in samples] for key in index_cols},
            **{
                key: batch_tensors(*[sample[key] for sample in samples])
                for key in feature_cols
            },
            label_col: batch_tensors(*[sample[label_col] for sample in samples]),
        }

        return batched_samples


import os

from config import PATH_DATA_DIR

if __name__ == "__main__":
    data_path = os.path.join(PATH_DATA_DIR, "data_US", "county_data")
    data_sources = {
        "YIELD": {
            "filename": "YIELD_COUNTY_US.csv",
            "index_cols": ["COUNTY_ID", "FYEAR"],
            "sel_cols": ["YIELD"],
        },
        "METEO": {
            "filename": "METEO_COUNTY_US.csv",
            "index_cols": ["COUNTY_ID", "FYEAR", "DEKAD"],
            "sel_cols": ["TMAX", "TMIN", "TAVG", "PREC", "ET0", "RAD"],
        },
        "SOIL": {
            "filename": "SOIL_COUNTY_US.csv",
            "index_cols": ["COUNTY_ID"],
            "sel_cols": ["SM_WHC"],
        },
        "REMOTE_SENSING": {
            "filename": "REMOTE_SENSING_COUNTY_US.csv",
            "index_cols": ["COUNTY_ID", "FYEAR", "DEKAD"],
            "sel_cols": ["FAPAR"],
        },
    }

    _dataset = CropYieldDataset(
        data_sources,
        spatial_id_col="COUNTY_ID",
        year_col="FYEAR",
        data_path=data_path,
        lead_time=6,
    )
    # print(_dataset["AL_LAWRENCE", 2000])

    _dataset_torch = TorchDataset(_dataset)
    # print(_dataset_torch['AL_LAWRENCE', 2000])

    _dataloader = torch.utils.data.DataLoader(
        _dataset_torch,
        collate_fn=_dataset_torch.collate_fn,
        shuffle=True,
        batch_size=2,
    )

    for _batch in _dataloader:
        # print(_batch)
        print(_batch["YIELD"].shape)
        print(_batch["FAPAR"].shape)

        break
