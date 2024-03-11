import os
import pandas as pd

from datasets.dataset import Dataset
from config import PATH_DATA_DIR
from config import KEY_LOC, KEY_YEAR, KEY_TARGET

data_path = os.path.join(PATH_DATA_DIR, "data_US", "county_data")
yield_csv = os.path.join(data_path, "YIELD_COUNTY_US.csv")
yield_df = pd.read_csv(yield_csv, index_col=[KEY_LOC, KEY_YEAR])
dataset = Dataset(data_target=yield_df, data_features=[])


def test_dataset_length():
    assert len(dataset) == len(yield_df.index)


def test_dataset_item():
    assert isinstance(dataset[0], dict)
    assert len(dataset[0]) == 3
    assert set(dataset[0].keys()) == set([KEY_LOC, KEY_YEAR, KEY_TARGET])
