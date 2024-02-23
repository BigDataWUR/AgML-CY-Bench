import os
import pandas as pd

from datasets.dataset import Dataset
from config import PATH_DATA_DIR

data_path = os.path.join(PATH_DATA_DIR, "data_US", "county_data")
yield_csv = os.path.join(data_path, "YIELD_COUNTY_US.csv")
yield_df = pd.read_csv(yield_csv, index_col=["COUNTY_ID", "FYEAR"])
dataset = Dataset(yield_df, feature_dfs=[])


def test_dataset_length():
    assert len(dataset) == len(yield_df.index)


def test_dataset_item():
    assert isinstance(dataset[0], dict)
    assert len(dataset[0]) == 3
    assert list(dataset[0].keys()) == ["COUNTY_ID", "FYEAR", "YIELD"]
