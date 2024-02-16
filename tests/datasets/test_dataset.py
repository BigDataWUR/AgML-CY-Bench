import os
import pandas as pd

from datasets.dataset import Dataset
from config import PATH_DATA_DIR

data_path = os.path.join(PATH_DATA_DIR, "data_US", "county_data")
yield_csv = os.path.join(data_path, "YIELD_COUNTY_US.csv")
data_sources = {
    "YIELD": {
        "index_cols": ["COUNTY_ID", "FYEAR"],
        "sel_cols": ["YIELD"],
    }
}

yield_df = pd.read_csv(yield_csv, index_col=data_sources["YIELD"]["index_cols"])
data_dfs = {"YIELD": yield_df}

dataset = Dataset(data_dfs, data_sources, "COUNTY_ID", "FYEAR")


def test_dataset_length():
    assert len(dataset) == len(yield_df.index)


def test_dataset_item():
    assert isinstance(dataset[0], dict)
    assert len(dataset[0]) == 3
    assert list(dataset[0].keys()) == ["COUNTY_ID", "FYEAR", "YIELD"]
