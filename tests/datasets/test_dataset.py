import os
import pandas as pd

from datasets.dataset import Dataset
from config import PATH_DATA_DIR
from config import KEY_LOC, KEY_YEAR, KEY_TARGET, KEY_DATES

data_path = os.path.join(PATH_DATA_DIR, "data_US", "county_data")
yield_csv = os.path.join(data_path, "YIELD_COUNTY_US.csv")
yield_df = pd.read_csv(yield_csv, index_col=[KEY_LOC, KEY_YEAR])
dataset = Dataset(data_target=yield_df, data_inputs=[])


def test_dataset_length():
    assert len(dataset) == len(yield_df.index)


def test_dataset_item():
    assert isinstance(dataset[0], dict)
    assert len(dataset[0]) == len([KEY_LOC, KEY_YEAR, KEY_TARGET, KEY_DATES])
    assert set(dataset[0].keys()) == set([KEY_LOC, KEY_YEAR, KEY_TARGET, KEY_DATES])


def test_split():
    data_path_county_features = os.path.join(
        PATH_DATA_DIR, "data_US", "county_features"
    )
    train_csv = os.path.join(data_path_county_features, "grain_maize_US_train.csv")
    train_df = pd.read_csv(train_csv, index_col=[KEY_LOC, KEY_YEAR])
    train_yields = train_df[[KEY_TARGET]].copy()
    feature_cols = [c for c in train_df.columns if c != KEY_TARGET]
    train_features = train_df[feature_cols].copy()
    dataset_cv = Dataset(train_yields, [train_features])

    even_years = {x for x in dataset_cv.years if x % 2 == 0}
    odd_years = dataset_cv.years - even_years

    ds1, ds2 = dataset_cv.split_on_years((even_years, odd_years))
    assert ds1.years == even_years
    assert ds2.years == odd_years
