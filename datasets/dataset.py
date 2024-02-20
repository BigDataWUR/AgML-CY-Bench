import pandas as pd
from pandas import MultiIndex

from data_preparation.county_us import get_yield_data, get_meteo_data, get_soil_data, get_remote_sensing_data

class Dataset:
    def __init__(
        self,
        data_dfs,
        data_sources,
        spatial_id_col="REGION",
        year_col="YEAR",
    ):
        self._data_sources = data_sources
        self._spatial_id_col = spatial_id_col
        self._year_col = year_col
        self._index_cols = data_sources["YIELD"]["index_cols"]
        self._data_dfs = data_dfs
        # Sort the data for faster lookups
        for src in self._data_dfs:
            self._data_dfs[src].sort_index(inplace=True)

        self._data_y = data_dfs["YIELD"]

    def __getitem__(self, index) -> dict:
        # Index is either integer or tuple of (location, year)
        if isinstance(index, int):
            sample_y = self._data_y.iloc[index]
            spatial_unit, year = sample_y.name

        elif isinstance(index, tuple):
            spatial_unit, year = index
            sample_y = self._data_y.loc[index]

        else:
            raise Exception(f"Unsupported index type {type(index)}")

        return {
            self._spatial_id_col: spatial_unit,
            self._year_col: year,
            "YIELD": sample_y["YIELD"],
        }

    def __len__(self) -> int:
        return len(self._data_y)

    def split_on_years(self, years_split: tuple) -> tuple:

        df_yield_1, df_yield_2 = self._split_df_on_index(self._data_y, years_split, level=1)
        df_meteo_1, df_meteo_2 = self._split_df_on_index(self._data_meteo, years_split, level=1)
        df_rs_1, df_rs_2 = self._split_df_on_index(self._data_remote_sensing, years_split, level=1)

        return (
            Dataset(
                df_yield=df_yield_1,
                df_meteo=df_meteo_1,
                df_soil=self._data_soil,  # No split required
                df_remote_sensing=df_rs_1,
            ),
            Dataset(
                df_yield=df_yield_2,
                df_meteo=df_meteo_2,
                df_soil=self._data_soil,  # No split required
                df_remote_sensing=df_rs_2,
            ),
        )

    def split_on_county_ids(self, county_ids_split: tuple) -> tuple:

        df_yield_1, df_yield_2 = self._split_df_on_index(self._data_y, county_ids_split, level=0)
        df_meteo_1, df_meteo_2 = self._split_df_on_index(self._data_meteo, county_ids_split, level=0)
        df_soil_1, df_soil_2 = self._split_df_on_index(self._data_soil, county_ids_split, level=0)
        df_rs_1, df_rs_2 = self._split_df_on_index(self._data_remote_sensing, county_ids_split, level=0)

        return (
            Dataset(
                df_yield=df_yield_1,
                df_meteo=df_meteo_1,
                df_soil=df_soil_1,
                df_remote_sensing=df_rs_1,
            ),
            Dataset(
                df_yield=df_yield_2,
                df_meteo=df_meteo_2,
                df_soil=df_soil_2,
                df_remote_sensing=df_rs_2,
            ),
        )

    @staticmethod
    def _split_df_on_index(df: pd.DataFrame, split: tuple, level: int):
        df.sort_index(inplace=True)

        keys1, keys2 = split

        df_1 = Dataset._filter_df_on_index(df, keys1, level)
        df_2 = Dataset._filter_df_on_index(df, keys2, level)

        return df_1, df_2

    @staticmethod
    def _filter_df_on_index(df: pd.DataFrame, keys: list, level: int):
        if not isinstance(df.index, MultiIndex):
            return df.loc[keys]
        else:
            return pd.concat(
                [df.xs(key, level=level, drop_level=False) for key in keys]
            )

    @staticmethod
    def train_test_datasets() -> tuple:  # TODO -- define the splits
        dataset = Dataset()
        return dataset.split_on_years(
            years_split=(Dataset.YEARS_TRAIN, Dataset.YEARS_TEST),
        )

    def get_feature_mean(self, key: str) -> float:  # TODO
        raise NotImplementedError

    def get_feature_std(self, key: str) -> float:
        raise NotImplementedError

    def get_feature_range(self, key: str) -> tuple:
        raise NotImplementedError

    @property
    def years(self) -> list:
        return list(set([year for _, year in self._data_y.index.values]))

    @property
    def indexCols(self) -> list:
        return self._index_cols

    @property
    def labelCol(self) -> str:
        return "YIELD"

if __name__ == '__main__':

    _dataset = Dataset()

    # print(_dataset.years)
    # print(_dataset.locations)
    print(_dataset['AL_LAWRENCE', 2000])

    dataset_train, dataset_test = Dataset.train_test_datasets()

    print(dataset_train.years)

    dataset_train, dataset_validation = dataset_train.split_on_years(([2000], [2001]))
