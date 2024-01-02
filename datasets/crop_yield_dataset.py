import os
import pandas as pd
from pandas import MultiIndex

from datasets.dataset import AgMLBaseDataset

class CropYieldDataset(AgMLBaseDataset):

    YEARS_TRAIN = [2000, 2001]  # TODO -- define
    YEARS_TEST = [2002]

    INDEX_NAME = ('COUNTY_ID', 'FYEAR')

    def __init__(self, data_dfs=None, data_path=None, data_sources=None):
        # Load the data from disk if necessary
        if (data_dfs is None):
            data_dfs = {}
            for src in data_sources:
                filename = data_sources[src]["filename"]
                index_cols = data_sources[src]["index_cols"]
                src_df = CropYieldDataset.get_data(data_path, filename, index_cols)
                data_dfs[src] = src_df

        df_yield = data_dfs["YIELD"]
        df_meteo = data_dfs["METEO"]
        df_soil = data_dfs["SOIL"]
        df_remote_sensing = data_dfs["REMOTE_SENSING"]

        # TODO -- align data
        # For now, only use counties that are present everywhere
        counties_yield = set(df_yield.index.get_level_values(0))
        counties_soil = set(df_soil.index.values)
        counties_meteo = set(df_meteo.index.get_level_values(0))
        counties_rs = set(df_remote_sensing.index.get_level_values(0))
        counties = set.intersection(counties_yield, counties_soil, counties_meteo, counties_rs)

        df_yield = CropYieldDataset._filter_df_on_index(df_yield, list(counties), level=0)
        df_soil = CropYieldDataset._filter_df_on_index(df_soil, list(counties), level=0)
        df_meteo = CropYieldDataset._filter_df_on_index(df_meteo, list(counties), level=0)
        df_remote_sensing = CropYieldDataset._filter_df_on_index(df_remote_sensing, list(counties), level=0)

        # TODO -- data preprocessing:
        #   - Start date, end date of season?
        #   - Missing data?

        self._data_y = df_yield  # Filter from raw data based on inputs/selection

        self._data_soil = df_soil
        self._data_meteo = df_meteo
        self._data_remote_sensing = df_remote_sensing

        # Sort the data for faster lookups
        self._data_y.sort_index(inplace=True)
        self._data_soil.sort_index(inplace=True)
        self._data_meteo.sort_index(inplace=True)
        self._data_remote_sensing.sort_index(inplace=True)

    @property
    def years(self) -> list:
        """
        Obtain a list containing all years occurring in the dataset
        """
        return list(set([year for _, year in self._data_y.index.values]))

    @property
    def county_ids(self) -> list:
        """
        Obtain a list containing all county ids occurring in the dataset
        """
        return list(set([loc for loc, _ in self._data_y.index.values]))

    @staticmethod
    def target_key() -> str:
        return 'YIELD'

    @staticmethod
    def feature_keys() -> tuple:
        return CropYieldDataset.feature_keys_meteo() + CropYieldDataset.feature_keys_soil() + CropYieldDataset.feature_keys_remote_sensing()

    @staticmethod
    def feature_keys_meteo() -> tuple:
        return 'TMAX', 'TMIN', 'TAVG', 'VPRES', 'WSPD', 'PREC', 'ET0', 'RAD'

    @staticmethod
    def feature_keys_soil() -> tuple:
        return 'SM_WHC', 'SM_DEPTH'

    @staticmethod
    def feature_keys_remote_sensing() -> tuple:
        return 'FAPAR',

    def __getitem__(self, index) -> dict:
        # Index is either integer or tuple of (year, location)

        if isinstance(index, int):
            sample_y = self._data_y.iloc[index]
            county_id, year = sample_y.name

        elif isinstance(index, tuple):
            county_id, year = index
            sample_y = self._data_y.loc[index]

        else:
            raise Exception(f'Unsupported index type {type(index)}')

        data_meteo = self._get_meteo_data(county_id, year)
        data_soil = self._get_soil_data(county_id)
        data_remote_sensing = self._get_remote_sensing_data(county_id, year)

        return {
            'FYEAR': year,
            'COUNTY_ID': county_id,
            'YIELD': sample_y.YIELD,
            **data_meteo,
            **data_soil,
            **data_remote_sensing,
        }

    def _get_meteo_data(self, county_id: str, year: int) -> dict:

        # Select data matching the location and year
        # Sort the index for improved lookup speed
        df = self._data_meteo.xs((county_id, year), drop_level=True)

        # Return the data as dict mapping
        #  key -> np.ndarray
        #  where the array contains data for all DEKADs
        return {
            key: df[key].values for key in self.feature_keys_meteo()
        }

    def _get_soil_data(self, county_id: str) -> dict:

        # Select the data matching the location
        df = self._data_soil.loc[county_id]

        return {
            key: df[key] for key in self.feature_keys_soil()
        }

    def _get_remote_sensing_data(self, county_id: str, year: int) -> dict:
        # Select data matching the location and year
        # Sort the index for improved lookup speed
        df = self._data_remote_sensing.xs((county_id, year), drop_level=True)

        # Return the data as dict mapping
        #  key -> np.ndarray
        #  where the array contains data for all DEKADs
        return {
            key: df[key].values for key in self.feature_keys_remote_sensing()
        }

    def __len__(self) -> int:
        return len(self._data_y)

    def split_on_years(self, years_split: tuple) -> tuple:

        df_yield_1, df_yield_2 = self._split_df_on_index(self._data_y, years_split, level=1)
        df_meteo_1, df_meteo_2 = self._split_df_on_index(self._data_meteo, years_split, level=1)
        df_rs_1, df_rs_2 = self._split_df_on_index(self._data_remote_sensing, years_split, level=1)

        return (
            CropYieldDataset(data_dfs={
                "YIELD" : df_yield_1,
                "METEO" : df_meteo_1,
                "SOIL" : self._data_soil.copy(), # No split required
                "REMOTE_SENSING" : df_rs_1,
            }),
            CropYieldDataset(data_dfs={
                "YIELD" : df_yield_2,
                "METEO" : df_meteo_2,
                "SOIL" : self._data_soil.copy(), # No split required
                "REMOTE_SENSING" : df_rs_2,
            }),
        )

    def split_on_county_ids(self, county_ids_split: tuple) -> tuple:

        df_yield_1, df_yield_2 = self._split_df_on_index(self._data_y, county_ids_split, level=0)
        df_meteo_1, df_meteo_2 = self._split_df_on_index(self._data_meteo, county_ids_split, level=0)
        df_soil_1, df_soil_2 = self._split_df_on_index(self._data_soil, county_ids_split, level=0)
        df_rs_1, df_rs_2 = self._split_df_on_index(self._data_remote_sensing, county_ids_split, level=0)

        return (
            CropYieldDataset(data_dfs={
                "YIELD" : df_yield_1,
                "METEO" : df_meteo_1,
                "SOIL" : df_soil_1,
                "REMOTE_SENSING" : df_rs_1,
            }),
            CropYieldDataset(data_dfs={
                "YIELD" : df_yield_2,
                "METEO" : df_meteo_2,
                "SOIL" : df_soil_2,
                "REMOTE_SENSING" : df_rs_2,
            }),
        )

    @staticmethod
    def get_data(data_path, filename, index_cols):
        path = os.path.join(data_path, filename)
        df = pd.read_csv(path, index_col=index_cols)
        return df

    @staticmethod
    def _split_df_on_index(df: pd.DataFrame, split: tuple, level: int):
        df.sort_index(inplace=True)

        keys1, keys2 = split

        df_1 = CropYieldDataset._filter_df_on_index(df, keys1, level)
        df_2 = CropYieldDataset._filter_df_on_index(df, keys2, level)

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
    def train_test_datasets(data_path, data_sources) -> tuple:  # TODO -- define the splits
        dataset = CropYieldDataset(data_path=data_path, data_sources=data_sources)
        return dataset.split_on_years(
            years_split=(CropYieldDataset.YEARS_TRAIN, CropYieldDataset.YEARS_TEST),
        )

from config import PATH_DATA_DIR

if __name__ == '__main__':
    data_path = os.path.join(PATH_DATA_DIR, "data_US", "county_data")
    data_sources = {
        "YIELD" : {
            "filename" : "YIELD_COUNTY_US.csv",
            "index_cols" : ["COUNTY_ID", "FYEAR"],
        },
        "METEO" : {
            "filename" : "METEO_COUNTY_US.csv",
            "index_cols" : ["COUNTY_ID", "FYEAR", "DEKAD"],
        },
        "SOIL" : {
            "filename" : "SOIL_COUNTY_US.csv",
            "index_cols" : ["COUNTY_ID"],
        },
        "REMOTE_SENSING" : {
            "filename" : "REMOTE_SENSING_COUNTY_US.csv",
            "index_cols" : ["COUNTY_ID", "FYEAR", "DEKAD"],
        }
    }

    _dataset = CropYieldDataset(data_path=data_path, data_sources=data_sources)

    # print(_dataset.years)
    # print(_dataset.locations)
    print(_dataset['AL_LAWRENCE', 2000])

    dataset_train, dataset_test = CropYieldDataset.train_test_datasets(data_path, data_sources)

    print(dataset_train.years)

    dataset_train, dataset_validation = dataset_train.split_on_years(([2000], [2001]))