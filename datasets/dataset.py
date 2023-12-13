import pandas as pd
from pandas import MultiIndex

from data_preparation.county_us import get_yield_data, get_meteo_data, get_soil_data, get_remote_sensing_data


class AgMLDataset:

    YEARS_TRAIN = [2000, 2001]  # TODO -- define
    YEARS_TEST = [2002]

    INDEX_NAME = ('COUNTY_ID', 'FYEAR')

    def __init__(self,
                 df_yield: pd.DataFrame = None,
                 df_meteo: pd.DataFrame = None,
                 df_soil: pd.DataFrame = None,
                 df_remote_sensing: pd.DataFrame = None,
                 ):

        # Load the data from disk if necessary
        if df_yield is None:
            df_yield = get_yield_data()
        if df_meteo is None:
            df_meteo = get_meteo_data()
        if df_soil is None:
            df_soil = get_soil_data()
        if df_remote_sensing is None:
            df_remote_sensing = get_remote_sensing_data()

        # TODO -- align data
        # For now, only use counties that are present everywhere
        counties_yield = set(df_yield.index.get_level_values(0))
        counties_soil = set(df_soil.index.values)
        counties_meteo = set(df_meteo.index.get_level_values(0))
        counties_rs = set(df_remote_sensing.index.get_level_values(0))
        counties = set.intersection(counties_yield, counties_soil, counties_meteo, counties_rs)

        df_yield = AgMLDataset._filter_df_on_index(df_yield, list(counties), level=0)
        df_soil = AgMLDataset._filter_df_on_index(df_soil, list(counties), level=0)
        df_meteo = AgMLDataset._filter_df_on_index(df_meteo, list(counties), level=0)
        df_remote_sensing = AgMLDataset._filter_df_on_index(df_remote_sensing, list(counties), level=0)

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
        return AgMLDataset.feature_keys_meteo() + AgMLDataset.feature_keys_soil() + AgMLDataset.feature_keys_remote_sensing()

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
            AgMLDataset(
                df_yield=df_yield_1,
                df_meteo=df_meteo_1,
                df_soil=self._data_soil,  # No split required
                df_remote_sensing=df_rs_1,
            ),
            AgMLDataset(
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
            AgMLDataset(
                df_yield=df_yield_1,
                df_meteo=df_meteo_1,
                df_soil=df_soil_1,
                df_remote_sensing=df_rs_1,
            ),
            AgMLDataset(
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

        df_1 = AgMLDataset._filter_df_on_index(df, keys1, level)
        df_2 = AgMLDataset._filter_df_on_index(df, keys2, level)

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
        dataset = AgMLDataset()
        return dataset.split_on_years(
            years_split=(AgMLDataset.YEARS_TRAIN, AgMLDataset.YEARS_TEST),
        )


if __name__ == '__main__':

    _dataset = AgMLDataset()

    # print(_dataset.years)
    # print(_dataset.locations)
    print(_dataset['AL_LAWRENCE', 2000])

    dataset_train, dataset_test = AgMLDataset.train_test_datasets()

    print(dataset_train.years)

    dataset_train, dataset_validation = dataset_train.split_on_years(([2000], [2001]))
