
import pandas as pd

from data_preparation.county_us import get_meteo_data, get_soil_data, get_remote_sensing_data


class Dataset:

    KEY_LOC = 'loc_id'  # TODO -- should be defined outside this class!! Global property of project
    KEY_YEAR = 'year'
    KEY_TARGET = 'yield'

    def __init__(self,
                 data_target: pd.DataFrame = None,
                 data_features: list = None,
                 ):
        if data_target is None:
            data_target = self._empty_df_target()
        if data_features is None:
            data_features = list()

        # Make sure there are no overlaps in feature names
        if len(data_features) > 0:
            assert len(set.intersection(*[set(df.columns) for df in data_features])) == 0
            column_names = set.union(*[set(df.columns) for df in data_features])
            assert Dataset.KEY_LOC not in column_names
            assert Dataset.KEY_YEAR not in column_names
            assert Dataset.KEY_TARGET not in column_names

        # Make sure the individual dataframes are properly configured
        assert self._validate_df_feature(data_target)
        assert all([self._validate_df_feature(df) for df in data_features])

        self._df_y = data_target
        self._dfs_x = list(data_features)

        self._df_y.sort_index(inplace=True)
        for df in self._dfs_x:
            df.sort_index(inplace=True)

        self._allow_incomplete = False

    @property
    def years(self) -> list:
        """
        Obtain a list containing all years occurring in the dataset
        """
        return list(set([year for _, year in self._df_y.index.values]))

    @property
    def location_ids(self) -> list:
        """
        Obtain a list containing all location ids occurring in the dataset
        """
        return list(set([loc for loc, _ in self._df_y.index.values]))

    @property
    def feature_names(self) -> set:
        return set.union(*[set(df.columns) for df in self._dfs_x])

    def __getitem__(self, index) -> dict:
        # Index is either integer or tuple of (year, location)

        if isinstance(index, int):
            sample_y = self._df_y.iloc[index]
            loc_id, year = sample_y.name

        elif isinstance(index, tuple):
            assert len(index) == 2
            loc_id, year = index
            sample_y = self._df_y.loc[index]

        else:
            raise Exception(f'Unsupported index type {type(index)}')

        sample = {
            Dataset.KEY_YEAR: year,
            Dataset.KEY_LOC: loc_id,
            Dataset.KEY_TARGET: sample_y[Dataset.KEY_TARGET],
        }

        data_x = self._get_feature_data(loc_id, year)
        sample = {** data_x, **sample}

        return sample

    def __len__(self) -> int:
        return len(self._df_y)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def _get_feature_data(self, loc_id: int, year: int) -> dict:
        data = dict()
        for df in self._dfs_x:
            n_levels = len(df.index.names)
            assert 1 <= n_levels <= 3
            if n_levels == 1:
                if self._allow_incomplete:
                    if loc_id not in df.index:
                        continue

                data = {
                    **df.loc[loc_id].to_dict(),
                    **data,
                }

            if n_levels == 2:
                if self._allow_incomplete:
                    if (loc_id, year) not in df.index:
                        continue
                data = {
                    **df.loc[loc_id, year].to_dict(),
                    **data,
                }

            if n_levels == 3:
                if self._allow_incomplete:
                    if loc_id not in df.index.get_level_values(0) or year not in df.index.get_level_values(1):
                        continue

                # Select data matching the location and year
                # Sort the index for improved lookup speed
                df_loc = df.xs((loc_id, year), drop_level=True)

                data_loc = {
                    key: df_loc[key].values for key in df_loc.columns
                }

                data = {
                    **data_loc,
                    **data,
                }

        return data

    @staticmethod
    def _empty_df_target() -> pd.DataFrame:
        df = pd.DataFrame(
            index=pd.MultiIndex.from_arrays(([], []), names=[Dataset.KEY_LOC, Dataset.KEY_YEAR]),
            columns=[Dataset.KEY_TARGET],
        )
        return df

    @staticmethod
    def _validate_df_target(df: pd.DataFrame) -> bool:
        return True  # TODO

    @staticmethod
    def _validate_df_feature(df: pd.DataFrame) -> bool:
        return True  # TODO

    @staticmethod
    def _filter_df_on_index(df: pd.DataFrame, keys: list, level: int):
        if not isinstance(df.index, pd.MultiIndex):
            return df.loc[keys]
        else:
            return pd.concat(
                [df.xs(key, level=level, drop_level=False) for key in keys]
            )

    @staticmethod
    def get_datasets() -> tuple:  # TODO -- proper preprocessing
        from data_preparation.county_us import get_yield_data

        df_target = get_yield_data()

        df_target.rename(
            columns={'YIELD': Dataset.KEY_TARGET},
            inplace=True,
        )
        df_target.index.rename([Dataset.KEY_LOC, Dataset.KEY_YEAR], inplace=True)

        df_meteo = get_meteo_data()

        df_meteo.index.rename([Dataset.KEY_LOC, Dataset.KEY_YEAR, df_meteo.index.names[2]], inplace=True)

        df_soil = get_soil_data()

        df_soil.index.rename(Dataset.KEY_LOC, inplace=True)

        df_remote_sensing = get_remote_sensing_data()

        df_remote_sensing.index.rename([Dataset.KEY_LOC, Dataset.KEY_YEAR, df_remote_sensing.index.names[2]], inplace=True)

        years = list(range(2010, 2011))

        df_target = Dataset._filter_df_on_index(df_target, list(years), level=1)
        df_meteo = Dataset._filter_df_on_index(df_meteo, list(years), level=1)
        df_remote_sensing = Dataset._filter_df_on_index(df_remote_sensing, list(years), level=1)

        # For now, only use counties that are present everywhere
        counties_yield = set(df_target.index.get_level_values(0))
        counties_soil = set(df_soil.index.values)
        counties_meteo = set(df_meteo.index.get_level_values(0))
        counties_rs = set(df_remote_sensing.index.get_level_values(0))
        counties = set.intersection(counties_yield, counties_soil, counties_meteo, counties_rs)

        df_target = Dataset._filter_df_on_index(df_target, list(counties), level=0)
        df_soil = Dataset._filter_df_on_index(df_soil, list(counties), level=0)
        df_meteo = Dataset._filter_df_on_index(df_meteo, list(counties), level=0)
        df_remote_sensing = Dataset._filter_df_on_index(df_remote_sensing, list(counties), level=0)

        df_target.drop_duplicates(keep='first', inplace=True)
        df_soil.drop_duplicates(keep='first', inplace=True)
        df_meteo.drop_duplicates(keep='first', inplace=True)
        df_remote_sensing.drop_duplicates(keep='first', inplace=True)

        # YEARS_TRAIN = tuple(range(2000, 2011 + 1))
        # YEARS_TEST = tuple(range(2012, 2018 + 1))

        dataset = Dataset(
            data_target=df_target,
            data_features=[
                df_meteo,
                df_soil,
                df_remote_sensing,
            ]
        )

        return dataset, dataset

        # dataset_train = Dataset(
        #     data_target=Dataset._filter_df_on_index(df_target, YEARS_TRAIN, level=1),
        #     data_features=[
        #         Dataset._filter_df_on_index(df_meteo, YEARS_TRAIN, level=1),
        #         df_soil,
        #         Dataset._filter_df_on_index(df_remote_sensing, YEARS_TRAIN, level=1),
        #     ]
        # )
        #
        # dataset_test = Dataset(
        #     data_target=Dataset._filter_df_on_index(df_target, YEARS_TEST, level=1),
        #     data_features=[
        #         Dataset._filter_df_on_index(df_meteo, YEARS_TEST, level=1),
        #         df_soil,
        #         Dataset._filter_df_on_index(df_remote_sensing, YEARS_TEST, level=1),
        #     ]
        # )

        # return dataset_train, dataset_test


if __name__ == '__main__':

    # _df = pd.DataFrame.from_dict({Dataset.KEY_LOC: [], Dataset.KEY_YEAR: []}, orient='index', columns=[Dataset.KEY_TARGET])

    _dataset_train, _dataset_test = Dataset.get_datasets()

    print(_dataset_train[3])

    pass

