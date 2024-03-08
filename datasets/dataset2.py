
import pandas as pd

from data_preparation.county_us import get_meteo_data, get_soil_data, get_remote_sensing_data


class Dataset:

    # Key used for the location index
    KEY_LOC = 'loc_id'  # TODO -- should be defined outside this class!! Global property of project
    # Key used for the year index
    KEY_YEAR = 'year'
    # Key used for yield targets
    KEY_TARGET = 'yield'

    def __init__(self,
                 data_target: pd.DataFrame = None,
                 data_features: list = None,
                 ):
        """
        Dataset class for regional yield forecasting

        Targets/features are provided using properly formatted pandas dataframes.

        :param data_target: pandas.DataFrame that contains yield statistics
                            Dataframe should meet the following requirements:
                                - The column containing yield targets should be named properly
                                  Expected column name is stored in `Dataset.KEY_TARGET`
                                - The dataframe is indexed by (location id, year) using the correct naming
                                  Expected names are stored in `Dataset.KEY_LOC`, `Dataset.KEY_YEAR`, resp.
        :param data_features: list of pandas.Dataframe objects each containing features
                            Dataframes should meet the following requirements:
                                - Columns should be named by their respective feature names
                                - Dataframes cannot have overlapping column (i.e. feature) names
                                - Each dataframe can be indexed in three different ways:
                                    - By location only -- for static location features
                                    - By location and year -- for yearly occurring features
                                    - By location, year, and some extra level assumed to be temporal (e.g. daily,
                                      dekadal, ...)
                                  The index levels should be named properly, i.e.
                                    - `Dataset.KEY_LOC` for the location
                                    - `Dataset.KEY_YEAR` for the year
                                    - the name of the extra optional temporal level is ignored and has no requirement
        """
        # If no data is given, create an empty dataset
        if data_target is None:
            data_target = self._empty_df_target()
        if data_features is None:
            data_features = list()

        # Make sure columns are named properly
        if len(data_features) > 0:
            column_names = set.union(*[set(df.columns) for df in data_features])
            assert Dataset.KEY_LOC not in column_names
            assert Dataset.KEY_YEAR not in column_names
            assert Dataset.KEY_TARGET not in column_names

        # Make sure there are no overlaps in feature names
        if len(data_features) > 1:
            assert len(set.intersection(*[set(df.columns) for df in data_features])) == 0

        # Make sure the individual dataframes are properly configured
        assert self._validate_df_feature(data_target)
        assert all([self._validate_df_feature(df) for df in data_features])

        self._df_y = data_target
        self._dfs_x = list(data_features)

        # Sort all data for faster lookups
        self._df_y.sort_index(inplace=True)
        for df in self._dfs_x:
            df.sort_index(inplace=True)

        self._allow_incomplete = False

    @property
    def years(self) -> set:
        """
        Obtain a set containing all years occurring in the dataset
        """
        return set([year for _, year in self._df_y.index.values])

    @property
    def location_ids(self) -> set:
        """
        Obtain a set containing all location ids occurring in the dataset
        """
        return set([loc for loc, _ in self._df_y.index.values])

    @property
    def feature_names(self) -> set:
        """
        Obtain a set containing all feature names
        """
        return set.union(*[set(df.columns) for df in self._dfs_x])

    def __getitem__(self, index) -> dict:
        """
        Get a single data point in the dataset

        Data point is returned as a dict

        :param index: index for accessing the data. Can be an int or the (location, year) that specify the data
        :return:
        """
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

        # Get the target label for the specified sample
        sample = {
            Dataset.KEY_YEAR: year,
            Dataset.KEY_LOC: loc_id,
            Dataset.KEY_TARGET: sample_y[Dataset.KEY_TARGET],
        }

        # Get feature data corresponding to the label
        data_x = self._get_feature_data(loc_id, year)
        # Merge label and feature data
        sample = {** data_x, **sample}

        return sample

    def __len__(self) -> int:
        """
        Get the number of samples in the dataset
        """
        return len(self._df_y)

    def __iter__(self):
        """
        Iterate through the samples in the dataset
        """
        for i in range(len(self)):
            yield self[i]

    def _get_feature_data(self, loc_id: int, year: int) -> dict:
        """
        Helper function for obtaining feature data corresponding to some index
        :param loc_id: location index value
        :param year: year index value
        :return:
        """
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

