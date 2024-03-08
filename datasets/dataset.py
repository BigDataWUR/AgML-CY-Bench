import pandas as pd

from config import KEY_LOC, KEY_YEAR, KEY_TARGET

class Dataset:

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
                                  Expected column name is stored in `KEY_TARGET`
                                - The dataframe is indexed by (location id, year) using the correct naming
                                  Expected names are stored in `KEY_LOC`, `KEY_YEAR`, resp.
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
                                    - `KEY_LOC` for the location
                                    - `KEY_YEAR` for the year
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
            assert KEY_LOC not in column_names
            assert KEY_YEAR not in column_names
            assert KEY_TARGET not in column_names

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
            KEY_LOC: loc_id,
            KEY_YEAR: year,
            KEY_TARGET: sample_y[KEY_TARGET],
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
            index=pd.MultiIndex.from_arrays(([], []), names=[KEY_LOC, KEY_YEAR]),
            columns=[KEY_TARGET],
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