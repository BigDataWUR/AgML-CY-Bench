import os
import pandas as pd
from pandas import MultiIndex

from util.data import csv_to_pandas


class CropYieldDataset:
    def __init__(
        self,
        data_sources,
        label_key="YIELD",
        spatial_id_col="REGION",
        year_col="YEAR",
        label_col="YIELD",
        data_dfs=None,
        data_path=None,
        combined_features_labels=False,
    ):
        assert (data_dfs is not None) or (data_path is not None)

        self._data_sources = data_sources
        self._label_key = label_key
        self._spatial_id_col = spatial_id_col
        self._year_col = year_col
        self._label_col = label_col
        self._combined_features = False

        # Check if data includes combined features and labels
        if combined_features_labels:
            self._combined_features = True
            index_cols = data_sources["COMBINED"]["index_cols"]
            self._index_cols = index_cols

            # Load the data from disk if necessary
            if data_dfs is None:
                filename = data_sources["COMBINED"]["filename"]
                data_df = csv_to_pandas(data_path, filename, index_cols)
            else:
                data_df = data_dfs["COMBINED"]

            self._data_df = data_df
            self._data_y = self._data_df[[self._label_col]]
            self._feature_cols = [c for c in data_df.columns if (c != self._label_col)]

        else:
            self._feature_cols = []
            if data_dfs is None:
                data_dfs = {}
                for src in data_sources:
                    filename = data_sources[src]["filename"]
                    index_cols = data_sources[src]["index_cols"]
                    src_df = csv_to_pandas(data_path, filename, index_cols)
                    data_dfs[src] = src_df

            self._time_series_cols = []
            for src in data_sources:
                if src == self._label_key:
                    index_cols = data_sources[src]["index_cols"]
                    self._index_cols = index_cols
                    continue

                sel_cols = data_sources[src]["sel_cols"]
                self._feature_cols += sel_cols
                if data_dfs[src].index.nlevels == 3:
                    self._time_series_cols += sel_cols

            # TODO -- align data
            # For now, only use counties that are present everywhere
            # TODO -- data preprocessing:
            #   - Start date, end date of season?
            #   - Missing data?

            self._data_dfs = data_dfs
            self._align_spatial_units()
            self._align_years()

            for src in self._data_dfs:
                # Sort the data for faster lookups
                self._data_dfs[src].sort_index(inplace=True)

            self._data_y = data_dfs[self._label_key]

    def _align_spatial_units(self):
        self._align_data(index_level=0)

    def _align_years(self):
        self._align_data(index_level=1)

    def _align_data(self, index_level):
        common_values = None
        for src in self._data_dfs:
            if index_level >= self._data_dfs[src].index.nlevels:
                continue

            src_values = set(self._data_dfs[src].index.get_level_values(index_level))
            if common_values is None:
                common_values = src_values
            else:
                common_values = set.intersection(common_values, src_values)

        for src in self._data_dfs:
            if index_level >= self._data_dfs[src].index.nlevels:
                continue

            src_df = CropYieldDataset._filter_df_on_index(
                self._data_dfs[src], list(common_values), level=index_level
            )
            self._data_dfs[src] = src_df

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

        data_item = {
            self._spatial_id_col: spatial_unit,
            self._year_col: year,
            self._label_col: sample_y[self._label_col],
        }

        if self._combined_features:
            df = self._data_df.xs((spatial_unit, year), drop_level=True)
            sample = {key: df[key] for key in self._feature_cols}
            data_item.update(**sample)
        else:
            for src in self._data_sources:
                if src == self._label_key:
                    continue

                if self._year_col in self._data_sources[src]["index_cols"]:
                    src_sample = self._get_data_sample(src, spatial_unit, year)
                else:
                    src_sample = self._get_data_sample(src, spatial_unit)

                data_item.update(**src_sample)

        return data_item

    def _get_data_sample(self, src, spatial_unit, year=None):
        sel_cols = self._data_sources[src]["sel_cols"]

        # static data, no year
        if year is None:
            df = self._data_dfs[src].loc[spatial_unit]
            sample = {key: df[key] for key in sel_cols}

        # time series data
        else:
            df = self._data_dfs[src].xs((spatial_unit, year), drop_level=True)
            sample = {key: df[key].values for key in sel_cols}

        return sample

    def __len__(self) -> int:
        return len(self._data_y)

    @property
    def indexCols(self) -> list:
        return self._index_cols

    @property
    def featureCols(self) -> list:
        return self._feature_cols

    @property
    def timeSeriesCols(self) -> list:
        return self._time_series_cols

    @property
    def labelCol(self) -> str:
        return self._label_col

    def split_on_years(self, years_split: tuple) -> tuple:
        if self._combined_features:
            df1, df2 = self._split_df_on_index(self._data_df, years_split, level=1)
            return (
                CropYieldDataset(
                    self._data_sources,
                    spatial_id_col=self._spatial_id_col,
                    year_col=self._year_col,
                    data_dfs={"COMBINED": df1},
                    combined_features_labels=True,
                ),
                CropYieldDataset(
                    self._data_sources,
                    spatial_id_col=self._spatial_id_col,
                    year_col=self._year_col,
                    data_dfs={"COMBINED": df2},
                    combined_features_labels=True,
                ),
            )
        else:
            data_dfs1 = {}
            data_dfs2 = {}
            for src in self._data_dfs:
                src_df = self._data_dfs[src]
                if self._year_col in self._data_sources[src]["index_cols"]:
                    df1, df2 = self._split_df_on_index(src_df, years_split, level=1)
                    data_dfs1[src] = df1
                    data_dfs2[src] = df2
                else:
                    data_dfs1[src] = src_df.copy()
                    data_dfs2[src] = src_df.copy()

            return (
                CropYieldDataset(
                    self._data_sources,
                    spatial_id_col=self._spatial_id_col,
                    year_col=self._year_col,
                    data_dfs=data_dfs1,
                ),
                CropYieldDataset(
                    self._data_sources,
                    spatial_id_col=self._spatial_id_col,
                    year_col=self._year_col,
                    data_dfs=data_dfs2,
                ),
            )

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


from config import PATH_DATA_DIR

if __name__ == "__main__":
    data_path = os.path.join(PATH_DATA_DIR, "data_US", "county_data")
    data_sources = {
        "YIELD": {
            "filename": "YIELD_COUNTY_US.csv",
            "index_cols": ["COUNTY_ID", "FYEAR"],
            "sel_cols": ["YIELD"],
        },
        "METEO": {
            "filename": "METEO_COUNTY_US.csv",
            "index_cols": ["COUNTY_ID", "FYEAR", "DEKAD"],
            "sel_cols": ["TMAX", "TMIN", "TAVG", "PREC", "ET0", "RAD"],
        },
        "SOIL": {
            "filename": "SOIL_COUNTY_US.csv",
            "index_cols": ["COUNTY_ID"],
            "sel_cols": ["SM_WHC"],
        },
        "REMOTE_SENSING": {
            "filename": "REMOTE_SENSING_COUNTY_US.csv",
            "index_cols": ["COUNTY_ID", "FYEAR", "DEKAD"],
            "sel_cols": ["FAPAR"],
        },
    }

    _dataset = CropYieldDataset(
        data_sources, spatial_id_col="COUNTY_ID", year_col="FYEAR", data_path=data_path
    )
    print(_dataset["AL_LAWRENCE", 2000])

    print("\n")
    print("Test dataset splits")
    dataset_train, dataset_test = _dataset.split_on_years(([2000], [2001]))
    print(dataset_train["AL_LAWRENCE", 2000])

    # Test with combined features and labels
    print("\n")
    print("Test combined features and labels")
    data_path = os.path.join(PATH_DATA_DIR, "data_US", "county_features")
    data_sources = {
        "COMBINED": {
            "filename": "grain_maize_US_train.csv",
            "index_cols": ["COUNTY_ID", "FYEAR"],
            "label_col": "YIELD",
        },
    }

    _dataset = CropYieldDataset(
        data_sources,
        spatial_id_col="COUNTY_ID",
        year_col="FYEAR",
        data_path=data_path,
        combined_features_labels=True,
    )
    print(_dataset["AL_LAWRENCE", 2000])

    dataset_train, dataset_test = _dataset.split_on_years(([2000], [2001]))
    print(dataset_train["AL_LAWRENCE", 2000])
