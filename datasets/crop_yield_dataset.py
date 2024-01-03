import os
import pandas as pd
from pandas import MultiIndex

from datasets.dataset import AgMLBaseDataset

class CropYieldDataset(AgMLBaseDataset):

    def __init__(self, data_sources, data_dfs=None, data_path=None):
        assert ((data_dfs is not None) or (data_path is not None))
        # Load the data from disk if necessary
        if (data_dfs is None):
            data_dfs = {}
            for src in data_sources:
                filename = data_sources[src]["filename"]
                index_cols = data_sources[src]["index_cols"]
                src_df = CropYieldDataset.get_data(data_path, filename, index_cols)
                data_dfs[src] = src_df

        # TODO -- align data
        # For now, only use counties that are present everywhere
        # TODO -- data preprocessing:
        #   - Start date, end date of season?
        #   - Missing data?

        spatial_units = None
        for src in data_dfs:
            src_units = set(data_dfs[src].index.get_level_values(0))
            if (spatial_units is None):
                spatial_units = src_units
            else:
                spatial_units = set.intersection(spatial_units, src_units)

        for src in data_dfs:
            src_df = CropYieldDataset._filter_df_on_index(data_dfs[src],
                                                          list(spatial_units),
                                                          level=0)
            # Sort the data for faster lookups
            src_df.sort_index(inplace=True)
            data_dfs[src] = src_df

        self._data_y = data_dfs["YIELD"]
        self._data_sources = data_sources
        self._data_dfs = data_dfs

    def __getitem__(self, index) -> dict:
        # Index is either integer or tuple of (year, location)

        if isinstance(index, int):
            sample_y = self._data_y.iloc[index]
            spatial_unit, year = sample_y.name

        elif isinstance(index, tuple):
            spatial_unit, year = index
            sample_y = self._data_y.loc[index]

        else:
            raise Exception(f'Unsupported index type {type(index)}')

        data_item = {
            'FYEAR': year,
            'COUNTY_ID': spatial_unit,
            'YIELD': sample_y.YIELD,
        }

        for src in self._data_sources:
            if (src == "YIELD"):
                continue

            if ("FYEAR" in self._data_sources[src]["index_cols"]):
                src_sample = self._get_data_sample(src, spatial_unit, year)
            else:
                src_sample = self._get_data_sample(src, spatial_unit)
            
            data_item.update(**src_sample)

        return data_item

    def _get_data_sample(self, src, spatial_unit, year=None):

        sel_cols = self._data_sources[src]["sel_cols"]

        # static data, no year
        if (year is None):
            df = self._data_dfs[src].loc[spatial_unit]
            sample = {
                key : df[key] for key in sel_cols
            }

        # time series data
        else:
            df = self._data_dfs[src].xs((spatial_unit, year), drop_level=True)
            sample = {
                key : df[key].values for key in sel_cols
            }

        return sample

    def __len__(self) -> int:
        return len(self._data_y)

    def split_on_years(self, years_split: tuple) -> tuple:

        data_dfs1 = {}
        data_dfs2 = {}
        for src in self._data_dfs:
            src_df = self._data_dfs[src]
            if ("FYEAR" in self._data_sources[src]["index_cols"]):
                df1, df2 = self._split_df_on_index(src_df, years_split, level=1)
                data_dfs1[src] = df1
                data_dfs2[src] = df2
            else:
                data_dfs1[src] = src_df.copy()
                data_dfs2[src] = src_df.copy()

        return (
            CropYieldDataset(self._data_sources, data_dfs=data_dfs1),
            CropYieldDataset(self._data_sources, data_dfs=data_dfs2),
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

from config import PATH_DATA_DIR

if __name__ == '__main__':
    data_path = os.path.join(PATH_DATA_DIR, "data_US", "county_data")
    data_sources = {
        "YIELD" : {
            "filename" : "YIELD_COUNTY_US.csv",
            "index_cols" : ["COUNTY_ID", "FYEAR"],
            "sel_cols" : ["YIELD"]
        },
        "METEO" : {
            "filename" : "METEO_COUNTY_US.csv",
            "index_cols" : ["COUNTY_ID", "FYEAR", "DEKAD"],
            "sel_cols" : ["TMAX", "TMIN", "TAVG", "PREC", "ET0", "RAD"]
        },
        "SOIL" : {
            "filename" : "SOIL_COUNTY_US.csv",
            "index_cols" : ["COUNTY_ID"],
            "sel_cols" : ["SM_WHC"]
        },
        "REMOTE_SENSING" : {
            "filename" : "REMOTE_SENSING_COUNTY_US.csv",
            "index_cols" : ["COUNTY_ID", "FYEAR", "DEKAD"],
            "sel_cols" : ["FAPAR"]
        }
    }

    _dataset = CropYieldDataset(data_sources, data_path=data_path)

    # print(_dataset.years)
    # print(_dataset.locations)
    print(_dataset['AL_LAWRENCE', 2000])

    dataset_train, dataset_test = _dataset.split_on_years(([2000], [2001]))