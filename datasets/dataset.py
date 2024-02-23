class Dataset:
    def __init__(
        self,
        target_df,
        feature_dfs=[],
    ):
        self._df_y = target_df
        self._index_cols = list(self._df_y.index.names)
        self._df_y.sort_index(inplace=True)

        self._feature_cols = []
        self._dfs_x = []
        # Sort the data for faster lookups
        for src_df in feature_dfs:
            self._feature_cols += list(src_df.columns)
            src_df.sort_index(inplace=True)
            self._dfs_x.append(src_df)

        self._loc_id_col = self._index_cols[0]
        self._year_col = self._index_cols[1]

    def __getitem__(self, index) -> dict:
        # Index is either integer or tuple of (location, year)
        if isinstance(index, int):
            sample_y = self._df_y.iloc[index]
            loc_id, year = sample_y.name

        elif isinstance(index, tuple):
            loc_id, year = index
            sample_y = self._data_y.loc[index]

        else:
            raise Exception(f"Unsupported index type {type(index)}")

        data_y = {
            self._loc_id_col: loc_id,
            self._year_col: year,
            "YIELD": sample_y["YIELD"],
        }

        data_x = self._get_feature_data(loc_id, year)
        return {**data_x, **data_y}

    def _get_feature_data(self, loc_id: str, year: int) -> dict:
        data = dict()
        for df in self._dfs_x:
            n_levels = len(df.index.names)
            assert 1 <= n_levels <= 3
            if n_levels == 1:
                data = {
                    **df.loc[loc_id].to_dict(),
                    **data,
                }

            if n_levels == 2:
                data = {
                    **df.loc[loc_id, year].to_dict(),
                    **data,
                }

            if n_levels == 3:
                df_loc = df.xs((loc_id, year), drop_level=True)
                data_loc = {key: df_loc[key].values for key in df_loc.columns}

                data = {
                    **data_loc,
                    **data,
                }

        return data

    def __len__(self) -> int:
        return len(self._df_y)

    @property
    def years(self) -> list:
        return list(set([year for _, year in self._df_y.index.values]))

    @property
    def indexCols(self) -> list:
        return self._index_cols

    @property
    def feature_cols(self) -> list:
        return self._feature_cols

    @property
    def labelCol(self) -> str:
        return "YIELD"
