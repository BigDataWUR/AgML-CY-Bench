class Dataset:
    def __init__(
        self,
        data_dfs,
    ):
        self._feature_cols = []
        self._feature_dfs = []
        # Sort the data for faster lookups
        for src in self._data_dfs:
            if (src == "YIELD"):
                self._target_df = data_dfs[src]
                self._index_cols = list(self._target_df.index.name)
                self._target_df.sort_index(inplace=True)
            else:
                self._feature_cols += list(data_dfs[src].columns)
                self._feature_dfs.append(data_dfs[src])
                self._data_dfs[src].sort_index(inplace=True)

        self._loc_id_col = self._index_cols[0]
        self._year_col = self._index_cols[1]

    def __getitem__(self, index) -> dict:
        # Index is either integer or tuple of (location, year)
        if isinstance(index, int):
            sample_y = self._data_y.iloc[index]
            loc_id, year = sample_y.name

        elif isinstance(index, tuple):
            loc_id, year = index
            sample_y = self._data_y.loc[index]

        else:
            raise Exception(f"Unsupported index type {type(index)}")

        sample =  {
            self._loc_id_col: loc_id,
            self._year_col: year,
            "YIELD": sample_y["YIELD"],
        }

        data_x = self._get_feature_data(loc_id, year)
        sample = {** data_x, **sample}

    def _get_feature_data(self, loc_id: str, year: int) -> dict:
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

    def __len__(self) -> int:
        return len(self._data_y)

    @property
    def years(self) -> list:
        return list(set([year for _, year in self._data_y.index.values]))

    @property
    def indexCols(self) -> list:
        return self._index_cols

    @property
    def labelCol(self) -> str:
        return "YIELD"
