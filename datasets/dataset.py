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

    @property
    def years(self) -> list:
        return list(set([year for _, year in self._data_y.index.values]))

    @property
    def indexCols(self) -> list:
        return self._index_cols

    @property
    def labelCol(self) -> str:
        return "YIELD"
