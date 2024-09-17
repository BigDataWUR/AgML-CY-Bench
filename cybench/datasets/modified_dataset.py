import pandas as pd

from cybench.datasets.dataset import Dataset


class ModifiedTargetsDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        modified_targets: pd.DataFrame = None,
    ):
        """
        Dataset wrapper to support modified target values.

        :param dataset: Dataset with original targets
        :param modified_targets: pandas.DataFrame that contains modified targets
        """
        self._crop = dataset.crop
        self._df_y = modified_targets.sort_index()
        self._dfs_x = dataset._dfs_x

        # Check indices match
        input_indices = dataset.indices()
        target_indices = self._df_y.index.values
        assert len(set(input_indices) - set(target_indices)) == 0

        # max crop season window length
        self._max_season_window_length = dataset.max_season_window_length

        # Bool value that specifies whether missing data values are allowed
        # For now always set to False
        self._allow_incomplete = False
