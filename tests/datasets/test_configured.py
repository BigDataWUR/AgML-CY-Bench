from cybench.datasets.configured import load_dfs_maize_nl


def test_load_dfs_maize():
    df_y, dfs_x = load_dfs_maize_nl()

    # Sort indices for fast lookup
    df_y.sort_index(inplace=True)
    for df_x in dfs_x:
        df_x.sort_index(inplace=True)

    for i, row in df_y.iterrows():
        for df_x in dfs_x:
            if len(df_x.index.names) == 1:
                assert i[0] in df_x.index
            else:
                assert i in df_x.index
