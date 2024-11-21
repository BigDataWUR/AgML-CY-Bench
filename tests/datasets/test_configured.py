from cybench.datasets.configured import load_dfs_crop


def test_load_dfs_crop():
    df_y, dfs_x = load_dfs_crop("maize", ["NL", "ES"])

    # Sort indices for fast lookup
    df_y.sort_index(inplace=True)
    for x in dfs_x:
        dfs_x[x] = dfs_x[x].sort_index()

    for i, row in df_y.iterrows():
        for df_x in dfs_x.values():
            if len(df_x.index.names) == 1:
                assert i[0] in df_x.index
            else:
                assert i in df_x.index
