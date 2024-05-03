import pandas as pd


def data_to_pandas(data_items):
    data = []
    data_cols = None
    for item in data_items:
        if data_cols is None:
            data_cols = list(item.keys())

        data.append([item[c] for c in data_cols])

    return pd.DataFrame(data, columns=data_cols)
