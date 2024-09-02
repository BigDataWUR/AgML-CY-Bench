import pandas as pd


def data_to_pandas(data_items, data_cols=None):
    """Convert data items as dict to pandas DataFrame

    Args:
      data_items : list of data items, each of which is a dict
      data_cols : list of keys to include as columns

    Returns:
      pd.DataFrame
    """
    data = []
    for item in data_items:
        if data_cols is None:
            data_cols = list(item.keys())

        data.append([item[c] for c in data_cols])

    return pd.DataFrame(data, columns=data_cols)
