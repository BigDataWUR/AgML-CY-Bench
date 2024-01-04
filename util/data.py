import pandas as pd
import os


def csv_to_pandas(data_path, filename, index_cols):
    path = os.path.join(data_path, filename)
    df = pd.read_csv(path, index_col=index_cols)

    return df


def dataset_to_pandas(dataset, data_cols=None):
    data = []
    for i in range(len(dataset)):
        data_item = dataset[i]
        if (data_cols is None) and (i == 0):
            data_cols = list(data_item.keys())

        data.append([data_item[c] for c in data_cols])

    return pd.DataFrame(data, columns=data_cols)
