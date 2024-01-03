import pandas as pd


def dataset_to_pandas(dataset, data_cols=None):
    data = []
    for i in range(len(dataset)):
        data_item = dataset[i]
        if (data_cols is None):
            data_cols = list(data_item.keys())

        data.append([data_item[c] for c in data_cols])

    return pd.DataFrame(data, columns=data_cols)
