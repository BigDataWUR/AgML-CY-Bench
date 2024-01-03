import pandas as pd


def dataset_to_pandas(dataset, data_cols):
    data = []
    for i in range(len(dataset)):
        data_item = dataset[i]
        data.append([data_item[c] for c in data_cols])

    return pd.DataFrame(data, columns=data_cols)
