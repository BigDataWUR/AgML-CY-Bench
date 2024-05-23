import pandas as pd


def data_to_pandas(data_items):
    data = []
    data_cols = None
    for item in data_items:
        if data_cols is None:
            data_cols = list(item.keys())

        data.append([item[c] for c in data_cols])

    return pd.DataFrame(data, columns=data_cols)


def flatten_nested_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_nested_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_nested_dict(d, sep='.'):
    out = {}
    for k, v in d.items():
        keys = k.split(sep)
        if len(keys) == 1:
            out[k] = v
        else:
            new_key = keys[0]
            new_subkey = sep.join(keys[1:])
            if new_key not in out: out[new_key] = {}
            out[new_key][new_subkey] = v
    for k, v in out.items():
        if isinstance(v, dict):
            out[k] = unflatten_nested_dict(v, sep=sep)
    return out
