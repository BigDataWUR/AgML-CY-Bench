# Datasets

1. Data alignment before creating a Dataset: Data from the CY-Bench dataset is aligned in two ways before creating a Dataset object.

    a. Alignment to the crop growing season. This is required only for time series data. Before alignment, `date` corresponds to the date of observation. After alignment, `date` is relative to the harvest date, which becomes December 31. For example, if the harvest date is July 24, after alignment this becomes December 31. July 23 will then be December 30 and so on. Similarly, time series data is truncated to the lead time (see `FORECAST_LEAD_TIME` in `config.py`). The crop growing season alignment we did earlier makes it easy to truncate time series from the end. A lead time of a one month means data from December will be dropped. This corresponds to data from one month before harvest getting dropped.

    b. Alignment of indices between predictors and labels (or targets): The data in CY-Bench may have different number of locations (`adm_id`) or years for different data sources. This alignment makes sure that remaining data will include the same set of locations and years.

2. Dataset class: A Dataset object will rely on data being aligned as described in 1. Otherwise data may be missing for certain locations or years.
3. Configured datasets: Data included in CY-Bench can be loaded using `Dataset.load(<dataset_name>)`, which relies on `configured.py`. Alignment of data for configured datasets is implemented in `alignment.py`.
4. Wrappers for PyTorch and other libraries: A Dataset object serves data items as numpy arrays. Casting to tensors or other logic required to work with PyTorch or similar datasets is implemented in wrapper classes (e.g. `TorchDataset`).
4. Transforms: Data items served by a Dataset object may be transformed before training a model. For example, `ExampleLSTM` transforms time series data from their original temporal resolution to dekadal resolution.