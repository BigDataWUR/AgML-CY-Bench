import os
import logging
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error, r2_score

from datasets.dataset import CropYieldDataset
from models.naive_models import AverageYieldModel, RandomYieldModel
from models.trend_models import LinearTrendModel
from models.linear_models import RidgeModel
from models.nn_models import LSTMModel

from config import PATH_DATA_DIR
from config import PATH_LOGS_DIR, LOG_LEVEL, LOGGER_NAME, LOG_FILE


def normalized_rmse(y_true, y_pred):
    return 100 * np.sqrt(mean_squared_error(y_true, y_pred)) / np.mean(y_true)


if __name__ == "__main__":
    # set up logging
    logger = logging.getLogger(LOGGER_NAME)
    handler = logging.FileHandler(os.path.join(PATH_LOGS_DIR, LOG_FILE))
    formatter = logging.Formatter("%(asctime)s %(name)-12s %(levelname)-8s %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(LOG_LEVEL)

    data_path = os.path.join(PATH_DATA_DIR, "data_US", "county_data")
    yield_csv = os.path.join(data_path, "YIELD_COUNTY_US.csv")
    yield_df = pd.read_csv(yield_csv, header=0)

    all_years = list(yield_df["FYEAR"].unique())
    test_years = [2012, 2018]
    train_years = [yr for yr in all_years if yr not in test_years]
    train_df = yield_df[yield_df["FYEAR"].isin(train_years)]
    test_df = yield_df[yield_df["FYEAR"].isin(test_years)]

    # AverageYieldModel
    average_model = AverageYieldModel(
        group_cols=["COUNTY_ID"], year_col="FYEAR", label_col="YIELD"
    )
    average_model.fit(train_df)
    avg_preds = average_model.predict(test_df)

    # RandomYieldModel
    random_model = RandomYieldModel()
    random_model.fit(train_df)
    random_preds = random_model.predict(test_df)

    # LinearTrendModel
    trend_model = LinearTrendModel("COUNTY_ID", year_col="FYEAR", trend_window=5)
    trend_model.fit(train_df)
    trend_preds = trend_model.predict(test_df)

    # RidgeModel: We need to switch to county_features
    data_path = os.path.join(PATH_DATA_DIR, "data_US", "county_features")
    data_file = os.path.join(data_path, "grain_maize_US.csv")
    data_df = pd.read_csv(data_file, header=0)
    all_years = list(data_df["FYEAR"].unique())
    train_years = [yr for yr in all_years if yr not in test_years]
    train_df = data_df[data_df["FYEAR"].isin(train_years)]
    test_df = data_df[data_df["FYEAR"].isin(test_years)]
    ridge_model = RidgeModel(region_col="COUNTY_ID", year_col="FYEAR")
    ridge_model.fit(train_df)

    ridge_preds = ridge_model.predict(test_df)

    test_preds = {
        "AverageYieldModel": avg_preds,
        "RandomYieldModel": random_preds,
        "LinearTrendModel": trend_preds,
        "RidgeModel": ridge_preds,
    }

    # LSTMModel: We need to switch to county_data
    data_path = os.path.join(PATH_DATA_DIR, "data_US", "county_data")
    data_sources = {
        "YIELD": {
            "filename": "YIELD_COUNTY_US.csv",
            "index_cols": ["COUNTY_ID", "FYEAR"],
            "sel_cols": ["YIELD"],
        },
        "METEO": {
            "filename": "METEO_COUNTY_US.csv",
            "index_cols": ["COUNTY_ID", "FYEAR", "DEKAD"],
            "sel_cols": ["TMAX", "TMIN", "TAVG", "PREC", "ET0", "RAD"],
        },
        "SOIL": {
            "filename": "SOIL_COUNTY_US.csv",
            "index_cols": ["COUNTY_ID"],
            "sel_cols": ["SM_WHC"],
        },
        "REMOTE_SENSING": {
            "filename": "REMOTE_SENSING_COUNTY_US.csv",
            "index_cols": ["COUNTY_ID", "FYEAR", "DEKAD"],
            "sel_cols": ["FAPAR"],
        },
    }

    dataset = CropYieldDataset(
        data_sources,
        spatial_id_col="COUNTY_ID",
        year_col="FYEAR",
        data_path=data_path,
        lead_time=6,
    )

    train_dataset, test_dataset = dataset.split_on_years((train_years, test_years))
    all_inputs = train_dataset.featureCols
    ts_inputs = train_dataset.timeSeriesCols
    trend_features = [c for c in all_inputs if "YIELD-" in c]
    other_features = [
        c for c in all_inputs if ((c not in ts_inputs) and ("YIELD-" not in c))
    ]

    lstm_model = LSTMModel(len(ts_inputs), 0, len(other_features))
    lstm_model.fit(train_dataset, epochs=10)
    lstm_preds = lstm_model.predict(test_dataset)
    test_preds["LSTMModel"] = lstm_preds

    performance_data = []
    for model_name in test_preds:
        print("\n")
        print("Predictions of ", model_name)
        model_preds = test_preds[model_name]
        print(model_preds.head(5).to_string())

        nrmse = normalized_rmse(
            model_preds["YIELD"].values, model_preds["PREDICTION"].values
        )
        r2 = r2_score(model_preds["YIELD"].values, model_preds["PREDICTION"].values)
        performance_data.append([model_name, nrmse, r2])

    data_cols = ["Model", "NRMSE", "R2"]
    performance_df = pd.DataFrame(performance_data, columns=data_cols)
    print("\n")
    print(performance_df.head())
