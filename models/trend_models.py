import pickle
import pandas as pd
import numpy as np

from models.model import AgMLBaseModel
from util.data_util import dataset_to_pandas


class LinearTrendModel(AgMLBaseModel):
    def __init__(
        self, spatial_id_col, year_col="YEAR", label_col="YIELD", trend_window=5
    ):
        self._id_col = spatial_id_col
        self._year_col = year_col
        self._label_col = label_col
        self._data_cols = [self._id_col, self._year_col, self._label_col]
        self._train_labels = None
        self._trend_window = trend_window

    def fit(self, train_dataset):
        self._train_labels = dataset_to_pandas(train_dataset, self._data_cols)

    def predict(self, data):
        test_labels = dataset_to_pandas(data, self._data_cols)

        return self._get_trend_predictions(test_labels)

    def predict(self, test_dataset):
        test_labels = dataset_to_pandas(test_dataset, self._data_cols)

        return self._get_trend_predictions(test_labels)

    def _get_trend_predictions(self, test_labels):
        test_years = test_labels[self._year_col].unique()
        trend_fts = pd.concat([self._train_labels, test_labels], axis=0)
        trend_fts = trend_fts.sort_values(by=[self._id_col, self._year_col])
        for i in range(self._trend_window, 0, -1):
            trend_fts["YEAR-" + str(i)] = trend_fts.groupby([self._id_col])[
                self._year_col
            ].shift(i)
        for i in range(self._trend_window, 0, -1):
            trend_fts["YIELD-" + str(i)] = trend_fts.groupby([self._id_col])[
                self._label_col
            ].shift(i)

        trend_fts = trend_fts.dropna(axis=0)
        trend_fts = trend_fts[trend_fts[self._year_col].isin(test_years)].copy()
        # linear fit between years and yields
        trend_fts[["SLOPE", "COEFF"]] = trend_fts.apply(
            (
                lambda row: np.polyfit(
                    [row["YEAR-" + str(i)] for i in range(self._trend_window, 0, -1)],
                    [row["YIELD-" + str(i)] for i in range(self._trend_window, 0, -1)],
                    1,
                )
            ),
            axis=1,
            result_type="expand",
        )
        # predict
        trend_fts["PREDICTION"] = (
            trend_fts["COEFF"] + trend_fts["SLOPE"] * trend_fts[self._year_col]
        )
        predictions_df = trend_fts[
            [self._id_col, self._year_col, self._label_col, "PREDICTION"]
        ]
        # round values: it does not make sense to give more precision than the original values.
        # TODO: figure out rounding based on precision of yield data.
        # predictions_df["PREDICTION"] = predictions_df["PREDICTION"].round(1)

        return predictions_df

    def save(self, model_name):
        with open(model_name, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, model_name):
        with open(model_name, "rb") as f:
            saved_model = pickle.load(f)

        return saved_model


import os

from datasets.crop_yield_dataset import CropYieldDataset
from config import PATH_DATA_DIR
from config import PATH_OUTPUT_DIR

if __name__ == "__main__":
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

    train_years = [y for y in range(2000, 2012)]
    test_years = [y for y in range(2012, 2019)]
    dataset = CropYieldDataset(data_sources, data_path=data_path)
    train_dataset, test_dataset = dataset.split_on_years((train_years, test_years))

    trend_model = LinearTrendModel("COUNTY_ID", year_col="FYEAR", trend_window=5)
    trend_model.fit(train_dataset)
    test_preds = trend_model.predict(test_dataset)
    print(test_preds.head(5).to_string())

    output_path = os.path.join(PATH_OUTPUT_DIR, "saved_models")
    os.makedirs(output_path, exist_ok=True)

    # Test saving and loading
    trend_model.save(output_path + "/saved_trend_model.pkl")
    saved_model = LinearTrendModel.load(output_path + "/saved_trend_model.pkl")
    test_preds = saved_model.predict(test_dataset)
    print("\n")
    print("Predictions of saved model. Should match earlier output.")
    print(test_preds.head(5).to_string())