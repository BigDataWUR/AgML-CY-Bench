import logging
import pickle
import pandas as pd
import numpy as np

from models.model import BaseModel
from util.data import trend_features
from config import LOGGER_NAME


class LinearTrendModel(BaseModel):
    def __init__(
        self,
        region_col="REGION",
        year_col="YEAR",
        label_col="YIELD",
        trend_window=5,
    ):
        self._region_col = region_col
        self._year_col = year_col
        self._label_col = label_col
        self._data_cols = [self._region_col, self._year_col, self._label_col]
        self._train_df = None
        self._trend_window = trend_window
        self._logger = logging.getLogger(LOGGER_NAME)

    def fit(self, train_df):
        self._train_df = train_df.copy()

    def predict(self, test_df):
        return self._get_trend_predictions(test_df)

    def _get_trend_predictions(self, test_df):
        test_years = test_df[self._year_col].unique()
        trend_fts = pd.concat([self._train_df, test_df], axis=0)

        trend_fts = trend_features(
            trend_fts,
            self._region_col,
            self._year_col,
            self._label_col,
            self._trend_window,
        )
        trend_fts = trend_fts.dropna(axis=0)
        trend_fts = trend_fts[trend_fts[self._year_col].isin(test_years)]  # .copy()
        # linear fit between years and yields
        trend_fts[["SLOPE", "COEFF"]] = trend_fts.apply(
            (
                lambda row: np.polyfit(
                    [
                        row[self._year_col + "-" + str(i)]
                        for i in range(self._trend_window, 0, -1)
                    ],
                    [
                        row[self._label_col + "-" + str(i)]
                        for i in range(self._trend_window, 0, -1)
                    ],
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
        # print(trend_fts.head(10).to_string())
        predictions_df = trend_fts[
            [self._region_col, self._year_col, self._label_col, "PREDICTION"]
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

from config import PATH_DATA_DIR
from config import PATH_OUTPUT_DIR

if __name__ == "__main__":
    data_path = os.path.join(PATH_DATA_DIR, "data_US", "county_data")
    yield_csv = os.path.join(data_path, "YIELD_COUNTY_US.csv")
    yield_df = pd.read_csv(yield_csv, header=0)

    all_years = list(yield_df["FYEAR"].unique())
    test_years = [2012, 2018]
    train_years = [yr for yr in all_years if yr not in test_years]
    train_df = yield_df[yield_df["FYEAR"].isin(train_years)]
    test_df = yield_df[yield_df["FYEAR"].isin(test_years)]

    trend_model = LinearTrendModel("COUNTY_ID", year_col="FYEAR", trend_window=5)
    trend_model.fit(train_df)
    test_preds = trend_model.predict(test_df)
    print(test_preds.head(5).to_string())

    output_path = os.path.join(PATH_OUTPUT_DIR, "saved_models")
    os.makedirs(output_path, exist_ok=True)

    # Test saving and loading
    trend_model.save(output_path + "/saved_trend_model.pkl")
    saved_model = LinearTrendModel.load(output_path + "/saved_trend_model.pkl")
    test_preds = saved_model.predict(test_df)
    print("\n")
    print("Predictions of saved model. Should match earlier output.")
    print(test_preds.head(5).to_string())
