import pickle
import pandas as pd

from models.model import BaseModel


class AverageYieldModel(BaseModel):
    def __init__(self, group_cols, year_col, label_col):
        self._averages = None
        self._group_cols = group_cols
        self._year_col = year_col
        self._label_col = label_col

    def fit(self, train_df: pd.DataFrame):
        # print(train_df.head(5))
        self._averages = (
            train_df.groupby(self._group_cols)
            .agg(GROUP_AVG=(self._label_col, "mean"))
            .reset_index()
        )
        # print(self.averages.head(5))

    def predict(self, test_df):
        predictions_df = test_df.merge(self._averages, on=self._group_cols)
        predictions_df = predictions_df.rename(columns={"GROUP_AVG": "PREDICTION"})

        return predictions_df

    def save(self, model_name):
        with open(model_name, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, model_name):
        with open(model_name, "rb") as f:
            saved_model = pickle.load(f)

        return saved_model


import random


class RandomYieldModel(BaseModel):
    def __init__(self, label_col="YIELD", num_samples=20):
        self._max_value = None
        self._min_value = None
        self._num_samples = num_samples
        self._label_col = label_col

    def fit(self, train_df):
        self._max_value = train_df[self._label_col].max()
        self._min_value = train_df[self._label_col].min()

    def predict(self, test_df):
        predictions_df = test_df.copy()
        predictions_df["PREDICTION"] = self._getPredictions(len(test_df.index))

        return predictions_df

    def _getPredictions(self, num_predictions):
        predictions = []
        for i in range(num_predictions):
            avg_value = 0.0
            for j in range(self._num_samples):
                sample = random.uniform(self._min_value, self._max_value)
                avg_value += sample

            avg_value /= self._num_samples
            predictions.append(avg_value)

        return predictions

    def save(self, model_name):
        with open(model_name, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, model_name):
        with open(model_name, "rb") as f:
            saved_model = pickle.load(f)

        return saved_model


import os

from datasets.dataset import CropYieldDataset
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
    average_model = AverageYieldModel(
        group_cols=["COUNTY_ID"], year_col="FYEAR", label_col="YIELD"
    )
    average_model.fit(train_df)
    test_preds = average_model.predict(test_df)
    print(test_preds.head(5).to_string())

    output_path = os.path.join(PATH_OUTPUT_DIR, "saved_models")
    os.makedirs(output_path, exist_ok=True)

    # Test saving and loading
    average_model.save(output_path + "/saved_average_model.pkl")
    saved_model = AverageYieldModel.load(output_path + "/saved_average_model.pkl")
    test_preds = saved_model.predict(test_df)
    print("\n")
    print("Predictions of saved model. Should match earlier output.")
    print(test_preds.head(5).to_string())

    print("\n")
    print("Predictions of random model.")
    random_model = RandomYieldModel()
    random_model.fit(train_df)
    test_preds = random_model.predict(test_df)
    print(test_preds.head(5).to_string())
