import pickle

from models.model import AgMLBaseModel
from util.data_util import dataset_to_pandas


class AverageYieldModel(AgMLBaseModel):
    def __init__(self, group_cols, year_col, label_col):
        self._averages = None
        self._group_cols = group_cols
        self._year_col = year_col
        self._label_col = label_col

    def fit(self, train_dataset):
        data_cols = self._group_cols + [self._year_col, self._label_col]
        train_df = dataset_to_pandas(train_dataset, data_cols)
        # print(train_df.head(5))
        self._averages = (
            train_df.groupby(self._group_cols)
            .agg(PREDICTION=(self._label_col, "mean"))
            .reset_index()
        )
        # print(self.averages.head(5))

    def predict(self, data):
        data_cols = self._group_cols + [self._year_col, self._label_col]
        test_df = dataset_to_pandas(data, data_cols)
        test_preds = test_df.merge(self._averages, on=self._group_cols)

        return test_preds

    def predict(self, test_dataset):
        data_cols = self._group_cols + [self._year_col, self._label_col]
        test_df = dataset_to_pandas(test_dataset, data_cols)
        test_preds = test_df.merge(self._averages, on=self._group_cols)

        return test_preds

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

    average_model = AverageYieldModel(
        group_cols=["COUNTY_ID"], year_col="FYEAR", label_col="YIELD"
    )
    average_model.fit(train_dataset)
    test_preds = average_model.predict(test_dataset)
    print(test_preds.head(5).to_string())

    output_path = os.path.join(PATH_OUTPUT_DIR, "saved_models")
    os.makedirs(output_path, exist_ok=True)

    # Test saving and loading
    average_model.save(output_path + "/saved_average_model.pkl")
    saved_model = AverageYieldModel.load(output_path + "/saved_average_model.pkl")
    test_preds = saved_model.predict(test_dataset)
    print("\n")
    print("Predictions of saved model. Should match earlier output.")
    print(test_preds.head(5).to_string())
