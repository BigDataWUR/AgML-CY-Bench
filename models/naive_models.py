import pandas as pd
import pickle

from models.model import AgMLBaseModel

class AverageYieldModel(AgMLBaseModel):

    def __init__(self, group_cols, year_col, target_col):
        self._averages = None
        self._group_cols = group_cols
        self._year_col = year_col
        self._target_col = target_col

    def fit(self, train_dataset):
        train_df = self._data_to_pandas(train_dataset)
        # print(train_df.head(5))
        self._averages = train_df.groupby(self._group_cols).agg(PREDICTION=(self._target_col, "mean")).reset_index()
        # print(self.averages.head(5))

    def predict(self, data):
        test_df = self._data_to_pandas(data)
        test_preds = test_df.merge(self._averages, on=self._group_cols)
    
        return test_preds

    def predict(self, test_dataset):
        test_df = self._data_to_pandas(test_dataset)
        test_preds = test_df.merge(self._averages, on=self._group_cols)
    
        return test_preds

    def _data_to_pandas(self, dataset):
        data = []
        for i in range(len(dataset)):
            data_item = dataset[i]
            sel_item = [data_item[c] for c in self._group_cols]
            sel_item += [data_item[self._year_col], data_item[self._target_col]]
            data.append(sel_item)

        data_cols = self._group_cols + [self._year_col, self._target_col]
        return pd.DataFrame(data, columns=data_cols)

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

if __name__ == '__main__':
    data_path = os.path.join(PATH_DATA_DIR, "data_US", "county_data")
    data_sources = {
        "YIELD" : {
            "filename" : "YIELD_COUNTY_US.csv",
            "index_cols" : ["COUNTY_ID", "FYEAR"],
            "sel_cols" : ["YIELD"]
        },
        "METEO" : {
            "filename" : "METEO_COUNTY_US.csv",
            "index_cols" : ["COUNTY_ID", "FYEAR", "DEKAD"],
            "sel_cols" : ["TMAX", "TMIN", "TAVG", "PREC", "ET0", "RAD"]
        },
        "SOIL" : {
            "filename" : "SOIL_COUNTY_US.csv",
            "index_cols" : ["COUNTY_ID"],
            "sel_cols" : ["SM_WHC"]
        },
        "REMOTE_SENSING" : {
            "filename" : "REMOTE_SENSING_COUNTY_US.csv",
            "index_cols" : ["COUNTY_ID", "FYEAR", "DEKAD"],
            "sel_cols" : ["FAPAR"]
        }
    }

    train_years = [y for y in range(2000, 2012)]
    test_years = [y for y in range(2012, 2019)]
    dataset = CropYieldDataset(data_sources, data_path=data_path)
    train_dataset, test_dataset = dataset.split_on_years((train_years, test_years))
 
    average_model = AverageYieldModel(group_cols=["COUNTY_ID"],
                                      year_col="FYEAR",
                                      target_col="YIELD")
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
