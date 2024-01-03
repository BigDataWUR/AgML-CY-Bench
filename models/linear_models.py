import pickle

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from models.model import AgMLBaseModel
from util.data_util import dataset_to_pandas


class RidgeModel(AgMLBaseModel):
    def __init__(self, weight_decay=0.5):
        self._ridge = Ridge(alpha=weight_decay)
        self._scaler = StandardScaler()
        self._pipeline = Pipeline(
            [("scaler", self._scaler), ("estimator", self._ridge)]
        )

    def fit(self, train_dataset):
        train_df = dataset_to_pandas(train_dataset)
        feature_cols = train_dataset.featureCols
        label_col = train_dataset.labelCol
        X = train_df[feature_cols].values
        y = train_df[label_col].values
        self._pipeline.fit(X, y)

    def predict(self, test_dataset):
        test_df = dataset_to_pandas(test_dataset)
        feature_cols = test_dataset.featureCols
        label_col = test_dataset.labelCol
        X = test_df[feature_cols].values
        predictions = self._pipeline.predict(X)
        id_cols = test_dataset.indexCols
        predictions_df = test_df[id_cols + [label_col]].copy()
        predictions_df["PREDICTION"] = predictions

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
    data_path = os.path.join(PATH_DATA_DIR, "data_US", "county_features")
    train_source = {
        "COMBINED": {
            "filename": "grain_maize_US_train.csv",
            "index_cols": ["COUNTY_ID", "FYEAR"],
            "label_col": "YIELD",
        },
    }

    test_source = {
        "COMBINED": {
            "filename": "grain_maize_US_test.csv",
            "index_cols": ["COUNTY_ID", "FYEAR"],
            "label_col": "YIELD",
        },
    }

    train_dataset = CropYieldDataset(
        train_source,
        spatial_id_col="COUNTY_ID",
        year_col="FYEAR",
        data_path=data_path,
        combined_features_labels=True,
    )

    ridge_model = RidgeModel(weight_decay=1.0)
    ridge_model.fit(train_dataset)

    test_dataset = CropYieldDataset(
        test_source,
        spatial_id_col="COUNTY_ID",
        year_col="FYEAR",
        data_path=data_path,
        combined_features_labels=True,
    )

    test_preds = ridge_model.predict(test_dataset)
    print(test_preds.head(5).to_string())

    output_path = os.path.join(PATH_OUTPUT_DIR, "saved_models")
    os.makedirs(output_path, exist_ok=True)

    # Test saving and loading
    ridge_model.save(output_path + "/saved_ridge_model.pkl")
    saved_model = RidgeModel.load(output_path + "/saved_ridge_model.pkl")
    test_preds = saved_model.predict(test_dataset)
    print("\n")
    print("Predictions of saved model. Should match earlier output.")
    print(test_preds.head(5).to_string())
