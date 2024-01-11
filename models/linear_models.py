import pickle

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, GroupKFold

from models.model import BaseModel


class RidgeModel(BaseModel):
    def __init__(
        self, region_col="REGION", year_col="YEAR", label_col="YIELD", scaler=None
    ):
        self._region_col = region_col
        self._year_col = year_col
        self._label_col = label_col
        self._non_feature_cols = [self._region_col, self._year_col, self._label_col]

        self._ridge = Ridge(alpha=1.0)
        if scaler is None:
            self._scaler = StandardScaler()
        else:
            self._scaler = scaler

        self._pipeline = Pipeline(
            [("scaler", self._scaler), ("estimator", self._ridge)]
        )
        self._best_est = None

    def fit(self, train_df):
        train_years = list(train_df[self._year_col].unique())
        param_grid = {"estimator__alpha": [0.01, 0.1, 0.0, 1.0, 5.0, 10.0]}

        feature_cols = [c for c in train_df.columns if c not in self._non_feature_cols]
        X = train_df[feature_cols].values
        y = train_df[self._label_col].values
        group_kfold = GroupKFold(n_splits=len(train_years))
        groups = train_df[self._year_col].values
        cv = group_kfold.split(X, y, groups)
        grid_search = GridSearchCV(self._pipeline, param_grid=param_grid, cv=cv)
        grid_search.fit(X, y)
        print("RidgeModel Optimal Hyperparameters:")
        print(grid_search.best_params_)
        print("\n")

        self._best_est = grid_search.best_estimator_

    def predict(self, test_df):
        feature_cols = [c for c in test_df.columns if c not in self._non_feature_cols]
        X = test_df[feature_cols].values
        predictions_df = test_df[self._non_feature_cols].copy()
        predictions_df["PREDICTION"] = self._best_est.predict(X)

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
import pandas as pd

if __name__ == "__main__":
    data_path = os.path.join(PATH_DATA_DIR, "data_US", "county_features")
    data_file = os.path.join(data_path, "grain_maize_US.csv")
    data_df = pd.read_csv(data_file, header=0)
    all_years = list(data_df["FYEAR"].unique())
    test_years = [2012, 2018]
    train_years = [yr for yr in all_years if yr not in test_years]
    train_df = data_df[data_df["FYEAR"].isin(train_years)]
    test_df = data_df[data_df["FYEAR"].isin(test_years)]
    ridge_model = RidgeModel(region_col="COUNTY_ID", year_col="FYEAR")
    ridge_model.fit(train_df)

    test_preds = ridge_model.predict(test_df)
    print(test_preds.head(5).to_string())

    output_path = os.path.join(PATH_OUTPUT_DIR, "saved_models")
    os.makedirs(output_path, exist_ok=True)

    # Test saving and loading
    ridge_model.save(output_path + "/saved_ridge_model.pkl")
    saved_model = RidgeModel.load(output_path + "/saved_ridge_model.pkl")
    test_preds = saved_model.predict(test_df)
    print("\n")
    print("Predictions of saved model. Should match earlier output.")
    print(test_preds.head(5).to_string())
