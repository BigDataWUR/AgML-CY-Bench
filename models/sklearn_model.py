import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline

from models.model import BaseModel
from datasets.dataset import Dataset
from util.data import data_to_pandas
from config import KEY_LOC, KEY_YEAR, KEY_TARGET


class SklearnModel(BaseModel):
    def __init__(self, sklearn_est, feature_cols, scaler=None):
        self._feature_cols = feature_cols

        if scaler is None:
            scaler = StandardScaler()

        self._est = Pipeline([("scaler", scaler), ("estimator", sklearn_est)])

    def fit(self, dataset: Dataset, **fit_params) -> tuple:
        """Fit or train the model.

        Args:
          dataset: Dataset

          **fit_params: Additional parameters.

        Returns:
          A tuple containing the fitted model and a dict with additional information.
        """
        train_df = data_to_pandas(dataset)
        train_years = dataset.years
        X = train_df[self._feature_cols].values
        y = train_df[KEY_TARGET].values
        if ("optimize_hyperparameters" in fit_params) and (
            fit_params["optimize_hyperparameters"]
        ):
            assert "param_space" in fit_params
            param_space = fit_params["param_space"]

            # NOTE: 1. optimize hyperparameters refits the estimator
            #          with the optimal hyperparameter values.
            #       2. use kfolds=len(train_years) for leave-one-out
            #
            cv_groups = train_df[KEY_YEAR].values
            self._est = self._optimize_hyperparameters(
                X,
                y,
                param_space,
                groups=cv_groups,
                kfolds=5,
            )

        else:
            self._est.fit(X, y)

        return self, {}

    def _optimize_hyperparameters(self, X, y, param_space, groups=None, kfolds=5):
        """Optimize hyperparameters

        Args:
          X: np.ndarray of training features

          y: np.ndarray of training labels

          param_space: a dict of parameters to optimize

          groups: np.ndarray with group values (e.g year values) for each row in X and y

          kfolds: number of splits cross validation

        Returns:
          A sklearn pipeline refitted with the optimal hyperparameters.
        """
        # GroupKFold to split by years
        if groups is not None:
            group_kfold = GroupKFold(n_splits=kfolds)
            # cv is here a list of tuples (train split, validation split)
            cv = group_kfold.split(X, y, groups)
        else:
            # regular k-fold cv
            cv = kfolds

        # Search for optimal value of hyperparameters
        grid_search = GridSearchCV(self._est, param_grid=param_space, cv=cv)
        grid_search.fit(X, y)

        return grid_search.best_estimator_

    def predict_batch(self, X: list):
        """Run fitted model on batched data items.

        Args:
          X: a list of data items, each of which is a dict

        Returns:
          A tuple containing a np.ndarray and a dict with additional information.
        """
        X_test = np.zeros((len(X), len(self._feature_cols)))
        for i, item in enumerate(X):
            for j, c in enumerate(self._feature_cols):
                X_test[i, j] = item[c]

        return self._est.predict(X_test), {}

    def save(self, model_name):
        """Save model, e.g. using pickle.
        Check here for options to save and load scikit-learn models:
        https://scikit-learn.org/stable/model_persistence.html

        Args:
          model_name: Filename that will be used to save the model.
        """
        with open(model_name, "wb") as f:
            pickle.dump(self, f)

    def load(cls, model_name):
        """Deserialize a saved model.

        Args:
          model_name: Filename that was used to save the model.

        Returns:
          The deserialized model.
        """
        with open(model_name, "rb") as f:
            saved_model = pickle.load(f)

        return saved_model
