import os
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error, r2_score

from datasets.dataset import CropYieldDataset
from models.naive_models import AverageYieldModel, RandomAverageYieldModel
from models.trend_models import LinearTrendModel
from models.linear_models import RidgeModel
from models.nn_models import LSTMModel

from config import PATH_DATA_DIR

def normalized_rmse(y_true, y_pred):
   return 100 * np.sqrt(mean_squared_error(y_true, y_pred)) / np.mean(y_true)

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
    dataset = CropYieldDataset(
        data_sources, spatial_id_col="COUNTY_ID", year_col="FYEAR", data_path=data_path,
        lead_time=6
    )

    train_dataset, test_dataset = dataset.split_on_years((train_years, test_years))
    all_inputs = train_dataset.featureCols
    ts_inputs = train_dataset.timeSeriesCols
    trend_features = [c for c in all_inputs if "YIELD-" in c]
    other_features = [
        c for c in all_inputs if ((c not in ts_inputs) and ("YIELD-" not in c))
    ]

    models = {
        "RandomAverageYieldModel" : RandomAverageYieldModel(index_cols=["COUNTY_ID", "FYEAR"],
                                                            label_col="YIELD"),
        "AverageYieldModel" : AverageYieldModel(
            group_cols=["COUNTY_ID"], year_col="FYEAR", label_col="YIELD"
        ),
        "LinearTrendModel" : LinearTrendModel(spatial_id_col="COUNTY_ID", year_col="FYEAR", trend_window=5),
        "LSTMModel" : LSTMModel(
            num_ts_inputs=len(ts_inputs),
            num_trend_features=len(trend_features),
            num_other_features=len(other_features),
        )
    }
    test_preds = {}

    for model_name in models:  
      print("\n")
      print("Predictions of ", model_name)
      model = models[model_name]
      model.fit(train_dataset)
      preds_df = model.predict(test_dataset)
      test_preds[model_name] = preds_df
      print(preds_df.head(5).to_string())

    # For RidgeModel we need to use feature and labels
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

    print("\n")
    print("Predictions of RidgeModel")
    preds_df = ridge_model.predict(test_dataset)
    print(preds_df.head(5).to_string())
    test_preds["RidgeModel"] = preds_df

    performance_data = []
    for model_name in test_preds:
       model_preds = test_preds[model_name]
       nrmse = normalized_rmse(model_preds["YIELD"].values,
                               model_preds["PREDICTION"].values)
       r2 = r2_score(model_preds["YIELD"].values,
                     model_preds["PREDICTION"].values)
       performance_data.append([model_name, nrmse, r2])
       print(model_name, "NRMSE:")
    
    data_cols = ["Model", "NRMSE", "R2"]
    performance_df = pd.DataFrame(performance_data, columns=data_cols)
    print(performance_df.head())