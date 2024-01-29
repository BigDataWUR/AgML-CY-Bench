import logging
import pandas as pd
import numpy as np
import random
import comet_ml
import torch
from torch import nn

from models.model import BaseModel
from datasets.torch_dataset import TorchDataset
from evaluation.metrics import normalized_rmse
from config import LOGGER_NAME, comet_api_key

# if comet_api_key is not None:
#     experiment = comet_ml.Experiment(
#         api_key=comet_api_key,
#         project_name="agml-crop-yield-forecasting",
#     )
#     experiment.set_name("python-package")
# else:
#     experiment = None
experiment = None


class LSTMModel(BaseModel, nn.Module):
    def __init__(
        self,
        num_ts_inputs,
        num_trend_inputs,
        num_other_features,
        num_rnn_layers=1,
        rnn_hidden_size=64,
        num_outputs=1,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._batch_norm1 = nn.BatchNorm1d(num_ts_inputs, dtype=torch.double)
        self._rnn = nn.LSTM(
            input_size=num_ts_inputs,
            hidden_size=rnn_hidden_size,
            num_layers=num_rnn_layers,
            batch_first=True,
            dtype=torch.double,
        )

        num_all_features = rnn_hidden_size + num_trend_inputs + num_other_features
        self._batch_norm2 = nn.BatchNorm1d(num_all_features, dtype=torch.double)
        self._fc = nn.Linear(num_all_features, num_outputs, dtype=torch.double)
        self._max_epochs = 10
        self._logger = logging.getLogger(LOGGER_NAME)
        # self._norm_params = None
        # self._normalization = "standard"

    def fit(self, train_dataset, epochs=None, **fit_params):
        self.train()
        label_col = train_dataset.labelCol
        all_inputs = train_dataset.featureCols
        ts_inputs = train_dataset.timeSeriesCols
        other_features = [c for c in all_inputs if (c not in ts_inputs)]
        batch_size = 16

        # self._norm_params = train_dataset.get_normalization_params(normalization=self._normalization)
        if epochs is None:
            epochs = self._max_epochs

        loss = nn.MSELoss()

        if ("optimize_hparameters" in fit_params) and fit_params[
            "optimize_hyperparameters"
        ]:
            # Hyperparameter optimization
            os.makedirs(os.path.join(PATH_OUTPUT_DIR, "saved_models"), exist_ok=True)
            save_model_path = os.path.join(
                PATH_OUTPUT_DIR, "saved_models", "saved_lstm_model"
            )
            torch.save(self.state_dict(), save_model_path)
            hparam_grid = {"lr": [0.0001, 0.00005], "weight_decay": [0.0001, 0.00001]}
            optimal_hparams = self._optimize_hyperparameters(
                train_dataset,
                label_col,
                ts_inputs,
                other_features,
                hparam_grid,
                loss,
                batch_size,
                epochs,
                save_model_path,
            )
            sel_lr = optimal_hparams["lr"]
            sel_wt_decay = optimal_hparams["weight_decay"]

            # load saved model to retrain with optimal hyperparameters
            self.load_state_dict(torch.load(save_model_path))
        else:
            sel_lr = 0.0001
            sel_wt_decay = 0.0001

        torch_dataset = TorchDataset(train_dataset)
        data_loader = torch.utils.data.DataLoader(
            torch_dataset,
            collate_fn=torch_dataset.collate_fn,
            shuffle=True,
            batch_size=batch_size,
        )
        trainer = torch.optim.Adam(
            self.parameters(), lr=sel_lr, weight_decay=sel_wt_decay
        )

        for epoch in range(epochs):
            train_metrics = self._train_epoch(
                data_loader, label_col, ts_inputs, other_features, loss, trainer, epoch
            )
            self._logger.debug(
                "LSTMModel epoch:%d, loss:%f, NRMSE:%f",
                epoch,
                train_metrics["loss"],
                train_metrics["train NRMSE"],
            )
            if experiment is not None:
                experiment.log_metrics(train_metrics, step=epoch)

    def _train_epoch(
        self, train_loader, label_col, ts_inputs, other_features, loss, trainer, epoch
    ):
        epoch_loss = 0
        num_elems = 0
        y_all = None
        y_hat_all = None
        for batch in train_loader:
            # batch = normalize_data(batch, self._norm_params, normalization=self._normalization)
            y = torch.unsqueeze(batch[label_col], 1)
            X_ts = torch.cat([torch.unsqueeze(batch[c], 1) for c in ts_inputs], dim=1)
            X_rest = torch.cat(
                [torch.unsqueeze(batch[c], 1) for c in other_features], dim=1
            )

            y_hat = self(X_ts, X_rest)
            l = loss(y_hat, y)
            trainer.zero_grad()
            l.backward()
            trainer.step()
            epoch_loss += float(l)
            num_elems += y.size().numel()

            if y_all is None:
                y_all = y
                y_hat_all = y_hat
            else:
                y_all = torch.cat([y_all, y], dim=0)
                y_hat_all = torch.cat([y_hat_all, y_hat], dim=0)

        train_nrmse = (
            100 * torch.sqrt(torch.mean((y_hat_all - y_all) ** 2)) / torch.mean(y_all)
        )

        return {"loss": epoch_loss / num_elems, "train NRMSE": train_nrmse.item()}

    def _optimize_hyperparameters(
        self,
        train_dataset,
        label_col,
        ts_inputs,
        other_features,
        hparam_grid,
        loss,
        batch_size,
        epochs,
        save_model_path,
    ):
        train_years = train_dataset.years
        # year_splits = [(list(range(2000, 2012)), list(range(2013, 2018)))]
        year_splits = self._get_validation_splits(
            train_years, num_folds=1, num_valid_years=5
        )
        self._logger.debug("Year splits for hyperparameter optimization")
        for i, (train_years2, valid_years) in enumerate(year_splits):
            self._logger.debug("Split %d Training years: %s", i, train_years2)
            self._logger.debug("Split %d Validation years: %s", i, valid_years)

        optimal_hparams = {}
        for param in hparam_grid:
            optimal_hparams[param] = None

        lowest_nrmse = None
        for lr in hparam_grid["lr"]:
            for wt_decay in hparam_grid["weight_decay"]:
                cv_nrmses = np.zeros(len(year_splits))
                for i, (train_years2, valid_years) in enumerate(year_splits):
                    train_dataset2, valid_dataset = train_dataset.split_on_years(
                        (train_years2, valid_years)
                    )
                    torch_dataset = TorchDataset(train_dataset2)
                    data_loader = torch.utils.data.DataLoader(
                        torch_dataset,
                        collate_fn=torch_dataset.collate_fn,
                        shuffle=True,
                        batch_size=batch_size,
                    )
                    trainer = torch.optim.Adam(
                        self.parameters(), lr=lr, weight_decay=wt_decay
                    )
                    self.load_state_dict(torch.load(save_model_path))
                    valid_nrmse = None
                    for epoch in range(epochs):
                        metrics = self._train_epoch(
                            data_loader,
                            label_col,
                            ts_inputs,
                            other_features,
                            loss,
                            trainer,
                            epoch,
                        )
                        predictions_df = self.predict(valid_dataset)
                        y_true = predictions_df[label_col].values
                        y_pred = predictions_df["PREDICTION"].values
                        valid_nrmse = normalized_rmse(y_true, y_pred)
                        metrics["valid NRMSE"] = valid_nrmse
                        if experiment is not None:
                            experiment.log_metrics(metrics, step=epoch)

                    # Using valid_nrmse from the last epoch
                    cv_nrmses[i] = valid_nrmse

                avg_nrmse = np.mean(cv_nrmses)
                self._logger.debug(
                    "LSTMModel lr:%f, wt_decay:%f, avg NRMSE:%f",
                    lr,
                    wt_decay,
                    avg_nrmse,
                )

                if (lowest_nrmse is None) or (avg_nrmse < lowest_nrmse):
                    lowest_nrmse = avg_nrmse
                    optimal_hparams["lr"] = lr
                    optimal_hparams["weight_decay"] = wt_decay

        self._logger.debug(
            "LSTMModel Optimal lr:%f, wt_decay:%f, avg NRMSE %f",
            optimal_hparams["lr"],
            optimal_hparams["weight_decay"],
            lowest_nrmse,
        )
        if experiment is not None:
            experiment.log_parameters(optimal_hparams)

        return optimal_hparams

    def _get_validation_splits(self, all_years, num_folds=1, num_valid_years=5):
        year_splits = []
        assert len(all_years) >= (num_folds * num_valid_years)
        if num_folds > 1:
            random.shuffle(all_years)

        for i in range(num_folds):
            valid_years = all_years[i * num_valid_years : (i + 1) * num_valid_years]
            train_years = [yr for yr in all_years if yr not in valid_years]
            year_splits.append((train_years, valid_years))

        return year_splits

    def predict(self, test_dataset):
        self.eval()
        label_col = test_dataset.labelCol
        index_cols = test_dataset.indexCols
        all_inputs = test_dataset.featureCols
        ts_inputs = test_dataset.timeSeriesCols
        other_features = [c for c in all_inputs if (c not in ts_inputs)]

        torch_dataset = TorchDataset(test_dataset)
        data_loader = torch.utils.data.DataLoader(
            torch_dataset,
            collate_fn=torch_dataset.collate_fn,
            shuffle=False,
            batch_size=16,
        )

        predictions_df = None
        data_cols = index_cols + [label_col, "PREDICTION"]
        for batch in data_loader:
            # batch = normalize_data(batch, self._norm_params, normalization=self._normalization)
            y = torch.unsqueeze(batch[label_col], 1)
            X_ts = torch.cat([torch.unsqueeze(batch[c], 1) for c in ts_inputs], dim=1)
            X_rest = torch.cat(
                [torch.unsqueeze(batch[c], 1) for c in other_features], dim=1
            )
            y_hat = self(X_ts, X_rest)
            data = []
            num_items = y.shape[0]
            for i in range(num_items):
                data_item = []
                for c in index_cols:
                    data_item.append(batch[c][i])

                data_item += [y[i].item(), y_hat[i].item()]
                data.append(data_item)

            batch_preds = pd.DataFrame(data, columns=data_cols)
            if predictions_df is None:
                predictions_df = batch_preds
            else:
                predictions_df = pd.concat([predictions_df, batch_preds], axis=0)

        # set mode to train
        self.train()
        return predictions_df

    def forward(self, X_ts, X_rest):
        X_ts_norm = self._batch_norm1(X_ts)
        # self._rnn expects (batch, sequence, input variables)
        _, ts_state = self._rnn(X_ts_norm.permute(0, 2, 1))
        ts_h_out = ts_state[0][self._rnn.num_layers - 1].view(-1, self._rnn.hidden_size)

        all_inputs = self._batch_norm2(torch.cat([ts_h_out, X_rest], 1))
        return self._fc(all_inputs)

    def save(self, model_name):
        torch.save(self, model_name)

    @classmethod
    def load(cls, model_name):
        return torch.load(model_name)


import os

from datasets.dataset import CropYieldDataset
from util.data import trend_features
from config import PATH_DATA_DIR, PATH_OUTPUT_DIR


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

    all_years = [yr for yr in range(2000, 2019)]
    test_years = [2012, 2018]
    train_years = [yr for yr in all_years if yr not in test_years]
    dataset = CropYieldDataset(
        data_sources,
        spatial_id_col="COUNTY_ID",
        year_col="FYEAR",
        time_step_col="DEKAD",
        max_time_steps=36,
        lead_time=6,
        data_path=data_path,
    )

    train_dataset, test_dataset = dataset.split_on_years((train_years, test_years))
    all_inputs = train_dataset.featureCols
    ts_inputs = train_dataset.timeSeriesCols
    trend_inputs = [c for c in all_inputs if "YIELD-" in c]
    other_features = [
        c for c in all_inputs if ((c not in ts_inputs) and ("YIELD-" not in c))
    ]

    lstm_model = LSTMModel(
        num_ts_inputs=len(ts_inputs),
        num_trend_inputs=len(trend_inputs),
        num_other_features=len(other_features),
    )
    lstm_model.fit(train_dataset, epochs=10)

    test_preds = lstm_model.predict(test_dataset)
    print(test_preds.head(5).to_string())

    # Test saving and loading
    output_path = os.path.join(PATH_OUTPUT_DIR, "saved_models")
    os.makedirs(output_path, exist_ok=True)

    lstm_model.save(output_path + "/saved_lstm_model.pkl")
    saved_model = LSTMModel.load(output_path + "/saved_lstm_model.pkl")
    test_preds = saved_model.predict(test_dataset)
    print("\n")
    print("Predictions of saved model. Should match earlier output.")
    print(test_preds.head(5).to_string())

    # Test with yield trend
    trend_window = 5
    all_years = list(range(2000, 2019))
    test_years = [2012, 2018]
    train_years = [yr for yr in all_years if yr not in test_years]

    yield_csv = os.path.join(data_path, "YIELD_COUNTY_US.csv")
    yield_df = pd.read_csv(yield_csv, header=0)

    # NOTE: we don't use train_years here
    # because yield data may have years earlier than 2000.
    train_df = yield_df[~yield_df["FYEAR"].isin(test_years)]
    test_df = yield_df[yield_df["FYEAR"].isin(test_years)]

    # For training, we can only use training data to create yield trend features
    train_trend = trend_features(train_df, "COUNTY_ID", "FYEAR", "YIELD", trend_window)
    train_trend = train_trend.dropna(axis=0)
    train_trend = train_trend.drop(
        columns=["YIELD"] + ["FYEAR-" + str(i) for i in range(1, trend_window + 1)]
    )

    # For test data, we can combine train_df and test_df to get trend features
    combined_df = pd.concat([train_df, test_df], axis=0)
    test_trend = trend_features(
        combined_df, "COUNTY_ID", "FYEAR", "YIELD", trend_window
    )
    test_trend = test_trend.dropna(axis=0)
    test_trend = test_trend[test_trend["FYEAR"].isin(test_years)]
    test_trend = test_trend.drop(
        columns=["YIELD"] + ["FYEAR-" + str(i) for i in range(1, trend_window + 1)]
    )

    # Combine the two.
    # NOTE: we cannot create trend features on the full yield data
    # because training set should not use test data.
    trend_data = pd.concat([train_trend, test_trend], axis=0)
    trend_csv = os.path.join(data_path, "YIELD_TREND_COUNTY_US.csv")
    # print(trend_data.head(5))
    trend_data.to_csv(trend_csv, index=False)
    data_sources["YIELD_TREND"] = {
        "filename": "YIELD_TREND_COUNTY_US.csv",
        "index_cols": ["COUNTY_ID", "FYEAR"],
        "sel_cols": ["YIELD-" + str(i) for i in range(1, trend_window + 1)],
    }

    dataset = CropYieldDataset(
        data_sources,
        spatial_id_col="COUNTY_ID",
        year_col="FYEAR",
        time_step_col="DEKAD",
        max_time_steps=36,
        lead_time=6,
        data_path=data_path,
    )

    train_dataset, test_dataset = dataset.split_on_years((train_years, test_years))
    all_inputs = train_dataset.featureCols
    ts_inputs = train_dataset.timeSeriesCols
    trend_inputs = [c for c in all_inputs if "YIELD-" in c]
    other_features = [
        c for c in all_inputs if ((c not in ts_inputs) and ("YIELD-" not in c))
    ]

    lstm_trend_model = LSTMModel(
        num_ts_inputs=len(ts_inputs),
        num_trend_inputs=len(trend_inputs),
        num_other_features=len(other_features),
    )
    lstm_trend_model.fit(train_dataset, epochs=10)

    test_preds = lstm_trend_model.predict(test_dataset)
    print(test_preds.head(5).to_string())
