import os
import logging
import pandas as pd
import numpy as np
import random
import torch
from torch import nn

from cybench.datasets.dataset_torch import TorchDataset
from cybench.models.model import BaseModel
from cybench.models.nn_models import ExampleLSTM
from cybench.evaluation.eval import normalized_rmse, evaluate_predictions

from cybench.config import (
    PATH_DATA_DIR,
    PATH_OUTPUT_DIR,
    KEY_LOC,
    KEY_YEAR,
    KEY_TARGET,
    METEO_INDICATORS,
    RS_FPAR,
)

STATIC_PREDICTORS = ["awc"]
TIME_SERIES_PREDICTORS = METEO_INDICATORS + [RS_FPAR]


class LSTMModel(BaseModel, nn.Module):
    def __init__(
        self, num_rnn_layers=1, rnn_hidden_size=64, num_outputs=1, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        num_ts_inputs = len(TIME_SERIES_PREDICTORS)
        num_other_inputs = len(STATIC_PREDICTORS)
        self._batch_norm1 = nn.BatchNorm1d(num_ts_inputs)
        self._rnn = nn.LSTM(
            input_size=num_ts_inputs,
            hidden_size=rnn_hidden_size,
            num_layers=num_rnn_layers,
            batch_first=True,
        )

        num_all_inputs = rnn_hidden_size + num_other_inputs
        self._batch_norm2 = nn.BatchNorm1d(num_all_inputs)
        self._fc = nn.Linear(num_all_inputs, num_outputs)
        self._logger = logging.getLogger(__name__)

    def fit(
        self, train_dataset, optimize_hyperparameters=False, epochs=10, **fit_params
    ):
        self.train()
        batch_size = 16
        loss = nn.MSELoss()
        if optimize_hyperparameters:
            # Hyperparameter optimization
            os.makedirs(os.path.join(PATH_OUTPUT_DIR, "saved_models"), exist_ok=True)
            save_model_path = os.path.join(
                PATH_OUTPUT_DIR, "saved_models", "saved_lstm_model"
            )
            torch.save(self.state_dict(), save_model_path)
            param_space = {"lr": [0.0001, 0.00005], "weight_decay": [0.0001, 0.00001]}
            opt_hparams = self._optimize_hyperparameters(
                train_dataset,
                param_space,
                loss,
                batch_size,
                epochs,
                save_model_path,
            )
            sel_lr = opt_hparams["lr"]
            sel_wt_decay = opt_hparams["weight_decay"]

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
        optimizer = torch.optim.Adam(
            self.parameters(), lr=sel_lr, weight_decay=sel_wt_decay
        )

        for epoch in range(epochs):
            train_metrics = self._train_epoch(data_loader, loss, optimizer)
            self._logger.debug(
                "LSTMModel epoch:%d, loss:%f, NRMSE:%f",
                epoch,
                train_metrics["loss"],
                train_metrics["train NRMSE"],
            )

    def _train_epoch(self, train_loader, loss, optimizer):
        epoch_loss = 0
        num_elems = 0
        y_all = None
        y_hat_all = None
        for batch in train_loader:
            y = torch.unsqueeze(batch[KEY_TARGET], 1)
            X_ts = torch.cat(
                [torch.unsqueeze(batch[c], 1) for c in TIME_SERIES_PREDICTORS], dim=1
            )
            X_rest = torch.cat(
                [torch.unsqueeze(batch[c], 1) for c in STATIC_PREDICTORS], dim=1
            )

            y_hat = self(X_ts, X_rest)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
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
        param_space,
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
        for param in param_space:
            optimal_hparams[param] = None

        lowest_nrmse = None
        for lr in param_space["lr"]:
            for wt_decay in param_space["weight_decay"]:
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
                    optimizer = torch.optim.Adam(
                        self.parameters(), lr=lr, weight_decay=wt_decay
                    )
                    self.load_state_dict(torch.load(save_model_path))
                    valid_nrmse = None
                    for epoch in range(epochs):
                        metrics = self._train_epoch(data_loader, loss, optimizer)
                        y_pred = self.predict(valid_dataset)
                        y_true = valid_dataset.targets
                        valid_nrmse = normalized_rmse(y_true, y_pred)
                        metrics["valid NRMSE"] = valid_nrmse

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

    def predict_items(
        self,
        X: list,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **predict_params,
    ):
        """Run fitted model on a list of data items.

        Args:
          X (list): a list of data items, each of which is a dict
          device (str): str, the device to use
          **predict_params: Additional parameters

        Returns:
          A tuple containing a np.ndarray and a dict with additional information.
        """
        self.to(device)
        self.eval()

        with torch.no_grad():
            X_collated = TorchDataset.collate_fn(
                [TorchDataset._cast_to_tensor(x) for x in X]
            )
            y_pred = self._forward_pass(X_collated, device)
            y_pred = y_pred.cpu().numpy()
            return y_pred, {}

    def predict(self, test_dataset):
        self.eval()
        torch_dataset = TorchDataset(test_dataset)
        data_loader = torch.utils.data.DataLoader(
            torch_dataset,
            collate_fn=torch_dataset.collate_fn,
            shuffle=False,
            batch_size=16,
        )

        with torch.no_grad():
            predictions = None
            for batch in data_loader:
                X_ts = torch.cat(
                    [torch.unsqueeze(batch[c], 1) for c in TIME_SERIES_PREDICTORS],
                    dim=1,
                )
                X_rest = torch.cat(
                    [torch.unsqueeze(batch[c], 1) for c in STATIC_PREDICTORS], dim=1
                )
                batch_preds = self(X_ts, X_rest)
                if batch_preds.dim() > 1:
                    batch_preds = batch_preds.squeeze(-1)

                if predictions is None:
                    predictions = batch_preds.cpu().numpy()
                else:
                    predictions = np.concatenate((predictions, batch_preds), axis=0)

            # set mode to train
            self.train()
            return predictions, {}

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


def date_from_dekad(dekad, year):
    date_str = str(year)
    month = int(np.ceil(dekad / 3))
    if month < 10:
        date_str += "0" + str(month)
    else:
        date_str += str(month)

    if dekad % 3 == 1:
        date_str += "01"
    elif dekad % 3 == 2:
        date_str += "11"
    else:
        date_str += "21"

    return date_str


from cybench.datasets.alignment import align_data
from cybench.datasets.dataset import Dataset


if __name__ == "__main__":
    path_data_cn = os.path.join(PATH_DATA_DIR, "workshop-data")
    df_y = pd.read_csv(os.path.join(path_data_cn, "yield_maize_US.csv"), header=0)
    # convert maize (corn) yield from bushels/acre to t/ha
    # See https://www.extension.iastate.edu/agdm/wholefarm/html/c6-80.html
    df_y[KEY_TARGET] = 0.0628 * df_y[KEY_TARGET]
    df_y = df_y.rename(columns={"harvest_year": KEY_YEAR})
    df_y.set_index([KEY_LOC, KEY_YEAR], inplace=True)

    df_crop_cal = pd.read_csv(
        os.path.join(path_data_cn, "crop_calendar_maize_US.csv"), header=0
    )
    season_start = int(df_crop_cal["sos"].mean())
    season_end = int(df_crop_cal["eos"].mean())
    dfs_x = []
    for input in ["soil", "meteo", "fpar"]:
        df_x = pd.read_csv(
            os.path.join(path_data_cn, input + "_maize_US.csv"), header=0
        )
        if input == "soil":
            df_x = df_x[[KEY_LOC, "awc"]]
            df_x.set_index([KEY_LOC], inplace=True)
        else:
            # lead time = 6 dekads
            df_x = df_x[df_x["dekad"] <= 30]
            df_x["date"] = df_x.apply(
                lambda r: date_from_dekad(r["dekad"], r["year"]), axis=1
            )
            df_x = df_x.drop(columns=["dekad"])
            df_x.set_index([KEY_LOC, KEY_YEAR, "date"], inplace=True)
            if input == "meteo":
                df_x["cwb"] = df_x["prec"] = df_x["et0"]
                df_x = df_x[METEO_INDICATORS]

        dfs_x.append(df_x)

    df_y, dfs_x = align_data(df_y, dfs_x)
    dataset = Dataset("maize", df_y, dfs_x)
    all_years = dataset.years
    test_years = [2012, 2018]
    train_years = [yr for yr in all_years if yr not in test_years]
    train_dataset, test_dataset = dataset.split_on_years((train_years, test_years))
    test_labels = test_dataset.targets()

    model_output = {
        KEY_LOC: [loc_id for loc_id, _ in test_dataset.indices()],
        KEY_YEAR: [year for _, year in test_dataset.indices()],
        "targets": test_labels,
    }

    for model_name in ["workshop_lstm", "benchmark_lstm"]:
        if model_name == "workshop_lstm":
            lstm_model = LSTMModel()
        else:
            # NOTE: config.py requires these updates to test with ExampleLSTM.
            # 1. SOIL_PROPERTIES = ["awc"]
            # 2. TIME_SERIES_PREDICTORS = METEO_INDICATORS + [RS_FPAR]
            # The workshop data does not include other inputs from the benchmark.
            lstm_model = ExampleLSTM(transforms=[])

        lstm_model.fit(train_dataset, epochs=10)
        test_preds, _ = lstm_model.predict(test_dataset)
        model_output[model_name] = test_preds
        results = evaluate_predictions(test_labels, test_preds)
        print(model_name, results)

    pred_df = pd.DataFrame.from_dict(model_output)
    print(pred_df.head())
