import logging
import pandas as pd
import torch
from torch import nn

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from models.model import BaseModel
from datasets.torch_dataset import TorchDataset
from util.data import normalize_data
from config import LOGGER_NAME


class LSTMModel(BaseModel, nn.Module):
    def __init__(
        self,
        num_ts_inputs,
        num_trend_features,
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

        num_all_features = rnn_hidden_size + num_trend_features + num_other_features
        self._batch_norm2 = nn.BatchNorm1d(num_all_features, dtype=torch.double)
        self._fc = nn.Linear(num_all_features, num_outputs, dtype=torch.double)
        self._max_epochs = 10
        self._logger = logging.getLogger(LOGGER_NAME)
        # self._norm_params = None
        # self._normalization = "standard"

    def fit(self, train_dataset, epochs=None):
        self.train()
        label_col = train_dataset.labelCol
        all_inputs = train_dataset.featureCols
        ts_inputs = train_dataset.timeSeriesCols
        other_features = [c for c in all_inputs if (c not in ts_inputs)]
        batch_size = 16

        # self._norm_params = train_dataset.get_normalization_params(normalization=self._normalization)
        torch_dataset = TorchDataset(train_dataset)
        data_loader = torch.utils.data.DataLoader(
            torch_dataset,
            collate_fn=torch_dataset.collate_fn,
            shuffle=True,
            batch_size=batch_size,
        )

        if epochs is None:
            epochs = self._max_epochs

        loss = nn.MSELoss()
        trainer = torch.optim.Adam(self.parameters(), lr=0.0001, weight_decay=0.0001)

        for epoch in range(epochs):
            epoch_loss = 0
            num_elems = 0
            y_all = None
            y_hat_all = None
            for batch in data_loader:
                # batch = normalize_data(batch, self._norm_params, normalization=self._normalization)
                y = torch.unsqueeze(batch[label_col], 1)
                X_ts = torch.cat(
                    [torch.unsqueeze(batch[c], 1) for c in ts_inputs], dim=1
                )
                X_rest = torch.cat([batch[c] for c in other_features])
                if len(X_rest.shape) == 1:
                    X_rest = torch.unsqueeze(X_rest, 1)

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

            nrmse = torch.sqrt(torch.mean((y_hat_all - y_all) ** 2)) / torch.mean(y_all)
            self._logger.info(
                "LSTMModel epoch:%d, loss:%f, NRMSE:%f",
                epoch,
                epoch_loss / num_elems,
                nrmse.item(),
            )

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
            X_rest = torch.cat([batch[c] for c in other_features])
            if len(X_rest.shape) == 1:
                X_rest = torch.unsqueeze(X_rest, 1)

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

        # y_true = predictions_df[label_col].values
        # y_pred = predictions_df["PREDICTION"].values
        # nrmse = 100 * np.sqrt(mean_squared_error(y_true, y_pred)) / np.mean(y_true)
        # print("Normalized RMSE:", nrmse)

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

    all_years = [yr for yr in range(2000, 2019)]
    test_years = [2012, 2018]
    train_years = [yr for yr in all_years if yr not in test_years]
    dataset = CropYieldDataset(
        data_sources,
        spatial_id_col="COUNTY_ID",
        year_col="FYEAR",
        data_path=data_path,
        lead_time=6,
    )

    train_dataset, test_dataset = dataset.split_on_years((train_years, test_years))
    all_inputs = train_dataset.featureCols
    ts_inputs = train_dataset.timeSeriesCols
    trend_features = [c for c in all_inputs if "YIELD-" in c]
    other_features = [
        c for c in all_inputs if ((c not in ts_inputs) and ("YIELD-" not in c))
    ]

    lstm_model = LSTMModel(
        num_ts_inputs=len(ts_inputs),
        num_trend_features=len(trend_features),
        num_other_features=len(other_features),
    )
    lstm_model.fit(train_dataset, epochs=10)

    test_preds = lstm_model.predict(test_dataset)
    print(test_preds.head(5).to_string())

    output_path = os.path.join(PATH_OUTPUT_DIR, "saved_models")
    os.makedirs(output_path, exist_ok=True)

    # Test saving and loading
    lstm_model.save(output_path + "/saved_lstm_model.pkl")
    saved_model = LSTMModel.load(output_path + "/saved_lstm_model.pkl")
    test_preds = saved_model.predict(test_dataset)
    print("\n")
    print("Predictions of saved model. Should match earlier output.")
    print(test_preds.head(5).to_string())
