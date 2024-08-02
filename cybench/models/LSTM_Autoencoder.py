import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as tf

from cybench.datasets.dataset import Dataset
from cybench.datasets.dataset_torch import TorchDataset
from cybench.evaluation.eval import evaluate_predictions
from cybench.util.features import dekad_from_date
from cybench.config import TIME_SERIES_PREDICTORS

# Copied from https://github.com/JulesBelveze/time-series-autoencoder/tree/master

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def init_hidden(x: torch.Tensor, hidden_size: int, num_dir: int = 1, xavier: bool = True):
    """
    Initialize hidden.

    Args:
        x: (torch.Tensor): input tensor
        hidden_size: (int):
        num_dir: (int): number of directions in LSTM
        xavier: (bool): wether or not use xavier initialization
    """
    if xavier:
        return nn.init.xavier_normal_(torch.zeros(num_dir, x.size(0), hidden_size)).to(device)

    return Variable(torch.zeros(num_dir, x.size(0), hidden_size)).to(device)


###########################################################################
################################ ENCODERS #################################
###########################################################################

class Encoder(nn.Module):
    def __init__(self, input_size: int):
        """
        Initialize the model.

        Args:
            config:
            input_size: (int): size of the input
        """
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = 64
        self.seq_len = 10
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=64)

    def forward(self, input_data: torch.Tensor):
        """
        Run forward computation.

        Args:
            input_data: (torch.Tensor): tensor of input daa
        """
        h_t, c_t = (init_hidden(input_data, self.hidden_size),
                    init_hidden(input_data, self.hidden_size))
        input_encoded = Variable(torch.zeros(input_data.size(0), self.seq_len, self.hidden_size))

        for t in range(self.seq_len):
            _, (h_t, c_t) = self.lstm(input_data[:, t, :].unsqueeze(0), (h_t, c_t))
            input_encoded[:, t, :] = h_t
        return _, input_encoded


class AttnEncoder(nn.Module):
    def __init__(self, input_size: int):
        """
        Initialize the network.

        Args:
            config:
            input_size: (int): size of the input
        """
        super(AttnEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = 64
        self.seq_len = 10
        self.add_noise = False
        self.directions = 1
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=1
        )
        self.attn = nn.Linear(
            in_features=2 * self.hidden_size + self.seq_len,
            out_features=1
        )
        self.softmax = nn.Softmax(dim=1)

    def _get_noise(self, input_data: torch.Tensor, sigma=0.01, p=0.1):
        """
        Get noise.

        Args:
            input_data: (torch.Tensor): tensor of input data
            sigma: (float): variance of the generated noise
            p: (float): probability to add noise
        """
        normal = sigma * torch.randn(input_data.shape)
        mask = np.random.uniform(size=(input_data.shape))
        mask = (mask < p).astype(int)
        noise = normal * torch.tensor(mask)
        return noise

    def forward(self, input_data: torch.Tensor):
        """
        Forward computation.

        Args:
            input_data: (torch.Tensor): tensor of input data
        """
        h_t, c_t = (init_hidden(input_data, self.hidden_size, num_dir=self.directions),
                    init_hidden(input_data, self.hidden_size, num_dir=self.directions))

        attentions, input_encoded = (Variable(torch.zeros(input_data.size(0), self.seq_len, self.input_size)),
                                     Variable(torch.zeros(input_data.size(0), self.seq_len, self.hidden_size)))

        if self.add_noise and self.training:
            input_data += self._get_noise(input_data).to(device)

        for t in range(self.seq_len):
            x = torch.cat((h_t.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           c_t.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           input_data.permute(0, 2, 1).to(device)), dim=2).to(
                device)  # bs * input_size * (2 * hidden_dim + seq_len)

            e_t = self.attn(x.view(-1, self.hidden_size * 2 + self.seq_len))  # (bs * input_size) * 1
            a_t = self.softmax(e_t.view(-1, self.input_size)).to(device)  # (bs, input_size)

            weighted_input = torch.mul(a_t, input_data[:, t, :].to(device))  # (bs * input_size)
            self.lstm.flatten_parameters()
            _, (h_t, c_t) = self.lstm(weighted_input.unsqueeze(0), (h_t, c_t))

            input_encoded[:, t, :] = h_t
            attentions[:, t, :] = a_t

        return attentions, input_encoded


###########################################################################
################################ DECODERS #################################
###########################################################################

class Decoder(nn.Module):
    def __init__(self, output_size):
        """
        Initialize the network.

        Args:
            config:
        """
        super(Decoder, self).__init__()
        self.seq_len = 10
        self.hidden_size = 64
        self.lstm = nn.LSTM(1, 64, bidirectional=False)
        self.fc = nn.Linear(64, output_size)

    def forward(self, _, y_hist: torch.Tensor):
        """
        Forward pass

        Args:
            _:
            y_hist: (torch.Tensor): shifted target
        """
        h_t, c_t = (init_hidden(y_hist, self.hidden_size),
                    init_hidden(y_hist, self.hidden_size))

        for t in range(self.seq_len):
            inp = y_hist[:, t].unsqueeze(0).unsqueeze(2)
            lstm_out, (h_t, c_t) = self.lstm(inp, (h_t, c_t))
        return self.fc(lstm_out.squeeze(0))


class AttnDecoder(nn.Module):
    def __init__(self, output_size):
        """
        Initialize the network.

        Args:
            config:
        """
        super(AttnDecoder, self).__init__()
        self.seq_len = 10
        self.encoder_hidden_size = 64
        self.decoder_hidden_size = 64
        self.out_feats = output_size

        self.attn = nn.Sequential(
            nn.Linear(2 * self.decoder_hidden_size + self.encoder_hidden_size, self.encoder_hidden_size),
            nn.Tanh(),
            nn.Linear(self.encoder_hidden_size, 1)
        )
        self.lstm = nn.LSTM(input_size=self.out_feats, hidden_size=self.decoder_hidden_size)
        self.fc = nn.Linear(self.encoder_hidden_size + self.out_feats, self.out_feats)
        self.fc_out = nn.Linear(self.decoder_hidden_size + self.encoder_hidden_size, self.out_feats)
        self.fc.weight.data.normal_()

    def forward(self, input_encoded: torch.Tensor, y_history: torch.Tensor):
        """
        Perform forward computation.

        Args:
            input_encoded: (torch.Tensor): tensor of encoded input
            y_history: (torch.Tensor): shifted target
        """
        h_t, c_t = (
            init_hidden(input_encoded, self.decoder_hidden_size), init_hidden(input_encoded, self.decoder_hidden_size))
        context = Variable(torch.zeros(input_encoded.size(0), self.encoder_hidden_size))

        for t in range(self.seq_len):
            x = torch.cat((h_t.repeat(self.seq_len, 1, 1).permute(1, 0, 2),
                           c_t.repeat(self.seq_len, 1, 1).permute(1, 0, 2),
                           input_encoded.to(device)), dim=2)

            x = tf.softmax(
                self.attn(
                    x.view(-1, 2 * self.decoder_hidden_size + self.encoder_hidden_size)
                ).view(-1, self.seq_len),
                dim=1)

            context = torch.bmm(x.unsqueeze(1), input_encoded.to(device))[:, 0, :]  # (batch_size, encoder_hidden_size)

            y_tilde = self.fc(torch.cat((context.to(device), y_history[:, t].to(device)),
                                        dim=1))  # (batch_size, out_size)

            self.lstm.flatten_parameters()
            _, (h_t, c_t) = self.lstm(y_tilde.unsqueeze(0), (h_t, c_t))

        return self.fc_out(torch.cat((h_t[0], context.to(device)), dim=1))  # predicting value at t=self.seq_length+1


class AutoEncoderForecast(nn.Module):
    def __init__(self, input_size):
        """
        Initialize the network.

        Args:
            config:
            input_size: (int): size of the input
        """
        super(AutoEncoderForecast, self).__init__()
        self.encoder = AttnEncoder(input_size).to(device)
        self.decoder = AttnDecoder(input_size).to(device)

    def forward(self, encoder_input: torch.Tensor, y_hist: torch.Tensor, return_attention: bool = False):
        """
        Forward computation. encoder_input_inputs.

        Args:
            encoder_input: (torch.Tensor): tensor of input data
            y_hist: (torch.Tensor): shifted target
            return_attention: (bool): whether or not to return the attention
        """
        attentions, encoder_output = self.encoder(encoder_input)
        outputs = self.decoder(encoder_output, y_hist.float())

        if return_attention:
            return outputs, attentions
        return outputs
  

def transform_ts_input_to_dekadal(ts_key, value, dates, min_date, max_date):
    # Transform dates to dekads
    min_dekad = dekad_from_date(min_date)
    max_dekad = dekad_from_date(max_date)
    dekads = list(range(0, max_dekad - min_dekad + 1))

    datestrings = [str(date) for date in dates]
    value_dekads = torch.tensor(
        [dekad_from_date(date) for date in datestrings], device=value.device
    )
    value_dekads -= 1

    # Aggregate timeseries to dekadal resolution
    new_value = torch.full(
        (value.shape[0], len(dekads)), float("inf"), dtype=value.dtype
    )
    for d in dekads:
        # Inefficient, but scatter_min or scatter_max are not supported by torch
        mask = value_dekads == d
        if value[:, mask].shape[1] == 0:
            new_value[:, d] = 0.0
        else:
            if ts_key in ["tmax"]:  # Max aggregation
                new_value[:, d] = value[:, mask].max(dim=1).values
            elif ts_key in ["tmin"]:  # Min aggregation
                new_value[:, d] = value[:, mask].min(dim=1).values
            else:  # for all other inputs
                new_value[:, d] = torch.mean(value[:, mask], dim=1)
    return new_value

def train(train_iter, test_iter,
          min_date, max_date,
          seq_length, prediction_window,
          model, criterion, optimizer):
    """
    Training function.

    Args:
        train_iter: (DataLoader): train data iterator
        test_iter: (DataLoader): test data iterator
        model: model
    """

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

    global_step = 0
    train_loss = 0.0
    for epoch in tqdm(range(5), unit="epoch"):
        for i, batch in tqdm(enumerate(train_iter), total=len(train_iter), unit="batch"):
            ts_inputs = {}
            for k in batch:
                if (k in TIME_SERIES_PREDICTORS):
                    ts_inputs[k] = transform_ts_input_to_dekadal(k, batch[k], batch["dates"][k], min_date, max_date)
            ts_inputs = torch.cat([v.unsqueeze(2) for k, v in ts_inputs.items()], dim=2).squeeze(0)
            ts_seq_len = ts_inputs.shape[0]
            X, y_hist, y = [], [], []
            for i in range(prediction_window, ts_seq_len - seq_length - prediction_window):
                X.append(torch.FloatTensor(ts_inputs[i:i + seq_length, :]).unsqueeze(0))
                y_hist.append(torch.FloatTensor(ts_inputs[i - 1: i + seq_length - 1, :]).unsqueeze(0))
                y.append(torch.FloatTensor(ts_inputs[i + seq_length:i + seq_length + prediction_window, :]))

            X = torch.cat(X)
            y_hist = torch.cat(y_hist)
            y = torch.cat(y)

            model.train()
            optimizer.zero_grad()

            output = model(X.to(device), y_hist.to(device))
            loss = criterion(output.to(device), y.to(device))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            train_loss += loss.item()

            optimizer.step()
            scheduler.step()
            global_step += 1

def test_autoencoder(seq_length = 10, prediction_window = 1):
    dataset = Dataset.load("maize_NL")
    all_years = sorted(dataset.years)
    test_years = [all_years[-1]]
    train_years = all_years[:-1]
    train_dataset, test_dataset = dataset.split_on_years((train_years, test_years))
    min_date, max_date = train_dataset.min_date, train_dataset.max_date
    train_dataset = TorchDataset(train_dataset)
    test_dataset = TorchDataset(test_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        collate_fn=TorchDataset.collate_fn,
        shuffle=True,
        drop_last=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, collate_fn=TorchDataset.collate_fn
    )
    model = AutoEncoderForecast(input_size=len(TIME_SERIES_PREDICTORS)).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.00001)
    train(train_loader, test_loader,
          min_date, max_date, seq_length, prediction_window,
          model, criterion, optimizer)

test_autoencoder()

# air_quality_df = pd.read_csv("C:/Users/paude006/Documents/git-repos/AgML-crop-yield-forecasting/cybench/data/AirQualityUCI.csv",
#                              header=0, index_col="Date_Time")
# X = air_quality_df.values
# print(X.shape)
# features, target, y_hist = [], [], []
# nb_obs, nb_features = X.shape
# seq_length = 10
# prediction_window = 1

# for i in range(1, nb_obs - seq_length - prediction_window):
#     features.append(torch.FloatTensor(X[i:i + seq_length, :]).unsqueeze(0))
#     y_hist.append(torch.FloatTensor(X[i - 1: i + seq_length - 1, :]).unsqueeze(0))
#     target.append(torch.FloatTensor(X[i + seq_length:i + seq_length + prediction_window, :]))

# features = torch.cat(features)
# print(features.shape)
# y_hist = torch.cat(y_hist)
# print(y_hist.shape)
# target = torch.cat(target)
# print(target.shape)

# train_dataset = torch.utils.data.TensorDataset(features, y_hist, target)
# train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=False, drop_last=True)
# for batch in train_iter:
#     features, y_hist, y = batch
#     print(features.shape, y_hist.shape, y.shape)
#     break
