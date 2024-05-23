import copy
import itertools
import random
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import ParameterGrid

from datasets.dataset_torch import TorchDataset
from datasets.dataset import Dataset
from models.model import BaseModel
from util.data import flatten_nested_dict, unflatten_nested_dict

from config import KEY_LOC, KEY_YEAR, KEY_TARGET


class BaseNNModel(BaseModel, nn.Module):
    def __init__(   self, 
                    *args, **kwargs):
        super(BaseModel, self).__init__()
        super(nn.Module, self).__init__()

        self._init_args = (args, kwargs)


    def fit(self, dataset: Dataset,
            optimize_hyperparameters: bool = False,
            param_space: dict = None,
            do_kfold: bool = False,
            kfolds: int = 5,
            *args, **kwargs):
        
        if optimize_hyperparameters:
            assert param_space is not None

            best_loss = float("inf")
            best_setting = None

            settings = self._generate_settings(param_space, kwargs)
            for setting in settings:
                val_loss = None
                if do_kfold:
                    # Split data into k folds
                    all_years = dataset.years

                    list_all_years = list(all_years)
                    random.shuffle(list_all_years) # Is this seeded?
                    cv_folds = [list_all_years[i::kfolds] for i in range(kfolds)]

                    val_loss_fold = []
                    for val_fold in cv_folds:
                        # Create train and val datasets for inner fold
                        val_years = val_fold
                        train_years = [y for y in all_years if y not in val_years]
                        train_dataset, val_dataset = dataset.split_on_years((train_years, val_years))
                        new_model = self.__class__(*self._init_args[0], **self._init_args[1])
                        _, output = new_model.train(train_dataset=train_dataset, val_dataset=val_dataset, *args, **setting)
                        val_loss_fold.append(output["val_loss"])
                    val_loss = np.mean(val_loss_fold)

                else:
                    new_model = self.__class__(*self._init_args[0], **self._init_args[1])
                    _, output = new_model.fit(*args, **setting)
                    if "val_loss" not in output:
                        raise ValueError("No validation loss recorder, set val_fraction > 0 when tuning without kfold cross validation.")
                    val_loss = output["val_loss"]
                
                assert val_loss is not None

                # Store best model setting
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_setting = setting
            
            return self.fit(*args, **best_setting)
        else:
           
            model, output = self.train(*args, **kwargs)
            return model, output
        

    def train(self, 
            train_dataset: Dataset,
            val_dataset: Dataset = None,
            val_fraction: float = 0.1,
            val_every_n_epochs: int = 1,

            num_epochs: int = 1,
            batch_size: int = 10,

            loss_fn: callable = None,
            loss_kwargs: dict = None,

            optim_fn: callable = None,
            optim_kwargs: dict = None,

            scheduler_fn: callable = None,
            scheduler_kwargs: dict = None,

            device: str = None,

             **fit_params):
        """
        Fit or train the model.

        Args:
            train_dataset: Dataset,
            val_fraction: float, percentage of data to use for validation, default is 0.1
            val_every_n_epochs: int, validation frequency, default is 1
            num_epochs: int, the number of epochs to train the model, default is 1
            batch_size: int, the batch size, default is 10
            loss_fn: callable, the loss function, default is torch.nn.functional.mse_loss
            loss_kwargs: dict, additional parameters for the loss function, default is {"reduction": "mean"}
            optim_fn: callable, the optimizer function, default is torch.optim.Adam
            optim_kwargs: dict, additional parameters for the optimizer function, default is {}
            scheduler_fn: callable, the scheduler function, default is None
            scheduler_kwargs: dict, additional parameters for the scheduler function, default is {}
            device: str, the device to use, default is "cpu"

            **fit_params: Additional parameters.

        Returns:
          A tuple containing the fitted model and a dict with additional information.
        """

        # Set default values 
        if loss_fn is None: loss_fn = torch.nn.functional.mse_loss
        if loss_kwargs is None: loss_kwargs = {"reduction": "mean"}
        if optim_fn is None: optim_fn = torch.optim.Adam
        if optim_kwargs is None: optim_kwargs = {}

        if device is None: 
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Warning: Device not specified, using {device}")

        assert num_epochs > 0

        self.batch_size = batch_size
        self.to(device)
        
        # Get torchdataset, random val indices and loader
        train_dataset = TorchDataset(train_dataset)

        if val_dataset is not None:
            val_dataset = TorchDataset(val_dataset)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, collate_fn=train_dataset.collate_fn, shuffle=True, drop_last=True)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, collate_fn=val_dataset.collate_fn)
        else:
            n = len(train_dataset)
            n_val = int(n * val_fraction)
            val_ids = np.random.choice(n, n_val, replace=False).tolist()
            train_ids = list(set(range(n)) - set(val_ids))
            assert len(set(val_ids).intersection(set(train_ids))) == 0

            train_loader = torch.utils.data.DataLoader(train_dataset, 
                                                    batch_size=batch_size, 
                                                    sampler=torch.utils.data.SubsetRandomSampler(train_ids),
                                                    collate_fn=train_dataset.collate_fn)
            if val_fraction > 0:
                val_loader = torch.utils.data.DataLoader(train_dataset, 
                                                    batch_size=batch_size,
                                                    sampler=torch.utils.data.SubsetRandomSampler(val_ids),
                                                    collate_fn=train_dataset.collate_fn)
            else: val_loader = None
        

        # Load optimizer and scheduler
        optimizer = optim_fn(self.parameters(), **optim_kwargs)
        if scheduler_fn is not None: scheduler = scheduler_fn(optimizer, **scheduler_kwargs)

        # Store training set feature means and sds for normalization
        self.feature_means = {}
        self.feature_sds = {}
        all_train_samples = TorchDataset.collate_fn([train_dataset[i] for i in range(len(train_dataset))])
        for key, features in all_train_samples.items():
            if key not in [KEY_TARGET, KEY_LOC, KEY_YEAR]:
                self.feature_means[key] = features.mean()
                self.feature_sds[key] = features.std()

        # Training loop
        for epoch in range(num_epochs):
            losses = []
            self.train()
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for batch in pbar:
                # Set gradients to zero
                optimizer.zero_grad()

                for key in batch: 
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(device)

                # Forward pass
                features = {k: v for k, v in batch.items() if k != KEY_TARGET}

                # Normalize features
                for key in features:
                    if key not in [KEY_LOC, KEY_YEAR]:
                        features[key] = (features[key] - self.feature_means[key]) / self.feature_sds[key]


                predictions = self(features)
                if predictions.dim() > 1: predictions = predictions.squeeze(-1)
                target = batch[KEY_TARGET]
                loss = loss_fn(predictions, target, **loss_kwargs)

                # Backward pass
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

                pbar.set_description(
                    f"Epoch {epoch+1}/{num_epochs} | Loss: {loss.item():.4f}"
                )


            if val_loader is not None and (epoch % val_every_n_epochs == 0 or epoch == num_epochs - 1):
                self.eval()

                # Validation loop
                with torch.no_grad():
                    val_losses = []
                    tqdm_val = tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{num_epochs}")
                    for batch in tqdm_val:
                        for key in batch: 
                            if isinstance(batch[key], torch.Tensor):
                                batch[key] = batch[key].to(device)

                        features = {k: v for k, v in batch.items() if k != KEY_TARGET}
                        # Normalize features
                        for key in features:
                            if key not in [KEY_LOC, KEY_YEAR]:
                                features[key] = (features[key] - self.feature_means[key]) / self.feature_sds[key]
                        predictions = self(features)
                        if predictions.dim() > 1: predictions = predictions.squeeze(-1)
                        target = batch[KEY_TARGET]
                        loss = loss_fn(predictions, target, **loss_kwargs)

                        val_losses.append(loss.item())

                        tqdm_val.set_description(
                            f"Validation Epoch {epoch+1}/{num_epochs} | Loss: {loss.item():.4f}"
                        )
                    
            if scheduler_fn is not None: scheduler.step()
        return self, {"train_loss": np.mean(losses.detach().cpu().numpy()), "val_loss": np.mean(val_losses.detach().cpu().numpy()) if val_loader is not None else None}

    def predict_batch(self, X: list, device: str = None, batch_size: int = None):
        """Run fitted model on batched data items.

        Args:
          X: a list of data items, each of which is a dict
          device: str, the device to use, default is "cuda" if available else "cpu"
          batch_size: int, the batch size, default is self.batch_size stored during fit method
          
        Returns:
          A tuple containing a np.ndarray and a dict with additional information.
        """
        if device is None: device = "cuda" if torch.cuda.is_available() else "cpu"
        if batch_size is None: batch_size = self.batch_size

        self.to(device)
        self.eval()

        with torch.no_grad():
            predictions = np.zeros((len(X)))
            for i in range(0, len(X), batch_size):
                batch_end = min(i + batch_size, len(X))
                batch = X[i:batch_end]
                batch = TorchDataset.collate_fn([TorchDataset._cast_to_tensor(sample) for sample in batch])
                batch = {key: batch[key].to(device) for key in batch.keys() if isinstance(batch[key], torch.Tensor)}
                features = {k: v for k, v in batch.items() if k != KEY_TARGET}
                for key in features:
                    if key not in [KEY_LOC, KEY_YEAR]:
                        features[key] = (features[key] - self.feature_means[key]) / self.feature_sds[key]
                y_pred = self(features)
                if y_pred.dim() > 1: y_pred = y_pred.squeeze(-1)
                y_pred = y_pred.cpu().numpy()
                predictions[i:i + len(y_pred)] = y_pred
            return predictions, {}
        

    def save(self, model_name):
        """Save model using torch.save.

        Args:
          model_name: Filename that will be used to save the model.
        """
        torch.save(self, model_name)

    @classmethod
    def load(cls, model_name):
        """Load model using torch.load.
        
        Args:
            model_name: Filename that was used to save the model.

        Returns:
            The loaded model.
        """
        return torch.load(model_name)
    
    
    def _generate_settings(self,
                           param_space: dict,
                           standard_settings: dict):
        
        # Flatten both dicts
        param_space = flatten_nested_dict(param_space)
        standard_settings = flatten_nested_dict(standard_settings)

        # Generate all combinations of parameter values
        combs = list(ParameterGrid(param_space))

        # For each comb, deepcopy standard_settings and replace values
        settings = []
        for comb in combs:
            setting = copy.deepcopy(standard_settings)
            setting.update(comb)
            settings.append(setting)

        # Unflatten settings
        settings = [unflatten_nested_dict(setting) for setting in settings]

        return settings
    

class ExampleLSTM(BaseNNModel):
    def __init__(self, n_ts_features, n_static_features, hidden_size, num_layers, output_size=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lstm = nn.LSTM(n_ts_features, hidden_size, num_layers, batch_first=True)
        self._fc = nn.Linear(hidden_size + n_static_features, output_size)

    def forward(self, x):
        max_n_dim = max([len(v.shape) for k, v in x.items() if isinstance(v, torch.Tensor)])
        x_ts = torch.cat([v.unsqueeze(2) for k, v in x.items() if isinstance(v, torch.Tensor) and len(v.shape) == max_n_dim], dim=2)
        x_static = torch.cat([v.unsqueeze(1) for k, v in x.items() if isinstance(v, torch.Tensor) and len(v.shape) < max_n_dim], dim=1)

        print(x_ts.mean(), x_ts.std())
        print(x_static.mean(), x_static.std())

        x_ts, _ = self._lstm(x_ts)
        x = torch.cat([x_ts[:, -1, :], x_static], dim=1)
        x = self._fc(x)
        return x


        


