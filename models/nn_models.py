import copy
import random
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import logging

from datasets.dataset_torch import TorchDataset
from datasets.dataset import Dataset
from datasets.transforms import transform_ts_features_to_dekadal, transform_stack_ts_static_features 
from models.model import BaseModel
from util.data import flatten_nested_dict, unflatten_nested_dict, generate_settings

from config import KEY_LOC, KEY_YEAR, KEY_TARGET, KEY_DATES


class BaseNNModel(BaseModel, nn.Module):
    def __init__(self, **kwargs):
        super(BaseModel, self).__init__()
        super(nn.Module, self).__init__()

        self._init_args = kwargs
        self._logger = logging.getLogger(__name__)

    def fit(self, dataset: Dataset,
            optimize_hyperparameters: bool = False,
            param_space: dict = None,
            do_kfold: bool = False,
            kfolds: int = 5,
            *args, **kwargs):
        
        # Set seed if seed is provided
        if "seed" in kwargs:
            seed = kwargs["seed"]
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        self.best_model = None

        if optimize_hyperparameters:
            assert param_space is not None

            best_loss = float("inf")
            best_setting = None

            settings = generate_settings(param_space, kwargs)
            for i, setting in enumerate(settings):
                val_loss = None
                if do_kfold:
                    # Split data into k folds
                    all_years = dataset.years
                    list_all_years = list(all_years)
                    random.shuffle(list_all_years)
                    cv_folds = [list_all_years[i::kfolds] for i in range(kfolds)]

                    # For each fold, create new model and datasets, train and record val loss. Finally, average val loss.
                    val_loss_fold = []
                    for j, val_fold in enumerate(cv_folds):
                        self._logger.debug(f"Running inner fold {j+1}/{kfolds} for hyperparameter setting {i+1}/{len(settings)}")
                        val_years = val_fold
                        train_years = [y for y in all_years if y not in val_years]
                        train_dataset, val_dataset = dataset.split_on_years((train_years, val_years))
                        new_model = self.__class__(**self._init_args)
                        _, output = new_model.train_model(train_dataset=train_dataset, val_dataset=val_dataset, *args, **setting)
                        
                        # If early stopping is used, use best val loss. Otherwise, use final val loss.
                        if output["best_val_loss"] is not None:
                            val_loss_fold.append(output["best_val_loss"])
                        else:
                            val_loss_fold.append(output["val_loss"])
                    val_loss = np.mean(val_loss_fold)

                else:
                    # Train new model with single randomly sampled validation set
                    self._logger.debug(f"Running setting {i+1}/{len(settings)}")
                    new_model = self.__class__(**self._init_args)
                    _, output = new_model.fit(dataset=dataset, *args, **setting)
                    if "val_loss" not in output:
                        raise ValueError("No validation loss recorder, set val_fraction > 0 when tuning without kfold cross validation.")
                    
                    # If early stopping is used, use best val loss. Otherwise, use final val loss.
                    if output["best_val_loss"] is not None:
                        val_loss = output["best_val_loss"]
                    else:
                        val_loss = output["val_loss"]
                assert val_loss is not None

                self._logger.debug(f"For setting {i+1}/{len(settings)}, average validation loss: {val_loss}")
                self._logger.debug(f"Settings: {setting}")

                # Store best model setting
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_setting = setting
            
            # Finally, train model with best setting
            final_model, final_output = self.fit(dataset=dataset, *args, **best_setting)
            self._logger.debug(f"Final validation loss of outer fold: {final_output['val_loss']}")
            self._logger.debug(f"Final best setting of outer fold: {best_setting}")
            return final_model, final_output
        else:
           
            model, output = self.train_model(dataset, *args, **kwargs)

            # If early stopping is used, use best val 
            # loss and save best model for prediction
            if output["best_val_loss"] is not None:
                self.best_model = output["best_model"]
                return self.best_model, output
            else:
                return model, output
        

    def train_model(self, 
            train_dataset: Dataset,
            val_dataset: Dataset = None,
            val_fraction: float = 0.1,
            val_split_by_year: bool = False,
            val_every_n_epochs: int = 1,
            do_early_stopping: bool = False,
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
            val_dataset: Dataset, default is None. If None, val_fraction is used to split train_dataset into train and val.
            val_fraction: float, percentage of data to use for validation, default is 0.1
            val_split_by_year: bool, whether to split validation data by year, default is False
            val_every_n_epochs: int, validation frequency, default is 1
            do_early_stopping: bool, whether to use early stopping, default is False
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
        if loss_fn is None:
            loss_fn = torch.nn.functional.mse_loss
        if loss_kwargs is None:
            loss_kwargs = {"reduction": "mean"}
        if optim_fn is None:
            optim_fn = torch.optim.Adam
        if optim_kwargs is None:
            optim_kwargs = {}

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._logger.debug(f"Warning: Device not specified, using {device}")

        assert num_epochs > 0

        self.batch_size = batch_size
        self.to(device)

        if 'seed' in fit_params.keys():
            random.seed(fit_params['seed'])

        if val_dataset is not None:
            train_dataset = TorchDataset(train_dataset)
            val_dataset = TorchDataset(val_dataset)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, collate_fn=train_dataset.collate_fn, shuffle=True, drop_last=True)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, collate_fn=val_dataset.collate_fn)
        elif val_fraction > 0 and not val_split_by_year:
            n = len(train_dataset)
            n_val = int(n * val_fraction)
            val_ids = np.random.choice(n, n_val, replace=False).tolist()
            train_ids = list(set(range(n)) - set(val_ids))
            assert len(set(val_ids).intersection(set(train_ids))) == 0

            train_dataset = TorchDataset(train_dataset)

            train_loader = torch.utils.data.DataLoader(train_dataset, 
                                                    batch_size=batch_size, 
                                                    sampler=torch.utils.data.SubsetRandomSampler(train_ids),
                                                    collate_fn=train_dataset.collate_fn)
            if val_fraction > 0:
                val_loader = torch.utils.data.DataLoader(train_dataset, 
                                                    batch_size=batch_size,
                                                    sampler=torch.utils.data.SubsetRandomSampler(val_ids),
                                                    collate_fn=train_dataset.collate_fn)
        elif val_fraction > 0 and val_split_by_year:
            all_years = train_dataset.years
            list_all_years = list(all_years)
            random.shuffle(list_all_years)
            n_val = int(np.ceil(len(all_years) * val_fraction))
            val_years = list_all_years[:n_val]
            train_years = list_all_years[n_val:]
            self._logger.debug(f"Validation years: {val_years}")
            self._logger.debug(f"Training years: {train_years}")
            train_dataset, val_dataset = train_dataset.split_on_years((train_years, val_years))
            train_dataset = TorchDataset(train_dataset)
            val_dataset = TorchDataset(val_dataset)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, collate_fn=train_dataset.collate_fn, shuffle=True, drop_last=True)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, collate_fn=val_dataset.collate_fn)
        else: 
            train_dataset = TorchDataset(train_dataset)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, collate_fn=train_dataset.collate_fn, shuffle=True, drop_last=True)
            val_loader = None

        # Load optimizer and scheduler
        optimizer = optim_fn(self.parameters(), **optim_kwargs)
        if scheduler_fn is not None:
            scheduler = scheduler_fn(optimizer, **scheduler_kwargs)

        # Store training set feature means and sds for normalization
        self.feature_means = {}
        self.feature_sds = {}
        if val_dataset is not None:
            all_train_samples = TorchDataset.collate_fn([train_dataset[i] for i in range(len(train_dataset))] + [val_dataset[i] for i in range(len(val_dataset))])
        else:
            all_train_samples = TorchDataset.collate_fn([train_dataset[i] for i in range(len(train_dataset))])
        for key, features in all_train_samples.items():
            if key not in [KEY_TARGET, KEY_LOC, KEY_YEAR, KEY_DATES]:
                self.feature_means[key] = features.mean()
                self.feature_sds[key] = features.std()

        all_train_losses = []
        all_val_losses = []

        best_val_loss = float("inf")
        best_model = None

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
                    if key not in [KEY_LOC, KEY_YEAR, KEY_DATES]:
                        features[key] = (features[key] - self.feature_means[key]) / self.feature_sds[key]


                predictions = self(features)
                if predictions.dim() > 1:
                    predictions = predictions.squeeze(-1)
                target = batch[KEY_TARGET]
                loss = loss_fn(predictions, target, **loss_kwargs)

                # Backward pass
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

                mean_train_loss = np.mean(losses)

                pbar.set_description(
                    f"Epoch {epoch+1}/{num_epochs} | Loss: {mean_train_loss:.4f}"
                )
            all_train_losses.append(mean_train_loss)


            if val_loader is not None and (epoch % val_every_n_epochs == 0 or epoch == num_epochs - 1):
                self.eval()

                # Validation loop
                with torch.no_grad():
                    val_losses = []
                    tqdm_val = tqdm(
                        val_loader, desc=f"Validation Epoch {epoch+1}/{num_epochs}"
                    )
                    for batch in tqdm_val:
                        for key in batch:
                            if isinstance(batch[key], torch.Tensor):
                                batch[key] = batch[key].to(device)

                        features = {k: v for k, v in batch.items() if k != KEY_TARGET}
                        # Normalize features
                        for key in features:
                            if key not in [KEY_LOC, KEY_YEAR, KEY_DATES]:
                                features[key] = (features[key] - self.feature_means[key]) / self.feature_sds[key]
                        predictions = self(features)
                        if predictions.dim() > 1:
                            predictions = predictions.squeeze(-1)
                        target = batch[KEY_TARGET]
                        loss = loss_fn(predictions, target, **loss_kwargs)

                        val_losses.append(loss.item())
                        mean_val_loss = np.mean(val_losses)

                        tqdm_val.set_description(
                            f"Validation Epoch {epoch+1}/{num_epochs} | Loss: {mean_val_loss:.4f}"
                        )
                    all_val_losses.append(mean_val_loss)
                    if mean_val_loss < best_val_loss and do_early_stopping:
                        best_val_loss = mean_val_loss
                        best_model = copy.deepcopy(self)

            if scheduler_fn is not None:
                scheduler.step()
        return self, {
            "train_loss": np.mean(losses), 
            "val_loss": np.mean(val_losses) if val_loader is not None else None,
            "train_losses": all_train_losses,
            "val_losses": all_val_losses if val_loader is not None else None,
            "best_val_loss": best_val_loss,
            "best_val_epoch": np.argmin(all_val_losses) if val_loader is not None else None,
            "best_model": best_model,
            "train_years": train_years if val_split_by_year else None,
            "val_years": val_years if val_split_by_year else None,
            }

    def predict_batch(self, X: list, device: str = None, batch_size: int = None):
        """Run fitted model on batched data items.

        Args:
          X: a list of data items, each of which is a dict
          device: str, the device to use, default is "cuda" if available else "cpu"
          batch_size: int, the batch size, default is self.batch_size stored during fit method

        Returns:
          A tuple containing a np.ndarray and a dict with additional information.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if batch_size is None:
            batch_size = self.batch_size

        if self.best_model is not None:
            # Log
            self._logger.debug("Using best model from early stopping for prediction")
            model = self.best_model
        else:
            model = self

        model.to(device)
        model.eval()

        with torch.no_grad():
            predictions = np.zeros((len(X)))
            for i in range(0, len(X), batch_size):
                batch_end = min(i + batch_size, len(X))
                batch = X[i:batch_end]
                batch = TorchDataset.collate_fn(
                    [TorchDataset._cast_to_tensor(sample) for sample in batch]
                )
                batch = {
                    key: batch[key].to(device)
                    for key in batch.keys()
                    if isinstance(batch[key], torch.Tensor)
                }
                features = {k: v for k, v in batch.items() if k != KEY_TARGET}
                for key in features:
                    if key not in [KEY_LOC, KEY_YEAR, KEY_DATES]:
                        features[key] = (features[key] - self.feature_means[key]) / self.feature_sds[key]
                y_pred = model(features)
                if y_pred.dim() > 1:
                    y_pred = y_pred.squeeze(-1)
                y_pred = y_pred.cpu().numpy()
                predictions[i : i + len(y_pred)] = y_pred
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
    

class ExampleLSTM(BaseNNModel):
    def __init__(
        self, 
        n_ts_features, 
        n_static_features, 
        hidden_size, 
        num_layers, 
        output_size=1,
        transforms=[
            transform_ts_features_to_dekadal,
            transform_stack_ts_static_features
        ],
        **kwargs):
        
        # Add all arguments to init_args to enable model reconstruction in fit method
        kwargs["n_ts_features"] = n_ts_features
        kwargs["n_static_features"] = n_static_features
        kwargs["hidden_size"] = hidden_size
        kwargs["num_layers"] = num_layers
        kwargs["output_size"] = output_size

        super().__init__(**kwargs)
        self._lstm = nn.LSTM(n_ts_features, hidden_size, num_layers, batch_first=True)
        self._fc = nn.Linear(hidden_size + n_static_features, output_size)
        self._transforms = transforms

    def forward(self, x):
        for transform in self._transforms: x = transform(x)
        x_ts = x["ts"]
        x_static = x["static"]
        x_ts, _ = self._lstm(x_ts)
        x = torch.cat([x_ts[:, -1, :], x_static], dim=1)
        x = self._fc(x)
        return x
    


