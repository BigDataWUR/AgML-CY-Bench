import copy
import random
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import logging

from sklearn.model_selection import ParameterGrid

from cybench.datasets.dataset_torch import TorchDataset
from cybench.datasets.dataset import Dataset
from cybench.datasets.transforms import (
    transform_ts_inputs_to_dekadal,
    transform_stack_ts_static_inputs,
)

from cybench.models.model import BaseModel
from cybench.evaluation.eval import evaluate_predictions

from cybench.config import (
    KEY_TARGET,
    STATIC_PREDICTORS,
    TIME_SERIES_PREDICTORS,
    ALL_PREDICTORS,
)


class EarlyStopping:
    """Based on the answer to
    https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
    """
    def __init__(self, patience=3, min_delta=0.0):
        self._patience = patience
        self._min_delta = min_delta
        self._counter = 0
        self._opt_metric = float("inf")

    def early_stop(self, val_metric):
        if val_metric < self._opt_metric:
            self._opt_metric = val_metric
            self._counter = 0
        elif val_metric > (self._opt_metric + self._min_delta):
            self._counter += 1
            if self._counter >= self._patience:
                return True

        return False

    @property
    def patience(self):
        return self._patience


class BaseNNModel(BaseModel, nn.Module):
    def __init__(self, **kwargs):
        super(BaseModel, self).__init__()
        super(nn.Module, self).__init__()

        self._norm_params = None
        self._init_args = kwargs
        self._early_stopper = EarlyStopping(patience=4, min_delta=0.05)
        self._logger = logging.getLogger(__name__)

    def fit(
        self,
        dataset: Dataset,
        optimize_hyperparameters: bool = False,
        param_space: dict = None,
        do_kfold: bool = False,
        kfolds: int = 5,
        **kwargs,
    ):
        """Fit or train the model.

        Args:
          dataset: Dataset. Training dataset.

          optimize_hyperparameters: bool

          param_space: dict. Each entry is a hyperparameter name and list or range of values.

          do_kfold: bool

          kfolds: int. k in k-fold cv.

          **fit_params: Additional parameters.

        Returns:
          A tuple containing the fitted model and a dict with additional information.
        """
        # Set seed if seed is provided
        if "seed" in kwargs:
            seed = kwargs["seed"]
            assert seed is not None
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        opt_param_setting = {}
        if (len(dataset.years) > 1) and optimize_hyperparameters:
            opt_param_setting = self._optimize_hyperparameters(dataset, param_space, do_kfold, kfolds, **kwargs)

        output = self._train_model(dataset, **opt_param_setting, **kwargs)

        return self, output

    def _optimize_hyperparameters(self, dataset: Dataset, do_kfold: bool = False, kfolds: int = 5,
                                  param_space: dict = None, **kwargs) -> dict:
        """Optimize hyperparameters

        Args:
          dataset: Dataset. Training dataset.

          param_space: a dict of parameters to optimize

          do_kfold: bool. Flag for whether or not to run k-fold cv.

          kfolds: k for k-fold cv.

        Returns:
          A dict of optimal hyperparameter setting
        """
        
        assert param_space is not None

        opt_loss = float("inf")
        opt_param_setting = {}
        all_years = list(dataset.years)
        random.shuffle(all_years)
        settings = list(ParameterGrid(param_space))
        for i, setting in enumerate(settings):
            if do_kfold and (kfolds > 1):
                # Split data into k folds
                # For each fold, create new model and datasets, train and record val loss.
                # Finally, average val loss.
                cv_years = [all_years[i::kfolds] for i in range(kfolds)]
                cv_losses = []
                for j, val_years in enumerate(cv_years):
                    self._logger.debug(
                        f"Running inner fold {j + 1}/{kfolds} for hyperparameter setting {i + 1}/{len(settings)}"
                    )
                    train_years = [y for y in all_years if y not in val_years]
                    train_dataset, val_dataset = dataset.split_on_years(
                        (train_years, val_years)
                    )
                    new_model = self.__class__(**self._init_args)
                    _, output = new_model._train_model(
                        train_dataset=train_dataset,
                        val_dataset=val_dataset,
                        **setting,
                        **kwargs,
                    )

                    cv_losses.append(output["val_loss"])

                val_loss = np.mean(cv_losses)
            else:
                # Train new model with single randomly sampled validation set
                self._logger.debug(f"Running setting {i + 1}/{len(settings)}")
                val_years = all_years[len(all_years) // 2:]

                train_years = [y for y in all_years if y not in val_years]
                train_dataset, val_dataset = dataset.split_on_years(
                        (train_years, val_years)
                )
                new_model = self.__class__(**self._init_args)
                _, output = new_model._train_model(
                    train_dataset=train_dataset,
                    val_dataset=val_dataset,
                    **setting,
                    **kwargs,
                )
                val_loss = output["val_loss"]

            assert val_loss is not None

            self._logger.debug(
                f"For setting {i + 1}/{len(settings)}, average validation loss: {val_loss}"
            )
            self._logger.debug(f"Settings: {setting}")

            # Store best model setting
            if val_loss < opt_loss:
                opt_loss = val_loss
                opt_param_setting = setting

        return opt_param_setting

    def train_model(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset = None,
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
        device: str = "cpu",
        **kwargs,
    ):
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

        if optim_fn is None:
            optim_fn = torch.optim.Adam
        if optim_kwargs is None:
            optim_kwargs = {}

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._logger.debug(f"Warning: Device not specified, using {device}")

        assert num_epochs > 0

        train_years = train_dataset.years
        self._min_date = train_dataset.min_date
        self._max_date = train_dataset.max_date
        self.batch_size = batch_size
        self.to(device)

        if val_dataset is not None:
            train_dataset = TorchDataset(train_dataset)
            val_dataset = TorchDataset(val_dataset)
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=batch_size,
                collate_fn=train_dataset.collate_fn,
                shuffle=True,
                drop_last=True,
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=batch_size, collate_fn=val_dataset.collate_fn
            )

        # Load optimizer and scheduler
        optimizer = optim_fn(self.parameters(), **optim_kwargs)
        if scheduler_fn is not None:
            scheduler = scheduler_fn(optimizer, **scheduler_kwargs)

        # Store training set feature means and sds for normalization
        self._norm_params = train_dataset.get_normalization_params(
            normalization="standard"
        )
        all_train_losses = []
        all_val_losses = []
        saved_metrics = []

        # Training loop
        max_epoch = num_epochs
        for epoch in range(num_epochs):
            losses = []
            self.train()
            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
            for batch in pbar:
                # Set gradients to zero
                optimizer.zero_grad()

                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(device)

                # Forward pass
                inputs = {k: v for k, v in batch.items() if k != KEY_TARGET}

                # Normalize inputs
                inputs = self._normalize_inputs(inputs)
                predictions = self(inputs)
                if predictions.dim() > 1:
                    predictions = predictions.squeeze(-1)
                target = batch[KEY_TARGET]
                loss = loss_fn(predictions, target, **loss_kwargs)

                # Backward pass
                loss.backward()
                optimizer.step()
                if scheduler_fn is not None:
                    scheduler.step()
                losses.append(loss.item())

                mean_train_loss = np.mean(losses)

                pbar.set_description(
                    f"Epoch {epoch + 1}/{num_epochs} | Loss: {mean_train_loss:.4f}"
                )
                all_train_losses.append(mean_train_loss)

            if val_loader is not None and (
                (epoch % val_every_n_epochs == 0) or (epoch == num_epochs - 1)
            ):
                self.eval()

                # Validation loop
                with torch.no_grad():
                    y_pred, _ = self.predict(val_dataset, device=device)
                    y_true = val_dataset.targets
                    val_metric = evaluate_predictions(y_true, y_pred)
                    saved_metrics.append(val_metric)
                
                    if do_early_stopping and self._early_stopper.early_stop(val_metric):
                        max_epoch = epoch - self._early_stopper.patience


        return self, {
            "train_losses": all_train_losses,
            "val_losses": all_val_losses if val_loader is not None else None,
            "max_epoch": max_epoch,
        }

    def _normalize_inputs(self, inputs):
        """Normalize inputs using saved normalization parameters.

        Args:
          inputs: a dict of inputs

        Returns:
          The same dict after normalizing the entries
        """
        for pred in ALL_PREDICTORS:
            assert pred in inputs
            assert pred in self._norm_params
            inputs[pred] = (
                inputs[pred] - self._norm_params[pred]["mean"]
            ) / self._norm_params[pred]["std"]

        return inputs

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
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(device)
                inputs = {k: v for k, v in batch.items() if k != KEY_TARGET}
                inputs = self._normalize_inputs(inputs)
                y_pred = model(inputs)
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
        hidden_size,
        num_layers,
        output_size=1,
        transforms=[
            transform_ts_inputs_to_dekadal,
            transform_stack_ts_static_inputs,
        ],
        **kwargs,
    ):
        # Add all arguments to init_args to enable model reconstruction in fit method
        n_ts_inputs = len(TIME_SERIES_PREDICTORS)
        n_static_inputs = len(STATIC_PREDICTORS)
        kwargs["n_ts_inputs"] = n_ts_inputs
        kwargs["n_static_inputs"] = n_static_inputs
        kwargs["hidden_size"] = hidden_size
        kwargs["num_layers"] = num_layers
        kwargs["output_size"] = output_size

        super().__init__(**kwargs)
        self._lstm = nn.LSTM(n_ts_inputs, hidden_size, num_layers, batch_first=True)
        self._fc = nn.Linear(hidden_size + n_static_inputs, output_size)
        self._transforms = transforms
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, x):
        for transform in self._transforms:
            x = transform(x, self._min_date, self._max_date)

        x_ts = x["ts"]
        x_static = x["static"]
        x_ts, _ = self._lstm(x_ts.to(device=self._device))
        x = torch.cat([x_ts[:, -1, :], x_static], dim=1)
        output = self._fc(x)
        return output
