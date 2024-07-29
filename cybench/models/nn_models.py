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


class BaseNNModel(BaseModel, nn.Module):
    def __init__(self, **kwargs):
        super(BaseModel, self).__init__()
        super(nn.Module, self).__init__()

        self._norm_params = None
        self._init_args = kwargs
        self._logger = logging.getLogger(__name__)

    def fit(
        self,
        dataset: Dataset,
        optimize_hyperparameters: bool = False,
        param_space: dict = None,
        optim_kwargs: dict = {},
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        seed: int = 42,
        **fit_params,
    ):
        """Fit or train the model.

        Args:
          dataset: Dataset. Training dataset.
          optimize_hyperparameters: bool. Flag to tune hyperparameters.
          param_space: dict. Each entry is a hyperparameter name and list or range of values.
          kfolds: int. k in k-fold cv.
          epochs: int. Number of epochs to train.
          seed: seed for random number generator.
          **fit_params: Additional parameters.

        Returns:
          A tuple containing the fitted model and a dict with additional information.
        """
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        # Default optimizer args
        if not optim_kwargs:
            optim_kwargs = {
                "lr": 0.0001,
                "weight_decay": 0.00001,
            }

        opt_param_setting = {}
        if optimize_hyperparameters and (len(dataset.years) > 1):
            opt_param_setting = self._optimize_hyperparameters(
                dataset,
                param_space,
                optim_kwargs=optim_kwargs,
                device=device,
                **fit_params,
            )
            # replace optimizer args with optimal values
            if "lr" in opt_param_setting:
                optim_kwargs["lr"] = opt_param_setting["lr"]
            if "weight_decay" in opt_param_setting:
                optim_kwargs["weight_decay"] = opt_param_setting["weight_decay"]

        train_losses = self._train_final_model(
            dataset,
            optim_kwargs=optim_kwargs,
            device=device,
            **opt_param_setting,
            **fit_params,
        )

        return self, {"train_losses": train_losses}

    def _optimize_hyperparameters(
        self,
        dataset: Dataset,
        param_space: dict,
        optim_kwargs: dict,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        kfolds: int = 1,
        epochs: int = 10,
        **fit_params,
    ) -> dict:
        """Optimize hyperparameters

        Args:
          dataset: Dataset. Training dataset.
          param_space: a dict of parameters to optimize
          optim_kwargs: a dict of arguments to the optimizer
          device: str, the device to use
          kfolds: k for k-fold cv.
          epochs: Number of epochs to train.
          **fit_params: Additional parameters.

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
            if "lr" in setting:
                optim_kwargs["lr"] = setting["lr"]
            if "weight_decay" in setting:
                optim_kwargs["weight_decay"] = setting["weight_decay"]

            val_loss = None
            if kfolds == 1:
                self._logger.debug(f"Running setting {i + 1}/{len(settings)}")
                val_years = all_years[len(all_years) // 2 :]
                train_years = [y for y in all_years if y not in val_years]
                new_model = self.__class__(**self._init_args)
                train_losses, val_losses = new_model._train_and_validate(
                    dataset,
                    train_years,
                    val_years,
                    optim_kwargs=optim_kwargs,
                    epochs=epochs,
                    device=device,
                    **setting,
                    **fit_params,
                )

                val_loss = val_losses[-1]
            else:
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
                    new_model = self.__class__(**self._init_args)
                    train_losses, val_losses = new_model._train_and_validate(
                        dataset,
                        train_years,
                        val_years,
                        optim_kwargs=optim_kwargs,
                        device=device,
                        epochs=epochs,
                        **setting,
                        **fit_params,
                    )

                    cv_losses.append(val_losses[-1])

                val_loss = np.mean(cv_losses)

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

    def _train_and_validate(
        self,
        dataset: Dataset,
        train_years: list,
        val_years: list,
        validation_interval: int = 5,
        epochs: int = 10,
        batch_size: int = 16,
        optimizer_fn: callable = torch.optim.Adam,
        optim_kwargs: dict = {},
        loss_fn: callable = torch.nn.functional.mse_loss,
        loss_kwargs: dict = {},
        scheduler_fn: callable = None,
        sched_kwargs: dict = {},
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs,
    ):
        """
        Fit or train the model and evaluate on validation data.

        Args:
            dataset: Dataset,
            train_years: training years
            val_years: validation years
            validation_interval: int, validation frequency, default is 5
            do_early_stopping: bool, whether to use early stopping, default is False
            epochs: int, the number of epochs to train the model, default is 10
            batch_size: int, the batch size, default is 16
            optim_fn: callable, the optimizer function, default is torch.optim.Adam
            optim_kwargs: dict, arguments to the optimizer function
            loss_fn: callable, the loss function, default is torch.nn.functional.mse_loss
            loss_kwargs: dict, arguments to the loss function
            scheduler_fn: callable, the scheduler function, default is None
            sched_kwargs: dict, arguments to the scheduler function
            device: str, the device to use
            **kwargs: Additional parameters.

        Returns:
          A tuple training losses, validation losses and maximum epochs to train.
        """
        self.to(device)
        assert epochs > 0

        train_dataset, val_dataset = dataset.split_on_years((train_years, val_years))
        self._min_date = train_dataset.min_date
        self._max_date = train_dataset.max_date

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
            val_dataset, batch_size=batch_size, collate_fn=TorchDataset.collate_fn
        )

        # Initialize optimizer and scheduler
        optimizer = optimizer_fn(self.parameters(), **optim_kwargs)
        if scheduler_fn is not None:
            assert sched_kwargs
            scheduler = scheduler_fn(optimizer, **sched_kwargs)
        else:
            scheduler = None

        # Store training set feature means and sds for normalization
        self._norm_params = train_dataset.get_normalization_params(
            normalization="standard"
        )

        train_losses = []
        val_losses = []

        # Training loop
        for epoch in range(epochs):
            self.train()
            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
            train_loss = self._train_epoch(
                pbar,
                device,
                optimizer,
                loss_fn=loss_fn,
                loss_kwargs=loss_kwargs,
                scheduler=scheduler,
            )
            pbar.set_description(f"Epoch {epoch + 1}/{epochs} | Loss: {train_loss:.4f}")
            train_losses.append(train_loss)

            if val_loader is not None and (
                (epoch % validation_interval == 0) or (epoch == epochs - 1)
            ):
                with torch.no_grad():
                    self.eval()
                    tqdm_val = tqdm(
                        val_loader, desc=f"Validation Epoch {epoch + 1}/{epochs}"
                    )

                    losses = []
                    for batch in tqdm_val:
                        targets = batch[KEY_TARGET]
                        batch_preds = self._forward_pass(batch, device)
                        loss = loss_fn(batch_preds, targets, **loss_kwargs)
                        losses.append(loss.item())

                    val_loss = np.mean(losses)
                    val_losses.append(val_loss)

        return train_losses, val_losses

    def _train_final_model(
        self,
        dataset: Dataset,
        epochs: int,
        optimizer_fn: callable = torch.optim.Adam,
        optim_kwargs: dict = {},
        loss_fn: callable = torch.nn.functional.mse_loss,
        loss_kwargs: dict = {"reduction": "mean"},
        scheduler_fn: callable = None,
        sched_kwargs: dict = {},
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 16,
        **kwargs,
    ):
        """
        Fit or train the model on the entire training set.

        Args:
            dataset: Dataset,
            epochs: int, number of epochs to train
            optimizer_fn: callable, the optimizer function, default is Adam
            optim_kwargs: dict, arguments to the optimizer function
            loss_fn: callable, the loss function, default is mse_loss
            loss_kwargs: dict, arguments to the loss function
            scheduler_fn: callable, the scheduler function, default is None
            sched_kwargs: dict, arguments to the scheduler function
            device: str, the device to use, default is "cpu"
            batch_size: int, default is 16
            **kwargs: Additional parameters.

        Returns:
          A list of training losses (one value per epoch).
        """
        self._min_date = dataset.min_date
        self._max_date = dataset.max_date

        train_dataset = TorchDataset(dataset)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            collate_fn=train_dataset.collate_fn,
            shuffle=True,
            drop_last=True,
        )

        # Initialize optimizer and scheduler
        optimizer = optimizer_fn(self.parameters(), **optim_kwargs)
        if scheduler_fn is not None:
            assert sched_kwargs
            scheduler = scheduler_fn(optimizer, **sched_kwargs)
        else:
            scheduler = None

        # Store training set feature means and sds for normalization
        self._norm_params = train_dataset.get_normalization_params(
            normalization="standard"
        )

        train_losses = []
        for epoch in range(epochs):
            self.train()
            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
            train_loss = self._train_epoch(
                pbar,
                device,
                optimizer,
                loss_fn=loss_fn,
                loss_kwargs=loss_kwargs,
                scheduler=scheduler,
            )
            pbar.set_description(f"Epoch {epoch + 1}/{epochs} | Loss: {train_loss:.4f}")
            train_losses.append(train_loss)

        return train_losses

    def _train_epoch(
        self,
        tqdm_loader: tqdm,
        device: str,
        optimizer: torch.optim.Optimizer,
        loss_fn: callable = torch.nn.functional.mse_loss,
        loss_kwargs: dict = {"reduction": "mean"},
        scheduler: torch.optim.lr_scheduler.LRScheduler = None,
    ):
        """Run one epoch during trainig

        Args:
          tqdm_loader: data loader with progress bar
          device: str, the device to use
          optimizer: the optimizer
          loss_fn: the loss function, default mse_loss
          loss_kwargs: dict, the arguments to loss_fn
          scheduler: scheduler for learning rate of optimizer

        Returns:
          The average of all batch losses
        """
        losses = []
        for batch in tqdm_loader:
            # Set gradients to zero
            optimizer.zero_grad()

            batch_preds = self._forward_pass(batch, device)
            targets = batch[KEY_TARGET]
            loss = loss_fn(batch_preds, targets, **loss_kwargs)

            # Backward pass
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            losses.append(loss.item())

        return np.mean(losses)

    def _forward_pass(self, batch: dict, device: str):
        """A forward pass for batched data.

        Args:
          batch: a dict of batched data
          device: str, the device to use

        Returns:
          An np.ndarray
        """
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)

        # Normalize inputs
        inputs = {k: v for k, v in batch.items() if k != KEY_TARGET}
        inputs = self._normalize_inputs(inputs)
        batch_preds = self(inputs)
        if batch_preds.dim() > 1:
            batch_preds = batch_preds.squeeze(-1)

        return batch_preds

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

    def predict_items(self, X: list, device: str = "cpu", **predict_params):
        """Run fitted model on a list of data items.

        Args:
          X: a list of data items, each of which is a dict
          device: str, the device to use
          **predict_params: Additional parameters

        Returns:
          A tuple containing a np.ndarray and a dict with additional information.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(device)
        self.eval()

        with torch.no_grad():
            X_collated = TorchDataset.collate_fn(
                [TorchDataset._cast_to_tensor(x) for x in X]
            )
            y_pred = self._forward_pass(X_collated, device)
            y_pred = y_pred.cpu().numpy()
            return y_pred, {}

    def predict(
        self,
        dataset: Dataset,
        device: str = "cpu",
        batch_size: int = 16,
        **predict_params,
    ):
        """Run fitted model on batched data items.

        Args:
          dataset: validation dataset
          device: str, the device to use
          **predict_params: Additional parameters

        Returns:
          A tuple containing a np.ndarray and a dict with additional information.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(device)
        self.eval()
        test_dataset = TorchDataset(dataset)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, collate_fn=TorchDataset.collate_fn
        )

        with torch.no_grad():
            predictions = None
            for batch in test_loader:
                batch_preds = self._forward_pass(batch, device)
                if predictions is None:
                    predictions = batch_preds
                else:
                    predictions = np.concatenate((predictions, batch_preds), axis=0)

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
        hidden_size=64,
        num_layers=1,
        output_size=1,
        **kwargs,
    ):
        # Add all arguments to init_args to enable model reconstruction in fit method
        n_ts_inputs = len(TIME_SERIES_PREDICTORS)
        n_static_inputs = len(STATIC_PREDICTORS)
        kwargs["hidden_size"] = hidden_size
        kwargs["num_layers"] = num_layers
        kwargs["output_size"] = output_size

        super().__init__(**kwargs)
        self._lstm = nn.LSTM(n_ts_inputs, hidden_size, num_layers, batch_first=True)
        self._fc = nn.Linear(hidden_size + n_static_inputs, output_size)
        self._transforms = [
            transform_ts_inputs_to_dekadal,
            transform_stack_ts_static_inputs,
        ]

    def fit(
        self,
        dataset: Dataset,
        optimize_hyperparameters: bool = False,
        param_space: dict = {},
        kfolds: int = 1,
        epochs: int = 10,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        seed: int = 42,
        **fit_params,
    ):
        """Fit or train the model.

        Args:
          dataset: Dataset. Training dataset.
          optimize_hyperparameters: bool. Flag to tune hyperparameters.
          param_space: dict. Each entry is a hyperparameter name and list or range of values.
          kfolds: int. k in k-fold cv.
          epochs: int. Number of epochs to train.
          seed: seed for random number generator.
          **fit_params: Additional parameters.

        Returns:
          A tuple containing the fitted model and a dict with additional information.
        """
        if ("scheduler_fn" in fit_params) and (fit_params["scheduler_fn"] is not None):
            fit_params["sched_kwargs"] = {"step_size": 2, "gamma": 0.5}

        if not param_space:
            param_space = {
                "lr": [0.0001, 0.00001],
                "weight_decay": [0.0001, 0.00001],
            }

        super().fit(
            dataset,
            optimize_hyperparameters=optimize_hyperparameters,
            param_space=param_space,
            kfolds=kfolds,
            epochs=epochs,
            device=device,
            seed=seed,
            **fit_params,
        )

    def forward(self, x):
        for transform in self._transforms:
            x = transform(x, self._min_date, self._max_date)

        x_ts = x["ts"]
        x_static = x["static"]
        x_ts, _ = self._lstm(x_ts)
        x = torch.cat([x_ts[:, -1, :], x_static], dim=1)
        output = self._fc(x)
        return output
