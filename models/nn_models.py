import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

from datasets.dataset_torch import TorchDataset
from datasets.dataset import Dataset
from models.model import BaseModel

from config import KEY_LOC, KEY_YEAR, KEY_TARGET


class BaseNNModel(BaseModel, nn.Module):
    def __init__(   self, 
                    *args, **kwargs):
        super(BaseModel, self).__init__()
        super(nn.Module, self).__init__()


    def fit(self, 
            train_dataset: Dataset,
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

            device: str = "cpu",

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
        
        # Set default values for loss_fn, optim_fn, optim_kwargs, scheduler_fn, scheduler_kwargs
        if loss_fn is None: loss_fn = torch.nn.functional.mse_loss
        if loss_kwargs is None: loss_kwargs = {"reduction": "mean"}
        if optim_fn is None: optim_fn = torch.optim.Adam
        if optim_kwargs is None: optim_kwargs = {}

        assert num_epochs > 0
    
        # Send model to device
        self.to(device) #TODO this doesn't make sense
        
        
        # Set optimizer and scheduler
        optimizer = optim_fn(self.parameters(), **optim_kwargs)
        if scheduler_fn is not None: scheduler = scheduler_fn(optimizer, **scheduler_kwargs)

        # Convert dataset to torch Dataset
        train_dataset = TorchDataset(train_dataset)

        # Get train and validation ids
        n = len(train_dataset)
        n_val = int(n * val_fraction)
        n_train = n - n_val
        train_ids = list(range(n_train))
        val_ids = list(range(n_train, n))


        # Get dataloaders
        train_batch_size = min(batch_size, len(train_ids))
        train_loader = torch.utils.data.DataLoader(train_dataset, 
                                                   batch_size=train_batch_size, 
                                                   sampler=torch.utils.data.SubsetRandomSampler(train_ids),
                                                   collate_fn=train_dataset.collate_fn)
        
        if val_fraction > 0:
            val_batch_size = min(batch_size, len(val_ids))
            val_loader = torch.utils.data.DataLoader(train_dataset, 
                                                    batch_size=val_batch_size,
                                                    sampler=torch.utils.data.SubsetRandomSampler(val_ids),
                                                    collate_fn=train_dataset.collate_fn)

        # Load optimizer and scheduler
        optimizer = optim_fn(self.parameters(), **optim_kwargs)
        if scheduler_fn is not None: scheduler = scheduler_fn(optimizer, **scheduler_kwargs)

        # Training loop
        for epoch in range(num_epochs):
            losses = []
            self.train()
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for batch in pbar:
                # Set gradients to zero
                optimizer.zero_grad()

                # Send data to device
                for key in batch: 
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(device)

                # Forward pass
                features = {k: v for k, v in batch.items() if k != KEY_TARGET}
                predictions = self(features).squeeze()
                target = batch[KEY_TARGET]
                loss = loss_fn(predictions, target, **loss_kwargs)

                # Backward pass
                loss.backward()
                optimizer.step()

                # Save loss
                losses.append(loss.item())

                pbar.set_description(
                    f"Epoch {epoch+1}/{num_epochs} | Loss: {loss.item():.4f}"
                )


            if val_loader is not None and (epoch+1) % val_every_n_epochs == 0:
                self.eval()

                # Validation loop
                with torch.no_grad():
                    val_losses = []
                    tqdm_val = tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{num_epochs}")
                    for batch in tqdm_val:
                        # Send data to device
                        for key in batch: 
                            if isinstance(batch[key], torch.Tensor):
                                batch[key] = batch[key].to(device)

                        # Forward pass
                        features = {k: v for k, v in batch.items() if k != KEY_TARGET}
                        predictions = self(features).squeeze()
                        target = batch[KEY_TARGET]
                        loss = loss_fn(predictions, target, **loss_kwargs)

                        # Save loss
                        val_losses.append(loss.item())

                        tqdm_val.set_description(
                            f"Validation Epoch {epoch+1}/{num_epochs} | Loss: {loss.item():.4f}"
                        )
                    
            if scheduler_fn is not None: scheduler.step()
        return self, {}

    def predict_batch(self, X: list, device: str = "cpu", as_single_batch: bool = False):
        """Run fitted model on batched data items.

        Args:
          X: a list of data items, each of which is a dict
          device: a string specifying the device to use. Default is "cpu".
          as_single_batch: a bool specifying whether to run all items in a single batch. Default is False.

        Returns:
          A tuple containing a np.ndarray and a dict with additional information.
        """

        if as_single_batch:
            X = TorchDataset.collate_fn([TorchDataset._cast_to_tensor(sample) for sample in X])
            X = {key: X[key].to(device) for key in X.keys() if isinstance(X[key], torch.Tensor)}

            self.to(device)
            self.eval()
            with torch.no_grad():
                features = {k: v for k, v in X.items() if k != KEY_TARGET}
                predictions = self(features).squeeze().cpu().numpy()
                return predictions, {}
        else:
            predictions = np.zeros((len(X), 1))
            self.to(device)
            self.eval()
            with torch.no_grad():
                for i, item in enumerate(X):
                    item = TorchDataset.collate_fn(TorchDataset._cast_to_tensor(item))
                    item = {key: item[key].to(device) for key in item.keys() if isinstance(item[key], torch.Tensor)}
                
                    features = {k: v for k, v in item.items() if k != KEY_TARGET}
                    predictions[i] = self(features).squeeze().cpu().numpy()
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
    def __init__(self, n_ts_features, n_static_features, hidden_size, num_layers, output_size=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lstm = nn.LSTM(n_ts_features, hidden_size, num_layers, batch_first=True)
        self._fc = nn.Linear(hidden_size + n_static_features, output_size)

    def forward(self, x):
        # Could be moved to training loop. Assumes that all individual features are one channel each
        max_n_dim = max([len(v.shape) for k, v in x.items() if isinstance(v, torch.Tensor)])
        x_ts = torch.cat([v.unsqueeze(2) for k, v in x.items() if isinstance(v, torch.Tensor) and len(v.shape) == max_n_dim], dim=2)
        x_static = torch.cat([v.unsqueeze(1) for k, v in x.items() if isinstance(v, torch.Tensor) and len(v.shape) < max_n_dim], dim=1)

        x_ts, _ = self._lstm(x_ts)
        x = torch.cat([x_ts[:, -1, :], x_static], dim=1)
        x = self._fc(x)
        return x


        


