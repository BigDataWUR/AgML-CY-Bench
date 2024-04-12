import torch
import torch.nn as nn
from tqdm import tqdm

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
            batch_size: int = 1,

            loss_fn: callable = None,
            loss_kwargs: dict = None,

            optim_fn: callable = None,
            optim_kwargs: dict = None,

            scheduler_fn: callable = None,
            scheduler_kwargs: dict = None,

            device: str = "cpu",

             **fit_params):
        
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
                predictions = self(batch).squeeze()
                target = batch[KEY_TARGET]

                # Unsqueeze number of dimensions of target to match predictions
                if len(predictions.shape) != len(target.shape):
                    target = target.unsqueeze(-1)
                target = target.float()

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
                        predictions = self(batch).squeeze()

                        target = batch[KEY_TARGET]

                        # Unsqueeze number of dimensions of target to match predictions
                        if len(predictions.shape) != len(target.shape):
                            target = target.unsqueeze(-1)
                        target = target.float() #TODO move to dataloader

                        loss = loss_fn(predictions, target, **loss_kwargs)

                        # Save loss
                        val_losses.append(loss.item())

                        tqdm_val.set_description(
                            f"Validation Epoch {epoch+1}/{num_epochs} | Loss: {loss.item():.4f}"
                        )
                    
            if scheduler_fn is not None: scheduler.step()
        return self, {}

    def predict_batch(self, X: list, device: str = "cpu"):
        # TODO this should be implemented per batch
        # TODO fix self.device

        # Convert batch to tensor
        X = torch.tensor(X, dtype=torch.float32, device=device)
        self.to(device)

        self.eval()
        with torch.no_grad():
            return self(X).detach(), {}

    def save(self, model_name):
        torch.save(self, model_name)

    @classmethod
    def load(cls, model_name):
        return torch.load(model_name)
    
class ExampleLSTM(BaseNNModel):
    def __init__(self, input_size, hidden_size, num_layers, output_size=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._bn1 = nn.BatchNorm1d(input_size)
        self._lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self._bn2 = nn.BatchNorm1d(hidden_size)
        self._fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # TODO: Keep this out of the model:
        # 1. Get only the time series data from the batch
        # 2. Ensure that there is always a batch dimension
        # 3. Ensure that input data is float32
        # 4. ?Concatenate along the channel dimension
        # 5. Rearrange dimensions to (b, l, c)
        # 6. Copy tensors in a better way, that does not give a warning


        # Find the dimension of the input with the most dimensions, corresponding to time series data
        max_n_dim = max([len(v.shape) for k, v in x.items() if isinstance(v, torch.Tensor)])

        # Keep only the time series data from the batch
        x = {k: v for k, v in x.items() if isinstance(v, torch.Tensor) and len(v.shape) == max_n_dim}

        # Add batch dimension if not present
        # If max_n_dim == 1, then the shape is (sequence_length)
        # In this case we add a batch dimension and a channel dimension to get (b, c, l)
        if max_n_dim == 1:
            x = {k: v.unsqueeze(0).unsqueeze(1) for k, v in x.items()}
        # If max_n_dim == 2, then the shape is (batch_size, n_features)
        # In this case we add a channel dimension to get (b, c, l)
        if max_n_dim == 2:
            x = {k: v.unsqueeze(1) for k, v in x.items()}

        # Concatenate along the channel dimension
        x = torch.cat([v for k, v in x.items()], dim=1)

        # Change dtype to float32 if needed
        x = x.float() #TODO move to dataloader

        x = self._bn1(x)
        x = x.permute(0, 2, 1)# Rearrange dimensions to (b, l, c)
        x, _ = self._lstm(x)
        x = x.permute(0, 2, 1)# Rearrange dimensions to (b, c, l)
        x = self._bn2(x)
        x = x.permute(0, 2, 1)# Rearrange dimensions to (b, l, c) #TODO check this
        x = self._fc(x)
        # Dimensions of x: (batch_size, sequence_length, output_size)
        # Take the last element of the sequence as the output
        x = x[:, -1, :]
        
        return x
    
if __name__ == "__main__":
    import os
    from config import PATH_DATA_DIR

    # TODO: Implement a test for the model
    
    data_path = os.path.join(PATH_DATA_DIR, "data_US", "county_features")
    # Training dataset
    train_csv = os.path.join(data_path, "grain_maize_US_train.csv")
    train_df = pd.read_csv(train_csv, index_col=["COUNTY_ID", "FYEAR"])
    train_yields = train_df[["YIELD"]].copy()
    feature_cols = [c for c in train_df.columns if c != "YIELD"]
    train_features = train_df[feature_cols].copy()
    train_dataset = Dataset(train_yields, [train_features])

    # Test dataset
    test_csv = os.path.join(data_path, "grain_maize_US_train.csv")
    test_df = pd.read_csv(test_csv, index_col=["COUNTY_ID", "FYEAR"])
    test_yields = test_df[["YIELD"]].copy()
    test_features = test_df[feature_cols].copy()
    test_dataset = Dataset(test_yields, [test_features])


        


