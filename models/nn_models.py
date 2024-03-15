import torch
import torch.nn as nn
from tqdm import tqdm

from models.model import BaseModel
from datasets.dataset import Dataset

class BaseNNModel(BaseModel, nn.Module):
    def __init__(   self, 
                    network: torch.nn.Module,
                    device: str = "cpu",
                    *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        
        self.network = network
        self.device = device

    def fit(self, 
            train_dataset: Dataset,
            val_dataset: Dataset = None,
            val_every_n_epochs: int = 1,

            num_epochs: int = 1,
            batch_size: int = 1,

            loss_fn: callable = None,
            loss_kwargs: dict = None,

            optim_fn: callable = None,
            optim_kwargs: dict = None,

            scheduler_fn: callable = None,
            scheduler_kwargs: dict = None,

             **fit_params):
        
        # Set default values for loss_fn, optim_fn, optim_kwargs, scheduler_fn, scheduler_kwargs
        if loss_fn is None: loss_fn = nn.MSELoss
        if optim_fn is None: optim_fn = torch.optim.Adam
        if optim_kwargs is None: optim_kwargs = {}

        assert num_epochs > 0
    
        # Send model to device
        self.network.to(self.device)
        
        # Set optimizer and scheduler
        optimizer = optim_fn(self.network.parameters(), **optim_kwargs)
        if scheduler is not None: scheduler = scheduler_fn(optimizer, **scheduler_kwargs)

        # Get dataloaders
        train_batch_size = min(batch_size, len(train_dataset))
        train_loader = torch.utils.data.DataLoader(train_dataset, 
                                                   batch_size=train_batch_size, 
                                                   shuffle=True,
                                                   collate_fn=train_dataset.collate_fn)
        
        if val_loader is not None:
            val_batch_size = min(batch_size, len(val_dataset))
            val_loader = torch.utils.data.DataLoader(val_dataset, 
                                                   batch_size=val_batch_size,
                                                   collate_fn=val_dataset.collate_fn)

        # Load optimizer and scheduler
        optimizer = optim_fn(self.network.parameters(), **optim_kwargs)
        scheduler = scheduler_fn(optimizer, **scheduler_kwargs)

        # Training loop
        for epoch in range(num_epochs):
            losses = []
            self.network.train()
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for batch in pbar:
                # Set gradients to zero
                optimizer.zero_grad()

                # TODO Should we normalize batch here?

                # Send data to device
                for key in batch: batch[key] = batch[key].to(self.device)

                # Forward pass
                predictions = self.network(batch)

                target = batch[train_dataset.KEY_TARGET]
                loss = loss_fn(predictions, target, **loss_kwargs)

                # Backward pass
                loss.backward()
                optimizer.step()

                # Save loss
                losses.append(loss.item())

                pbar.set_description(
                    f"Epoch {epoch+1}/{num_epochs} | Loss: {loss.item():.4f}"
                )


            if val_dataset is not None and (epoch+1) % val_every_n_epochs == 0:
                self.network.eval()

                # Validation loop
                with torch.no_grad():
                    val_losses = []
                    tqdm_val = tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{num_epochs}")
                    for batch in tqdm_val:
                        # Send data to device
                        for key in batch: batch[key] = batch[key].to(self.device)

                        # Forward pass
                        predictions = self.network(batch)

                        target = batch[val_dataset.KEY_TARGET]
                        loss = loss_fn(predictions, target, **loss_kwargs)

                        # Save loss
                        val_losses.append(loss.item())

                        tqdm_val.set_description(
                            f"Validation Epoch {epoch+1}/{num_epochs} | Loss: {loss.item():.4f}"
                        )
                    
            scheduler.step()
        return self, {}

    def predict_batch(self, X: list):
        # TODO this should be implemented per batch

        # Convert batch to tensor
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        self.network.to(self.device)

        self.network.eval()
        with torch.no_grad():
            return self.network(X).detach(), {}

    def save(self, model_name):
        torch.save(self, model_name)

    @classmethod
    def load(cls, model_name):
        return torch.load(model_name)
    
class ExampleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._bn1 = nn.BatchNorm1d(input_size)
        self._lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self._bn2 = nn.BatchNorm1d(output_size)
        self._fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        
        # TODO load dict items correctly into tensor

        x = self._bn1(x)
        x, _ = self._lstm(x)
        x = self._bn2(x)
        x = self._fc(x)
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


        


