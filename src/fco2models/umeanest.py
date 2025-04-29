import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

def train_mean_estimator(model, num_epochs, old_epoch, train_dataloader, val_dataloader, optimizer, lr_scheduler, save_model_path=None, rmse_const=None, is_sequence=True):
    """training loop for diffusion model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    model.to(device)
    
    loss_fn = nn.MSELoss()
    train_losses = []
    val_losses = []
    for epoch in range(old_epoch, num_epochs):
        model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in progress_bar:
            batch = batch[0].to(device)
            target = batch[:, 0:1, :].float()
            context = batch[:, 1:, :]

            not_nan_mask = torch.ones_like(target).bool()
            if is_sequence:
                not_nan_mask = context[:, -1:, :].bool()
                # replace nan with zeros
                target = torch.where(~not_nan_mask, torch.zeros_like(target), target).float()
        
            #concatenate the noisy target with the context and the mask
            model_input = context
            model_input = model_input.to(device).float()
            
            # Get the model prediction
            mean_pred = model(model_input, torch.zeros(batch.shape[0], ).to(device).float(), return_dict=False)[0]

            # Calculate the loss
            optimizer.zero_grad()
            loss = loss_fn(mean_pred[not_nan_mask], target[not_nan_mask])	
            loss.backward()
            epoch_loss += loss.item()
            
            # Update the model parameters with the optimizer
            optimizer.step()

            #update loss in progress bar
            progress_bar.set_postfix({"Loss": loss.item()})
            # batch.detach()
        train_losses.append(epoch_loss / len(train_dataloader))
        print(f"Epoch {epoch+1} Loss: {epoch_loss / len(train_dataloader):.6f}")
        # Update the learning rate
        lr_scheduler.step()
        #check_gradients(model)

        # print validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_dataloader:
                batch = batch[0].to(device)
                target = batch[:, 0:1, :].float()
                context = batch[:, 1:, :]

                not_nan_mask = torch.ones_like(target).bool()
                if is_sequence:
                    not_nan_mask = context[:, -1:, :].bool()
                    # replace nan with zeros
                    target = torch.where(~not_nan_mask, torch.zeros_like(target), target).float()

                model_input = context
                model_input = model_input.to(device).float()

                mean_pred = model(model_input, torch.zeros(batch.shape[0], ).to(device), return_dict=False)[0]

                # Calculate the loss
                val_loss += loss_fn(mean_pred[not_nan_mask], target[not_nan_mask]).item()
        val_losses.append(val_loss / len(val_dataloader))
        print(f"Validation Loss: {val_loss / len(val_dataloader):.6f}")
        if rmse_const is not None:
            print(f"Validation RMSE: {np.sqrt(val_loss / len(val_dataloader)) * rmse_const:.6f}")

        if save_model_path and (epoch+1) % 10 == 0:
            torch.save(model.state_dict(), save_model_path+f"e_{epoch+1}.pt")

    # save model checkpoint
    if save_model_path:
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': lr_scheduler.state_dict() if lr_scheduler else None,
            'rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            'train_losses': train_losses,
            'val_losses': val_losses,
        }, save_model_path+f"final_model_e_{num_epochs}.pt")
    return model, train_losses, val_losses

def predict_mean_eval(model, dataloader, is_sequence=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Predicting on {device}")
    loss_fn=nn.MSELoss(reduction='none')
    losses = []
    predictions = []

    model.to(device)
    model.eval()
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Predicting")
        for batch in progress_bar:
            batch = batch[0].to(device)
            target = batch[:, 0:1, :].float()
            context = batch[:, 1:, :]

            input = context
            input = input.to(device).float()
            mean_pred = model(input, torch.zeros(batch.shape[0], ).to(device), return_dict=False)[0]
            # Calculate the loss
            losses.append(loss_fn(mean_pred, target).cpu().numpy())
            # Append the predictions to the list
            predictions.append(mean_pred.cpu().numpy())
            progress_bar.set_postfix({"Loss": loss_fn(mean_pred, target).mean().item()})

    return np.concatenate(losses, axis=0), np.concatenate(predictions, axis=0)

import pandas as pd
def sota_split():
    n_splits = 7
    test_folds = []
    val_folds = []
    end = pd.to_datetime('2021-12-31')
    for i in range(1, n_splits+1):
        test_start = pd.to_datetime(f'1982-{i:02d}-01')
        test_months = pd.date_range(test_start, end, freq='7MS')
        test_folds.append(test_months)
        
        val_start = pd.to_datetime(f'1982-{i:02d}-01') + pd.DateOffset(months=3)
        val_months = pd.date_range(val_start, end, freq='7MS')
        val_folds.append(val_months)
    
    return test_folds, val_folds

from torch.utils import data
from torch.utils.data import TensorDataset

import torch.optim as optim
from fco2models.utraining import normalize_dss
def cross_val(ds, coords, split_method='sota', batch_size=2048, **kwargs):
    """
    Cross-validation function to split the data into training, validation, and test sets.
       It creates the folds and then calls train_mean_estimator in a loop to train the model on each fold.
    Args:
        ds (numpy array): The dataset to be split.
        coords (dataframe): df with date, lat, lon coordinates corresponding to the dataset.
        split_method (str): The method to use for splitting the data. Default is 'sota'.
        **kwargs: all additional arguments for train_mean_estimator.
    """
    if split_method == 'sota':
        test_folds, val_folds = sota_split()
    else:
        raise ValueError(f"Unknown split method: {split_method}")
    
    test_predictions = []
    test_losses = []
    for (test_months, val_months) in zip(test_folds, val_folds):
        # Create test and validation sets based on the months
        test_months = test_months.strftime("%Y-%m").tolist()
        val_months = val_months.strftime("%Y-%m").tolist()

        dates = coords['time_1d'].dt.strftime("%Y-%m").tolist()
        test_ds = ds[np.isin(dates, test_months)]
        val_ds = ds[np.isin(dates, val_months)]
        train_ds = ds[~np.isin(dates, test_months) & ~np.isin(dates, val_months)]

        print(f"test_ds shape: {test_ds.shape}")
        print(f"val_ds shape: {val_ds.shape}")
        print(f"train_ds shape: {train_ds.shape}")

        # Shuffle the data
        np.random.shuffle(train_ds)

        # Normalize the data
        mode = kwargs.get('mode', 'mean_std')
        train_ds, rest_ds, train_means, train_stds, train_mins, train_maxs = normalize_dss([train_ds, val_ds], mode=mode)
        val_ds = rest_ds[0]
        rsme_const = train_stds[0] if mode == 'mean_std' else (train_maxs[0] - train_mins[0]) / 2

        # Create DataLoader objects
        train_dataset = TensorDataset(torch.tensor(train_ds))
        val_dataset = TensorDataset(torch.tensor(val_ds))
        train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Train the model
        model, _, _ = train_mean_estimator(**kwargs, train_dataloader=train_dataloader, val_dataloader=val_dataloader, rmse_const=rsme_const)

        # Evaluate the model on the test set
        test_dataset = TensorDataset(torch.tensor(test_ds))
        test_dataloader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        test_loss, predictions = predict_mean_eval(model, test_dataloader, device='cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Test Loss: {np.sqrt(test_loss.mean()) * rsme_const:.6f}")
        if kwargs.get('rmse_const') is not None:
            print(f"Test RMSE: {np.sqrt(test_loss) * kwargs['rmse_const']:.6f}")
        
        test_losses.append(test_loss)
        test_predictions.append(predictions)

    test_predictions = np.concatenate(test_predictions, axis=0)
    coords['predictions'] = test_predictions

    return coords


# define simple pointwise mlp model
class  MLPModel(nn.Module):
    """simple baseline pointwise mlp model"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.batchnorm2 = nn.BatchNorm1d(hidden_dim)
        self.batchnorm3 = nn.BatchNorm1d(hidden_dim // 2)


    
    def forward(self, x, t, return_dict=False):
        x = self.fc1(x.squeeze(-1))
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.batchnorm3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return (x.unsqueeze(-1), None)

from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
def get_loader_point_ds(df, stats, mode, batch_size=2048, shuffle=False):
    """ small helper function to prepare the dataset for pointwise mlp model"""
    ds = df.dropna()
    ds = ds.values[:, :, np.newaxis]
    ds = normalize_dss([ds], stats, mode)[0]
    ds = torch.tensor(ds).float()
    dataloader = DataLoader(TensorDataset(ds), batch_size=batch_size, shuffle=False)
    return dataloader
    


    

