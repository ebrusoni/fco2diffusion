import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

def compute_gradient_norm(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

def train_pointwise_mlp(model, num_epochs, old_epoch, train_dataloader, val_dataloader, optimizer, lr_scheduler, save_model_path=None, rmse_const=None):
    """training loop for pointwise mlp model"""
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
        
        grad_norms = []
        for batch in progress_bar:
            optimizer.zero_grad()

            batch = batch[0].to(device)
            target = batch[:, 0:1].float()
            context = batch[:, 1:]

            #concatenate the noisy target with the context and the mask
            model_input = context
            model_input = model_input.to(device).float()
            
            # Get the model prediction
            mean_pred = model(model_input.to(device).float(), 
                              torch.zeros(batch.shape[0], ).to(device).float(), 
                              return_dict=False)[0]

            # Calculate the loss
            
            loss = loss_fn(mean_pred, target)	
            loss.backward()
            grad_norms.append(compute_gradient_norm(model))
            epoch_loss += loss.item()
            
            # Update the model parameters with the optimizer
            optimizer.step()

            #update loss in progress bar
            progress_bar.set_postfix({"Loss": loss.item()})
            # batch.detach()
        train_losses.append(epoch_loss / len(train_dataloader))
        grad_norms = torch.tensor(grad_norms)
        print(f"Epoch {epoch+1}: gradient norm stats — mean: {grad_norms.mean():.4f}, std: {grad_norms.std():.4f}, max: {grad_norms.max():.4f}, min: {grad_norms.min():.4f}")
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
                target = batch[:, 0:1].float()
                context = batch[:, 1:]

                mean_pred = model(context.to(device).float(),
                                   torch.zeros(batch.shape[0], ).to(device), 
                                   return_dict=False)[0]

                # Calculate the loss
                val_loss += loss_fn(mean_pred, target).item()
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
    return np.concatenate(losses, axis=1), np.concatenate(predictions, axis=1)



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
        #self.fc4 = nn.Linear(hidden_dim // 4, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.batchnorm2 = nn.BatchNorm1d(hidden_dim)
        self.batchnorm3 = nn.BatchNorm1d(hidden_dim // 2)
        #self.batchnorm4 = nn.BatchNorm1d(hidden_dim // 4)

        self.sequential = nn.Sequential(
            self.fc1,
            self.batchnorm2,
            self.relu,
            self.dropout,
            self.fc2,
            self.batchnorm3,
            self.relu,
            self.dropout,
            self.fc3,
            # self.batchnorm4,
            # self.relu,
            # self.dropout,
            # self.fc4
        )

    def forward(self, x, t, return_dict=False):
        """forward pass of the model"""
        ndim = x.ndim
        if ndim == 3:
            x = x.squeeze(2)
        x = self.sequential(x)
        if ndim == 3:
            x = x.unsqueeze(2)
        return (x, None)

from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
def get_loader_point_ds(df, stats, mode, batch_size=2048, shuffle=False, dropna=True):
    """ small helper function to prepare the dataset for pointwise mlp model"""
    
    ds = df.dropna() if dropna else df
    ds = ds.values[:, :, np.newaxis]
    ds = normalize_dss([ds], stats, mode)[0]
    ds = torch.tensor(ds).float()
    dataloader = DataLoader(TensorDataset(ds), batch_size=batch_size, shuffle=False)
    return dataloader
    


    

