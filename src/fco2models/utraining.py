import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
# import wandb
from tqdm import tqdm
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers import DDPMScheduler, UNet1DModel
import time
import pandas as pd

def check_gradients(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    print(f"Total gradient norm: {total_norm}")

import numpy as np


def sinusoidal_day_embedding(num_days=365, d_model=64):
    """
    Create sinusoidal embeddings for days of the year.
    
    Args:
        num_days (int): Number of time steps (typically 365 for days in a year)
        d_model (int): Dimension of the embedding vector (must be even)
        
    Returns:
        np.ndarray of shape (num_days, d_model)
    """
    assert d_model % 2 == 0, "Embedding dimension (d_model) must be even."
    
    days = np.arange(num_days).reshape(-1, 1)  # Shape: (365, 1)
    div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))  # Shape: (d_model/2,)
    
    pe = np.zeros((num_days, d_model))
    pe[:, 0::2] = np.sin(days * div_term)
    pe[:, 1::2] = np.cos(days * div_term)
    
    return pe

    


# Training function
def train_diffusion(model, num_epochs, train_dataloader, val_dataloader, noise_scheduler, optimizer, lr_scheduler, save_model_path=None):
    """training loop for diffusion model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    model.to(device)
    
    loss_fn = nn.MSELoss()
    
    # Initialize wandb
    # wandb.init(project="conditional-diffusion")
    # wandb.watch(model)
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        # noise = torch.randn((batch_size,1,64)).to(device)
        
        for batch in progress_bar:
            batch = batch[0].to(device)
            target = batch[:, 0:1, :]
            context = batch[:, 1:, :]
            noise = torch.randn_like(target).to(device).float()
            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (batch.shape[0],), device=device).long()
            nan_mask = torch.isnan(target)
            # replace nan with zeros
            target = torch.where(nan_mask, torch.zeros_like(target), target)
            # Add noise to the clean images according to the noise magnitude at each timestep
            noisy_target = noise_scheduler.add_noise(target, noise, timesteps)

            #concatenate the noisy target with the context and the mask
            noisy_input = torch.cat([noisy_target, context, (~nan_mask).float()], dim=1)
            noisy_input = noisy_input.to(device).float()
            
            # Get the model prediction
            noise_pred = model(noisy_input, timesteps, return_dict=False)[0]

            # Calculate the loss
            optimizer.zero_grad()
            loss = loss_fn(noise_pred[~nan_mask], noise[~nan_mask])	
            loss.backward(loss)
            epoch_loss += loss.item()
            
            # Update the model parameters with the optimizer
            optimizer.step()

            # Update the learning rate
            lr_scheduler.step()

            #update loss in progress bar
            progress_bar.set_postfix({"Loss": loss.item()})
            # batch.detach()
        train_losses.append(epoch_loss / len(train_dataloader))
        print(f"Epoch {epoch+1} Loss: {epoch_loss / len(train_dataloader):.6f}")
        check_gradients(model)

        # print validation loss
        model.eval()
        t_tot = noise_scheduler.config.num_train_timesteps
        val_losses_t = []
        for t in range(0, t_tot, t_tot//10):
            val_loss = 0.0
            for batch in val_dataloader:
                timesteps = torch.full((batch[0].shape[0],), t, device=device, dtype=torch.long)
                noisy_input, noise, nan_mask, timesteps = prep_sample(batch, noise_scheduler, timesteps, device)
                noise_pred = model(noisy_input, timesteps, return_dict=False)[0]
                loss = loss_fn(noise_pred[~nan_mask], noise[~nan_mask])
                val_loss += loss.item()
            val_losses_t.append(val_loss / len(val_dataloader))
            print(f"Validation Loss for timestep {t}: {val_loss / len(val_dataloader):.6f}")
        print(f"Epoch {epoch+1} Validation Loss: {np.mean(val_losses_t):.6f}")
        val_losses.append(val_losses_t)

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


def prep_sample(batch, noise_scheduler, timesteps, device):
    batch = batch[0].to(device)
    target = batch[:, 0:1, :]
    context = batch[:, 1:, :]
    
    noise = torch.randn_like(target).to(device).float()
    # Replace nan with zeros
    nan_mask = torch.isnan(target)
    target = torch.where(nan_mask, torch.zeros_like(target), target)
    # Add noise to the clean images according to the noise magnitude at each timestep
    noisy_target = noise_scheduler.add_noise(target, noise, timesteps)
    # Concatenate the noisy target with the context
    noisy_input = torch.cat([noisy_target, context, (~nan_mask).float()], dim=1)
    noisy_input = noisy_input.to(device).float()
    return noisy_input, noise, nan_mask, timesteps



def full_denoise(model, noise_scheduler, context_loader, jump=None):
    """full denoising loop for diffusion model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    model.to(device)
    samples = []
    # tdqm progress bar for context_loader
    context_loader = tqdm(context_loader, desc=f"Inference")
    for context_batch in context_loader:
        context = context_batch.to(device)
        # context = context.unsqueeze(0)
        sample = torch.randn((context.shape[0], 1, context.shape[2])).to(device)
        mask = torch.ones_like(sample).float().to(device)
        sample_context = torch.zeros(context.shape[0], context.shape[1] + 2, context.shape[2]).to(device)
        sample_context[:, 0:1, :] = sample
        sample_context[:, 1:-1, :] = context
        sample_context[:, -1:, :] = mask
        if jump is not None:
            timestep = noise_scheduler.timesteps[::jump]
        else:
            timestep = noise_scheduler.timesteps
        for t in timestep:
            # concat noise, context and mask
            sample_context[:, 0:1, :] = sample
            # Get model pred
            with torch.no_grad():
                residual = model(sample_context, t, return_dict=False)[0]

            output_scheduler = noise_scheduler.step(residual, t, sample)
            if jump is not None:
                x_0 = output_scheduler.pred_original_sample
                if t < jump:
                    sample = x_0
                else:
                    sample = noise_scheduler.add_noise(x_0, torch.randn_like(sample), t - jump)
            else:
                # Update sample with step
                sample = output_scheduler.prev_sample
            # end_time = time.time()
            # update progress bar
            context_loader.set_postfix({"timestep":  t})
        context_batch.detach()
        samples.append(sample.detach().cpu())
    samples = torch.cat(samples, dim=0)

    return np.array(samples)


def df_to_ds(df,):
    num_bins = df.index.get_level_values('bin').nunique()
    num_segments = df.index.get_level_values('segment').nunique()
    ds_reconstructed = np.zeros((len(df.columns), num_segments, num_bins), dtype=np.float32)
    for i, col in enumerate(df.columns):
        ds_reconstructed[i, :, :] = df[col].values.reshape(num_segments, num_bins)
    return ds_reconstructed


import logging as log
from fco2dataset.ucruise import filter_nans
def prep_data(df, predictors, logging=None):
    """prepare data for training"""
    
    if logging is None:
        # log to stdout
        logging = log
        log.basicConfig(
            level=log.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[log.StreamHandler()]
        )
    
    ds_raw = df_to_ds(df)
    col_map = dict(zip(df.columns, range(len(df.columns))))
    
    # fill missing sss_cci values with salt_soda values
    logging.info("Filling missing sss_cci values with salt_soda values")
    salt_soda = ds_raw[col_map['salt_soda']]
    sss_cci = ds_raw[col_map['sss_cci']]
    mask = np.isnan(sss_cci)
    ds_raw[col_map['sss_cci'], mask] = salt_soda[np.isnan(sss_cci)]
    
    y = ds_raw[0]
    # logging.info("Checking for nans in y")
    # assert np.apply_along_axis(lambda x: np.isnan(x).all(), 1, y).sum() == 0
    logging.info("predictors: %s", predictors)
    ds_map = dict(zip(predictors, range(1, len(predictors) + 1)))
    X, y = filter_nans(ds_raw[:, :, :], y[:, :], predictors, col_map)
    print(X.shape, y.shape)

    # assert np.apply_along_axis(lambda x: np.isnan(x).all(), 1, y).sum() == 0
    
    print(X.shape, y[np.newaxis].shape)
    assert np.isnan(X).sum() == 0
    n_samples = X.shape[1]
    n_dims = X.shape[2]
    ds = np.zeros((n_samples, X.shape[0] + 1, n_dims))
    
    ds[:, 0, :] = y
    for i in range(X.shape[0]):
        ds[:, i + 1, :] = X[i]
    
    # clip 0th channel to 0-500
    print("number of fco2 measurements greater than 500: ", np.sum(ds[:, 0, :] > 500))
    logging.info("clipping fco2 values to 0-500")
    ds[:, 0, :] = np.clip(ds[:, 0, :], 0, 500)
    

    if 'lat' in predictors:
        lat_col = ds_map['lat']
        logging.info("add latitude feature")
        # round latitude column to 1 degree and shift to range 0-180
        ds[:, lat_col, :] = (np.rint(ds[:, lat_col, :]) + 90).astype(int)
        sinemb_lat = sinusoidal_day_embedding(num_days=181, d_model=64)
        ds[:, lat_col, :] = sinemb_lat[ds[:, lat_col, 0].astype(int), :] # take first bin for latitude feature

    if 'lon' in predictors:
        lat_col = ds_map['lon']
        logging.info("add longitude feature")
        # round latitude column to 1 degree and shift to range 0-180
        ds[:, lat_col, :] = (np.rint(ds[:, lat_col, :])).astype(int)
        sinemb_lon = sinusoidal_day_embedding(num_days=361, d_model=64)
        ds[:, lat_col, :] = sinemb_lon[ds[:, lat_col, 0].astype(int), :] # take first bin for latitude feature
    
    if 'day_of_year' in ds_map:
        ix_day = ds_map['day_of_year']
        logging.info("add day of year feature")
        # embed time feature
        sinemb_day = sinusoidal_day_embedding(num_days=365, d_model=64)
        ix_day = ds_map['day_of_year']
        # clip to 0-364
        ds[:, ix_day, :] = np.clip(ds[:, ix_day, :] - 1, 0, 364)
        ds[:, ix_day, :] = sinemb_day[ds[:, ix_day, 0].astype(int), :] # just take the first bin for the time feature

    return ds

