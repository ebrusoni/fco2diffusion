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
from fco2models.models import ClassEmbedding

def check_gradients(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    print(f"Total gradient norm: {total_norm}")

def pos_to_timestep(pos_encodings, noise_scheduler):
    # since I am using the same embedding for timesteps and positions, I have to map them to the same range
    # this is probably nonsensical, I have five positions and I just take the mean of the positions to get a single values in [0, noise_scheduler.config.num_train_timesteps]
    return ((pos_encodings.mean(axis=(1, 2)) + 1) / 2 * noise_scheduler.config.num_train_timesteps).long()

import numpy as np
# Training function
def train_diffusion(model, num_epochs, old_epoch, train_dataloader, val_dataloader, noise_scheduler, optimizer, lr_scheduler, save_model_path=None, 
                    pos_encodings_start=None, device=None, class_embedder=None):
    """training loop for diffusion model"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    model.to(device)
    if class_embedder is not None:
        class_embedder.to(device)
    
    loss_fn = nn.MSELoss()
    
    # Initialize wandb
    # wandb.init(project="conditional-diffusion")
    # wandb.watch(model)
    train_losses = []
    val_losses = []
    for epoch in range(old_epoch, num_epochs):
        model.train()
        if class_embedder is not None:
            class_embedder.train()
        epoch_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        # noise = torch.randn((batch_size,1,64)).to(device)
        
        for batch in progress_bar:
            batch = batch[0].to(device)
            target = batch[:, 0:1, :]
            context = batch[:, 1:pos_encodings_start, :]
            pos_encodings = batch[:, pos_encodings_start:, :]
            
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
            #class_labels = None if pos_encodings_start is None else class_embedder(pos_encodings.int())
            noise_pred = model(noisy_input, timesteps, return_dict=False)[0]

            # Calculate the loss
            optimizer.zero_grad()
            loss = loss_fn(noise_pred[~nan_mask], noise[~nan_mask])	
            loss.backward()
            epoch_loss += loss.item()
            
            # Update the model parameters with the optimizer
            optimizer.step()
            
            #update loss in progress bar
            progress_bar.set_postfix({"Loss": loss.item()})
            # batch.detach()
        train_losses.append(epoch_loss / len(train_dataloader))
        print(f"Epoch {epoch+1} Loss: {epoch_loss / len(train_dataloader):.6f}")
        check_gradients(model)
        # Update the learning rate
        lr_scheduler.step()

        # print validation loss
        model.eval()
        if class_embedder is not None:
            class_embedder.eval()
        t_tot = noise_scheduler.config.num_train_timesteps
        val_losses_t = []
        for t in range(0, t_tot, t_tot//10):
            val_loss = 0.0
            for batch in val_dataloader:
                timesteps = torch.full((batch[0].shape[0],), t, device=device, dtype=torch.long)
                noisy_input, noise, nan_mask, timesteps, class_labels = prep_sample(batch, noise_scheduler, timesteps, pos_encodings_start, class_embedder, device)
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


def prep_sample(batch, noise_scheduler, timesteps, pos_encodings_start, class_embedder, device):
    batch = batch[0].to(device)
    target = batch[:, 0:1, :]
    context = batch[:, 1:pos_encodings_start, :]
    pos_encodings = batch[:, pos_encodings_start:, :]
    
    noise = torch.randn_like(target).to(device).float()
    # Replace nan with zeros
    nan_mask = torch.isnan(target)
    target = torch.where(nan_mask, torch.zeros_like(target), target)
    # Add noise to the clean images according to the noise magnitude at each timestep
    noisy_target = noise_scheduler.add_noise(target, noise, timesteps)
    # Concatenate the noisy target with the context
    noisy_input = torch.cat([noisy_target, context, (~nan_mask).float()], dim=1)
    noisy_input = noisy_input.to(device).float()

    class_labels = None if pos_encodings_start is None else class_embedder(pos_encodings.int())
    return noisy_input, noise, nan_mask, timesteps, class_labels



def full_denoise(model, noise_scheduler, context_loader, jump=None, pos_encodings_start=None, eta=0.0):
    """full denoising loop for diffusion model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    model.to(device)
    samples = []
    # tdqm progress bar for context_loader
    context_loader = tqdm(context_loader, desc=f"Inference")
    for context_batch in context_loader:
        context = context_batch.to(device)
        context = context[:, :pos_encodings_start, :]
        pos_encodings = context_batch[:, pos_encodings_start:, :]
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
                class_labels = None if pos_encodings_start is None else pos_to_timestep(pos_encodings, noise_scheduler)
                residual = model(sample_context, t, return_dict=False, class_labels=class_labels)[0]

            output_scheduler = noise_scheduler.step(residual, t, sample, eta=eta)
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


def sinusoidal_embedding(num_days=365, d_model=64):
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

def df_to_ds(df,):
    num_bins = df.index.get_level_values('bin').nunique()
    num_segments = df.index.get_level_values('segment').nunique()
    ds_reconstructed = np.zeros((len(df.columns), num_segments, num_bins), dtype=np.float32)
    for i, col in enumerate(df.columns):
        ds_reconstructed[i, :, :] = df[col].values.reshape(num_segments, num_bins)
    return ds_reconstructed


import logging as log
from fco2dataset.ucruise import filter_nans
def prepare_segment_ds(dfs, predictors, logging=None, with_mask=False, info=None):
    """prepare data for training (only for models working with segments)
         - filters out nans from all predictor variables
         - adds sinusoidal embeddings for lat, lon and day of year if lat, lon and day_of_year are in predictors
         - adds a mask for the first column (target variable) if with_mask is True"""
    
    logging = make_logger(logging)
    dss = []
    return_mask = False
    if info is not None:
        logging.info("info features: %s", info)
        return_mask = True
    for df in dfs:
    
        ds_raw = df_to_ds(df)
        col_map = dict(zip(df.columns, range(len(df.columns))))
    
    
        logging.info("predictors: %s", predictors)
        ds_map = dict(zip(predictors, range(1, len(predictors) + 1)))
        yX, filter_mask = filter_nans(ds_raw, predictors, col_map, return_mask=return_mask)
        print(f"yX shape: {yX.shape}")
    
        # assert np.apply_along_axis(lambda x: np.isnan(x).all(), 1, y).sum() == 0
        
        assert np.isnan(yX[1:, :, :]).sum() == 0
        n_samples = yX.shape[1]
        n_dims = yX.shape[2]
        ds = np.zeros((n_samples, yX.shape[0], n_dims))
        
        for i in range(yX.shape[0]):
            ds[:, i, :] = yX[i]

        if info is not None:
            infos = ds_raw[[col_map[i] for i in info]]
            infos = infos[:, filter_mask, :]
            info_ds = np.zeros((n_samples, len(info), n_dims))
            for i in range(len(info)):
                info_ds[:, i, :] = infos[i]
        
        if 'lat' in predictors:
            lat_col = ds_map['lat']
            logging.info("add latitude feature")
            # round latitude column to 1 degree and shift to range 0-180
            ds[:, lat_col, :] = (np.rint(ds[:, lat_col, :]) + 90).astype(int)
            sinemb_lat = sinusoidal_embedding(num_days=181, d_model=64)
            ds[:, lat_col, :] = sinemb_lat[ds[:, lat_col, 0].astype(int), :] # take first bin for latitude feature
    
        if 'lon' in predictors:
            lat_col = ds_map['lon']
            logging.info("add longitude feature")
            # round latitude column to 1 degree and shift to range 0-180
            ds[:, lat_col, :] = (np.rint(ds[:, lat_col, :])).astype(int)
            sinemb_lon = sinusoidal_embedding(num_days=361, d_model=64)
            ds[:, lat_col, :] = sinemb_lon[ds[:, lat_col, 0].astype(int), :] # take first bin for latitude feature
        
        if 'day_of_year' in ds_map:
            ix_day = ds_map['day_of_year']
            logging.info("add day of year feature")
            # embed time feature
            sinemb_day = sinusoidal_embedding(num_days=365, d_model=64)
            ix_day = ds_map['day_of_year']
            # clip to 0-364
            ds[:, ix_day, :] = np.clip(ds[:, ix_day, :] - 1, 0, 364)
            ds[:, ix_day, :] = sinemb_day[ds[:, ix_day, 0].astype(int), :] # just take the first bin for the time feature

    
        if with_mask:
            # adding additional channel with nan mask for first column
            mask = np.zeros((n_samples, 1, n_dims), dtype=bool)
            mask[:, 0, :] = np.isnan(ds[:, 0, :])
            ds = np.concatenate([ds, ~mask], axis=1)
        
        if info is not None:
            dss.append((ds, info_ds))
        else:
            dss.append(ds)

    return dss

def add_clims(df, co2_clim):
    """add climatology data to the dataframe"""
    selector = df[['lat', 'lon', 'day_of_year']].to_xarray()
    # rename the columns to match the xarray dataset
    selector = selector.rename({'day_of_year': 'dayofyear'})
    #url = 'https://data.up.ethz.ch/shared/.gridded_2d_ocean_data_for_ML/co2_clim/prior_dfco2-lgbm-ens_avg-t46y720x1440.zarr/'
    #co2_clim = xr.open_zarr(url)
    df['co2_clim8d'] = co2_clim.dfco2_clim_smooth.sel(**selector, method='nearest')
    return df

def add_xco2(df, xco2_mbl):
    selector = df[['lat', 'time_1d']].to_xarray()
    # rename the columns to match the xarray dataset
    selector = selector.rename({'time_1d': 'time'})
    #xco2mbl = xr.open_dataarray('../data/atmco2/xco2mbl-timeP7D_1D-lat25km.nc')
    matched_xco2 = xco2_mbl.sel(**selector, method='nearest').to_series()
    
    df['xco2'] = matched_xco2

    return df 
import xarray as xr
def prep_df(dfs, logger=None, bound=False, index=None, with_target=True, with_log=True, add_clim=True, add_seas=True):
    """prepare dataframe for training
        - the idea is to use it for "segment independent" feature extraction (which is easier to do in a dataframe)
        - this is a bit of a hack, but it works for now
        - should be usable for learning pointwise estimates and the segmented estimates
    """
    logger = make_logger(logger)
    if not isinstance(dfs, list):
        dfs = [dfs]
    
    res = []
    for df in dfs:
        df.reset_index(inplace=True)
        # add day of year feature if not present
        if 'day_of_year' not in df.columns:
            df['day_of_year'] = df['time_1d'].dt.dayofyear
        if with_log:
            logger.info("salinity stacking")
        df['sss_cci'] = df['sss_cci'].fillna(df['salt_soda'])

        if with_log:
            logger.info("adding positional and temporal encodings")
        df['sin_day_of_year'] = np.sin(df['day_of_year']* np.pi / 365)
        df['cos_day_of_year'] = np.cos(df['day_of_year']* np.pi / 365)
        # normalize lons to range [-180, 180] from [0, 360]
        df['lon'] = (df['lon'] + 180) % 360 - 180
        df['sin_lat'] = np.sin(df['lat'] * np.pi / 180)
        # embed lat and lon features
        df['sin_lon_cos_lat'] = np.sin(df['lon'] * np.pi / 180) * np.cos(df['lat'] * np.pi / 180)
        df['cos_lon_cos_lat'] = - np.cos(df['lon'] * np.pi / 180) * np.cos(df['lat'] * np.pi / 180)

        df['sin_lon'] = np.sin(df['lon'] * np.pi / 180)
        df['cos_lon'] = np.cos(df['lon'] * np.pi / 180)
        df['is_north'] = df['lat'] > 0
    
        #logger.info("clip values of fco2 between 0 and 400")
        #df['fco2rec_uatm'] = df['fco2rec_uatm'].clip(lower=None, upper=400)
        if with_log:
            logger.info("add climatology data")
        co2_clim = xr.open_zarr('https://data.up.ethz.ch/shared/.gridded_2d_ocean_data_for_ML/co2_clim/prior_dfco2-lgbm-ens_avg-t46y720x1440.zarr/')
        df = add_clims(df, co2_clim)
        
        # add xco2 data if not present
        if 'xco2' not in df.columns:
            if with_log:
                logger.info("adding xco2 data")
            xco2_mbl = xr.open_dataarray('https://data.up.ethz.ch/shared/.gridded_2d_ocean_data_for_ML/xco2mbl-timeP7D_1D-lat25km.nc')
            #xco2_mbl = xr.open_dataarray('../data/atmco2/xco2mbl-timeP7D_1D-lat25km.nc')
            df = add_xco2(df, xco2_mbl)
        
        if 'seamask' not in df.columns:
            if with_log:
                logger.info("adding seamask data")
            masks = xr.open_dataset('/home/jovyan/work/datapolybox/masks/RECCAP2_masks.nc')
            # masks = xr.open_dataset("../data/masks/RECCAP2_masks.nc")
            selection = df[['lat', 'lon']].to_xarray()
            df['seamask'] = masks.seamask.sel(selection, method='nearest')
        
        if add_clim:
            if with_log:
                logger.info("adding climatology data")
            clims = xr.open_dataset("https://data.up.ethz.ch/shared/.gridded_2d_ocean_data_for_ML/inference_for_gregor2024/clims_8daily_25km_v01.zarr/", engine='zarr')
            selection = df[['lat', 'lon', 'day_of_year']].to_xarray()
            selection = selection.rename({'day_of_year': 'dayofyear'})
            clims_df = clims.sel(selection, method='nearest').to_dataframe()
            # rename columns to match the dataframe
            clims_df = clims_df.rename(columns=lambda col: col + "_clim")
            # add climatology data to the dataframe
            df = pd.concat([df, clims_df], axis=1)

        if add_seas:
            if with_log:
                logger.info("adding climatology data")
            clims = xr.open_dataset("https://data.up.ethz.ch/shared/.gridded_2d_ocean_data_for_ML/inference_for_gregor2024/seas_8daily_25km_v01.zarr/", engine='zarr')
            selection = df[['lat', 'lon', 'day_of_year']].to_xarray()
            selection = selection.rename({'day_of_year': 'dayofyear'})
            clims_df = clims.sel(selection, method='nearest').to_dataframe()
            # rename columns to match the dataframe
            clims_df = clims_df.rename(columns=lambda col: col + "_seas")
            # add climatology data to the dataframe
            df = pd.concat([df, clims_df], axis=1)
        
        if with_target:
            logger.info("removing xco2 levels from fco2rec_uatm")
            df['fco2rec_uatm'] = df['fco2rec_uatm'] - df['xco2']
    
        if bound:
            if with_log:
                logger.info("replacing outliers with Nans, fco2rec_uatm > 400")
            # fco2rec_uatm_95th = df['fco2rec_uatm'].quantile(0.95)
            # fco2rec_uatm_5th = df['fco2rec_uatm'].quantile(0.05)
            # df['fco2rec_uatm'] = df['fco2rec_uatm'].clip(lower=fco2rec_uatm_5th, upper=fco2rec_uatm_95th)
            df.loc[df.fco2rec_uatm > 400, 'fco2rec_uatm'] = np.nan
        if index is not None:
            # set index to the given index
            df.set_index(index, inplace=True)
        res.append(df)
    
    return res

def quantize_positional_encodings(dfs, positional_encodings, num_bins, logger=None):
    logger = make_logger(logger)
    logger.info(f"quantizing positional encodings {positional_encodings} to {num_bins} bins") 
    res = []
    for df in dfs:
        # assumes all positional encodings are in the same range [-1,1]
        df[positional_encodings] = (((df[positional_encodings] + 1) / 2) * num_bins).round().astype(int)
        # clip to range [0, num_bins-1]
        df[positional_encodings] = df[positional_encodings].clip(0, num_bins-1)
        res.append(df)
    return res
    
def get_augmentations(ds, aug_names):
    """get augmentations for the dataset of shape (n_samples, n_features, n_bins)"""
    if 'mirror' in aug_names:
        # mirror the dataset along the first axis
        ds = np.concatenate([ds, ds[:, :, ::-1]], axis=0)

def normalize_dss(dss, stats, mode, logger=None, ignore=None):
    logger = make_logger(logger)
    if ignore is None:
        ignore = []
    
    # use the given stats for normalization
    train_means = stats['means']
    train_stds = stats['stds']
    train_mins = stats['mins']
    train_maxs = stats['maxs']
    logger.info("Using given stats for normalization")
    
    logger.info(f"Normalizing data using {mode} normalization")
    logger.info(f"Not normalizing features: {ignore}")
    
    def normalize(x, i,  mode):
        """normalize the data"""
        if mode == 'mean_std':
            x = (x - train_means[i]) / train_stds[i]
        elif mode == 'min_max':
            # normalize -1 to 1
            x = 2 * (x - train_mins[i]) / (train_maxs[i] - train_mins[i]) - 1
        else:
            raise ValueError(f"Unknown normalization mode: {mode}")
        return x
    
    for ds in dss:
        for i in range(ds.shape[1]):
            if i in ignore:
                continue
            ds[:, i, :] = normalize(ds[:, i, :], i, mode)

    return dss

def get_stats(ds, logger=None):
    """get stats for the dataset of shape (n_samples, n_features, n_bins)"""
    logger = make_logger(logger)
    means = []
    stds = []
    mins = []
    maxs = []
    for i in range(ds.shape[1]):
        means.append(np.nanmean(ds[:, i, :]))
        stds.append(np.nanstd(ds[:, i, :]))
        mins.append(np.nanmin(ds[:, i, :]))
        maxs.append(np.nanmax(ds[:, i, :]))
    
    logger.info(f"Means: {means}")
    logger.info(f"Stds: {stds}")
    logger.info(f"Mins: {mins}")
    logger.info(f"Maxs: {maxs}")
    
    return {
        'means': means,
        'stds': stds,
        'mins': mins,
        'maxs': maxs
    }
    
    
    
def load_checkpoint(path, model, optimizer, scheduler=None, logger=None):
    """Load a checkpoint from a given path."""

    logger = make_logger(logger)
    logger.info(f"Loading checkpoint from {path}")
    # Load checkpoint
    checkpoint = torch.load(path)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    epoch = checkpoint['epoch'] + 1  # Continue from next epoch
    train_losses = checkpoint['train_losses']
    val_losses = checkpoint['val_losses']
    logger.info(f'starting from epoch {epoch + 1}')
    return model, optimizer, scheduler, epoch, train_losses, val_losses

def make_logger(logger):
    """define a logger if there is none"""
    if logger is None:
        # log to stdout
        logger = log
        log.basicConfig(
            level=log.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[log.StreamHandler()]
        )
    return logger


import json
def save_losses_and_png(train_losses, val_losses, save_dir):
    with open(save_dir+'losses.json', 'w') as f:
        losses_dict = {
            'train_losses': train_losses,
            'val_losses': val_losses
        }
        json.dump(losses_dict, f)
    
    val_losses = np.array(val_losses).T
    # plot and save the training and validation losses
    import matplotlib.pyplot as plt
    plt.plot(train_losses, label='train')
    plt.plot(val_losses, label='val')
    
    # loglog the losses
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Training and Validation Losses')
    plt.savefig(save_dir + 'losses.png')
    plt.show()

def save_losses_and_png_diffusion(train_losses, val_losses, save_dir, t_tot):
    with open(save_dir+'losses.json', 'w') as f:
        losses_dict = {
            'train_losses': train_losses,
            'val_losses': val_losses
        }
        json.dump(losses_dict, f)
    
    val_losses = np.array(val_losses).T
    # plot and save the training and validation losses
    import matplotlib.pyplot as plt
    plt.plot(train_losses, label='train')
    for (i, t) in enumerate(range(0, t_tot, t_tot//10)):
        plt.plot(val_losses[i], label=f'val {t}')
    
    # loglog the losses
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Training and Validation Losses')
    plt.savefig(save_dir + 'losses.png')
    plt.show()

# STUFF FOR NEW DATASET
def replace_with_cruise_data(segments, cruise_data, prob=0.3, logger=None):
    # segments has shape (n_samples, n_features, n_bins)
    # cruise_data has shape (n_samples, n_features, n_bins)
    # the first column of cruise_data is temperature, then salinity
    # these correspond to the 2nd and 3rd
    # if the cruise_data is not available, we keep the remote sensing data
    logger = make_logger(logger)
    logger.info(f"Replacing segments with cruise data with probability {prob}")

    # replace with probability prob
    n_segments = segments.shape[0]
    mask = np.random.rand(n_segments) < prob

    nan_mask_temp = ~np.isnan(cruise_data[mask, 0, :], axis=1)
    segments[mask, 1, nan_mask_temp] = cruise_data[mask, 0, nan_mask_temp]
    nan_mask_sal = ~np.isnan(cruise_data[mask, 1, :], axis=1)
    segments[mask, 2, nan_mask_sal] = cruise_data[mask, 1, nan_mask_sal]

    return segments

def perturb_fco2(segments, logger=None):
    """perturb fco2 values in the segments with noise +-5 uatm"""
    logger = make_logger(logger)
    logger.info("Perturbing fco2 values with noise +-5 uatm")
    noise = np.random.uniform(-5, 5, size=segments[:, 0, :].shape)
    segments[:, 0, :] += noise
    return segments
    
def get_segments_random(df, cols, num_windows=64, n=3):
    """extraxt arrays of size num_windows from the dataframe df randomly"""


    # Get the number of rows in the dataframe
    num_rows = df.shape[0]

    if num_rows < num_windows:
        return np.full((1, len(cols), num_windows), np.nan, dtype=np.float32) # Return NaN array if not enough data
    elif num_rows == num_windows:
        starting_indices = [0] # Only one segment possible
    else:
        starting_indices = np.random.randint(0, num_rows - num_windows, size=(num_rows // num_windows) * n)
    # Ensure that starting indices are unique and sorted
    starting_indices = np.unique(starting_indices)
    num_segments = len(starting_indices)
    cruise_ds = np.zeros((num_segments, len(cols), num_windows), dtype=np.float32)

    starting_indices.sort()
    for i, start in enumerate(starting_indices):
        cruise_ds[i] = df[cols].iloc[start:start + num_windows].values.T

    return cruise_ds

def get_segments(df, cols, num_windows=64, step=64, offset=0):
    """extract arrays of size num_windows from the dataframe df.
       The idea is to use this only for evaluating the model on the test set."""
    # Get the number of rows in the dataframe
    num_rows = df.shape[0]
    starting_indices = np.arange(offset, num_rows - num_windows, step)

    num_segments = len(starting_indices)
    cruise_ds = np.zeros((num_segments, len(cols), num_windows), dtype=np.float32)

    for i, start in enumerate(starting_indices):
        cruise_ds[i] = df[cols].iloc[start:start + num_windows].values.T

    return cruise_ds

def make_monthly_split(df, month_step=7, val_offset=3, leave_out_2021=False):
    end_year = 2022 if not leave_out_2021 else 2021
    print(f"Splitting data into train/val/test with parameters: month_step={month_step}, val_offset={val_offset}, leave_out_2021={leave_out_2021}")

    # list of months to use for testing and validation
    test_months = pd.date_range('1982-01', f'{end_year}-01', freq='7MS').values.astype('datetime64[M]')
    val_months = pd.date_range(f'1982-{val_offset}', f'{end_year}-01', freq='7MS').values.astype('datetime64[M]')
    # find mean date for each expocode so that we filter entire cruises
    expocode_dates = df.groupby(['expocode']).mean().time_1d.apply(lambda date: np.datetime64(date.strftime('%Y-%m')))
    mask_test = expocode_dates.isin(test_months)
    mask_val = expocode_dates.isin(val_months) & ~mask_test # those should not overlap but just in case
    mask_train = ~mask_test & ~mask_val

    return mask_train, mask_val, mask_test

def get_stats_df(df, cols, logger=None):
    """get stats for the dataframe df"""
    logger = make_logger(logger)
    means = []
    stds = []
    mins = []
    maxs = []
    for col in cols:
        means.append(np.nanmean(df[col]).astype(float))
        stds.append(np.nanstd(df[col]).astype(float))
        mins.append(np.nanmin(df[col]).astype(float))
        maxs.append(np.nanmax(df[col]).astype(float))
    
    logger.info(f"Means: {means}")
    logger.info(f"Stds: {stds}")
    logger.info(f"Mins: {mins}")
    logger.info(f"Maxs: {maxs}")
    
    return {
        'means': means,
        'stds': stds,
        'mins': mins,
        'maxs': maxs
    }

def get_context_mask(dss, logger=None):
    """returns mask for filtering out samples with Nans in the context variables"""
    logger = make_logger(logger)
    masks = []
    for ds in dss:
        context_ds = ds[:, 1:, :]
        not_nan_mask = np.isnan(context_ds).any(axis=2)
        not_nan_mask = np.sum(not_nan_mask, axis=1) == 0
        masks.append(not_nan_mask)
        logger.info(f"Number of samples after filtering: {np.sum(not_nan_mask)}")
    return masks

