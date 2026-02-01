from utils import add_src_and_logger
save_dir = f'../models/sota_anoms64/'
DATA_PATH, logger = add_src_and_logger(True, save_dir)

import pandas as pd 
import numpy as np
import xarray as xr
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.utils.data as data
import torch.optim as optim
import torch.nn as nn
from diffusers.optimization import get_cosine_schedule_with_warmup, get_constant_schedule

import json
import logging

from fco2models.utraining import prep_df, normalize_dss, save_losses_and_png, get_stats, make_monthly_split
from fco2models.umeanest import train_mean_estimator, MLPModel, train_pointwise_mlp
from fco2models.models import MLPEnsemble, MLPNaiveEnsemble

np.random.seed(1)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

lr = 1e-2
batch_size = 2048

logging.info("------------ Starting training ------------------")

dfs = []
xco2_mbl = xr.open_dataarray(DATA_PATH+'atmco2/xco2mbl-timeP7D_1D-lat25km.nc')
co2_clim = xr.open_zarr('https://data.up.ethz.ch/shared/.gridded_2d_ocean_data_for_ML/co2_clim/prior_dfco2-lgbm-ens_avg-t46y720x1440.zarr/')
masks = xr.open_dataset(DATA_PATH+"masks/RECCAP2_masks.nc")
for year in range(1982, 2022):
    print(f"Processing year: {year}")
    df = pd.read_parquet(f'{DATA_PATH}SOCATv2024-1d_005deg-colloc-r20250224/SOCATv2024_1d_005deg_collocated_{year}-r20250224.pq', engine='pyarrow')
    print(f"Loaded data for year {year}, shape: {df.shape}")
    #add day_of_year column
    df.reset_index(inplace=True)
    df['day_of_year'] = df['time_1d'].dt.dayofyear
    df['year'] = year
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)
print("Added seamask, shape:", df.shape)
print(f"Concatenated data, shape: {df.shape}")

# remove entries with high ice concentration
logger.info(f"Removed ice concentration > 0.8")
df = df[df.ice_cci < 0.8]


# renane lon and lat columns
df = df.rename(columns={'lon_005':'lon', 'lat_005': 'lat'})
df = prep_df(df, bound=True, logger=logging)[0]

mask_train, mask_val, mask_test = make_monthly_split(df)
df_train = df.loc[df.expocode.map(mask_train), :]
df_val = df.loc[df.expocode.map(mask_val), :]
df_test = df.loc[df.expocode.map(mask_test), :]

logger.info(f"Removed all points not in seamask in validation and train set")
df_train = df_train[df_train.seamask == 1]
df_val = df_val[df_val.seamask == 1]

predictors = ['sst_anom', 'sss_anom', 'chl_anom', 'ssh_anom', 'mld_anom',
              'sst_clim', 'sss_clim', 'chl_clim', 'ssh_clim', 'mld_clim',
              'xco2', 'co2_clim8d']
target = 'fco2rec_uatm'
df_train = df_train[[target] + predictors].dropna()
df_val = df_val[[target] + predictors].dropna()

train_ds = df_train.values
val_ds = df_val.values

# add an extra dimension for compatibility
train_ds = train_ds[:, :, np.newaxis]
val_ds = val_ds[:, :, np.newaxis]

print(f"train_ds shape: {train_ds.shape}")
print(f"val_ds shape: {val_ds.shape}")
logging.info(f"train_ds shape: {train_ds.shape}")
logging.info(f"val_ds shape: {val_ds.shape}")


# normalize the data
mode = 'mean_std'
train_stats = get_stats(train_ds, logger=logging)
train_ds, val_ds = normalize_dss([train_ds, val_ds], train_stats, mode, logger=logging)

# remove last dimension
train_ds = train_ds[:, :, 0]
val_ds = val_ds[:, :, 0]
train_dataset = TensorDataset(torch.tensor(train_ds))
val_dataset = TensorDataset(torch.tensor(val_ds))
train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


model_params_mlp = {
    "input_dim": train_ds.shape[1] - 1,
    "hidden_dim": 64,
    "output_dim": 1,
}

model_params = {
    "ensemble_size": 20,
    "mlp_class": "fco2models.umeanest.MLPModel",
    "mlp_kwargs": model_params_mlp,
}

model = MLPNaiveEnsemble(**model_params)
def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Number of trainable parameters: {count_trainable_parameters(model)}")
num_epochs = 50
optimizer = optim.Adam(model.parameters(), lr=lr)

lr_params = {}
lr_scheduler = get_constant_schedule(optimizer)

#checkpoint_path = ""
epoch = 0
train_losses_old = []
val_losses_old = []
#model, optimizer, scheduler, epoch, train_losses_old, val_losses_old = load_checkpoint(checkpoint_path, model, optimizer, lr_scheuduler)

# lr_params = {
#     "num_warmup_steps": 0.05 * num_epochs * len(train_dataloader), 
#     "num_training_steps": num_epochs * len(train_dataloader)
#     }
# lr_scheduler = get_cosine_schedule_with_warmup(optimizer, **lr_params)
# save all hyperparameters to a json file
param_dict = {
    "model_params": model_params,
    "lr_params": lr_params,
    "noise_params": None,
    "batch_size": batch_size,
    "num_epochs": num_epochs,
    "lr": lr,
    "optimizer": optimizer.__class__.__name__,
    "predictors": predictors,
    "train_means": train_stats['means'],
    "train_stds": train_stats['stds'],
    "train_mins": train_stats['mins'],
    "train_maxs": train_stats['maxs'],
    "mode": mode,
    }
logging.info("All parameters: %s", param_dict)
# save the model parameters to a json file
with open(save_dir +'hyperparameters.json', 'w') as f:
    param_dict = json.dumps(param_dict, indent=4)
    f.write(param_dict)
    


model, train_losses, val_losses = train_pointwise_mlp(model,
                                                       num_epochs=num_epochs, 
                                                       old_epoch=epoch,
                                                       optimizer=optimizer, 
                                                       lr_scheduler=lr_scheduler,
                                                       train_dataloader=train_dataloader,
                                                       val_dataloader=val_dataloader,
                                                       save_model_path=save_dir,
                                                       rmse_const=train_stats['stds'][0],
                                                       )


train_losses_old += train_losses
val_losses_old += val_losses

save_losses_and_png(train_losses_old, val_losses_old, save_dir)

logging.info("Completed training")
logging.info("Training losses: %s", train_losses)
logging.info("Validation losses: %s", val_losses)

