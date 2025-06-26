
import fco2models.umeanest
from utils import add_src_and_logger

save_dir = f'../models/sota_ensemble_anoms/'
DATA_PATH, logger = add_src_and_logger(False, save_dir)

import pandas as pd 
import numpy as np
import xarray as xr
import torch
# import dataloader and tensordataset
from torch.utils.data import TensorDataset, DataLoader
import torch.utils.data as data
import torch.optim as optim
import torch.nn as nn
from diffusers.optimization import get_cosine_schedule_with_warmup, get_constant_schedule

import json
import logging

from fco2models.utraining import prep_df, normalize_dss, save_losses_and_png, get_stats
from fco2models.umeanest import train_mean_estimator, MLPModel, train_pointwise_mlp
from fco2models.models import MLPEnsemble, MLPNaiveEnsemble
import fco2models

np.random.seed(1)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

lr = 1e-2
batch_size = 2048

logging.info("------------ Starting training ------------------")

def add_xco2(df, xco2_mbl):
    selector = df[['lat_005', 'time_1d']].to_xarray()
    # rename the columns to match the xarray dataset
    selector = selector.rename({'lat_005': 'lat', 'time_1d': 'time'})
    #xco2mbl = xr.open_dataarray('../data/atmco2/xco2mbl-timeP7D_1D-lat25km.nc')
    matched_xco2 = xco2_mbl.sel(**selector, method='nearest').to_series()
    
    df['xco2'] = matched_xco2

    return df 

def add_clims(df, co2_clim):
    selector = df[['lat_005', 'lon_005', 'day_of_year']].to_xarray()
    # rename the columns to match the xarray dataset
    selector = selector.rename({'lat_005': 'lat', 'lon_005': 'lon', 'day_of_year': 'dayofyear'})
    #url = 'https://data.up.ethz.ch/shared/.gridded_2d_ocean_data_for_ML/co2_clim/prior_dfco2-lgbm-ens_avg-t46y720x1440.zarr/'
    #co2_clim = xr.open_zarr(url)
    df['co2_clim8d'] = co2_clim.dfco2_clim_smooth.sel(**selector, method='nearest')
    return df

def add_seamask(df, masks):
    selector = df[['lat_005', 'lon_005']].to_xarray()
    selector = selector.rename({'lat_005': 'lat', 'lon_005': 'lon'})
    df['seamask'] = masks.seamask.sel(**selector, method='nearest')
    return df

dfs = []
xco2_mbl = xr.open_dataarray('../data/atmco2/xco2mbl-timeP7D_1D-lat25km.nc')
co2_clim = xr.open_zarr('https://data.up.ethz.ch/shared/.gridded_2d_ocean_data_for_ML/co2_clim/prior_dfco2-lgbm-ens_avg-t46y720x1440.zarr/')
masks = xr.open_dataset("../data/masks/RECCAP2_masks.nc")
for year in range(1982, 2022):
    print(f"Processing year: {year}")
    df = pd.read_parquet(f'../data/SOCATv2024-1d_005deg-colloc-r20250224/SOCATv2024_1d_005deg_collocated_{year}-r20250224.pq', engine='pyarrow')
    print(f"Loaded data for year {year}, shape: {df.shape}")
    #add day_of_year column
    df.reset_index(inplace=True)
    df['day_of_year'] = df['time_1d'].dt.dayofyear
    df['year'] = year
    # df = add_xco2(df, xco2_mbl)
    # print(f"Added xco2, shape: {df.shape}")
    # df = add_clims(df, co2_clim)
    # print(f"Added co2_clim, shape: {df.shape}")
    # df = add_seamask(df, masks)
    # print(f"Added seamask, shape: {df.shape}")
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)
df = add_clims(df, co2_clim)
print("Added co2_clim, shape:", df.shape)
df = add_xco2(df, xco2_mbl)
print("Added xco2, shape:", df.shape)
df = add_seamask(df, masks)
print("Added seamask, shape:", df.shape)
print(f"Concatenated data, shape: {df.shape}")

# remove entries with high ice concentration
logger.info(f"Removed ice concentration > 0.8")
df = df[df.ice_cci < 0.8]


# renane lon and lat columns
df = df.rename(columns={'lon_005':'lon', 'lat_005': 'lat'})
df = prep_df(df, bound=True, logger=logging)[0]
df['sst_clim'] += 273.15
df['sst_anom'] = df['sst_cci'] - df['sst_clim']
df['sss_anom'] = df['sss_cci'] - df['sss_clim']
df['chl_anom'] = df['chl_globcolour'] - df['chl_clim']
df['ssh_anom'] = df['ssh_sla'] - df['ssh_clim']
df['mld_anom'] = df['mld_dens_soda'] - df['mld_clim']

test_months = pd.date_range('1982-01', '2022-01', freq='7MS').values.astype('datetime64[M]')
months = df.time_1d.values.astype('datetime64[M]')
mask_test = np.isin(months, test_months)

df_train = df[~mask_test]
df_val = df[mask_test]
logger.info(f"Removed all points not in seamask in validation set")
df_val = df_val[df_val.seamask == 1]

# drop nan rows
# predictors = ['sst_cci', 'sss_cci', 'chl_globcolour', 'ssh_sla', 
#               'mld_dens_soda', 'xco2', 'co2_clim8d', 'sin_day_of_year',
#               'cos_day_of_year', 'sin_lat', 'sin_lon_cos_lat', 'cos_lon_cos_lat']
predictors = ['sst_anom', 'sss_anom', 'chl_anom', 'ssh_anom', 'mld_anom',
              'sst_clim', 'sss_clim', 'chl_clim', 'ssh_clim', 'mld_clim',
              'xco2', 'co2_clim8d']
target = 'fco2rec_uatm'
df_train = df_train[[target] + predictors].dropna()
df_val = df_val[[target] + predictors].dropna()



train_ds = df_train.values
val_ds = df_val.values

import lightgbm as lgb
from sklearn import metrics
y_train = train_ds[:, 0]
X_train = train_ds[:, 1:]
y_val = val_ds[:, 0]
X_val = val_ds[:, 1:]


# model = lgb.LGBMRegressor(
#     n_estimators=300,  # these are just guesses - but probably good enough for this test
#     learning_rate=0.1, 
#     num_leaves=84,
#     min_split_gain=0.05,
# )

# model.fit(X_train, y_train)
# y_hat_val = model.predict(X_val)
# y_hat_train = model.predict(X_train)
# val_rmse = metrics.root_mean_squared_error(y_val, y_hat_val)
# print(f"validation RMSE: {val_rmse}")
# train_rmse = metrics.root_mean_squared_error(y_train, y_hat_train)
# print(f"train RMSE: {train_rmse}")
# #bias
# val_bias = np.mean(y_val - y_hat_val)
# print(f"validation bias: {val_bias}")
# train_bias = np.mean(y_train - y_hat_train)
# print(f"train bias: {train_bias}")
# # r2
# val_r2 = metrics.r2_score(y_val, y_hat_val)
# print(f"validation r2: {val_r2}")
# train_r2 = metrics.r2_score(y_train, y_hat_train)
# print(f"train r2: {train_r2}")

# train_ds = np.concatenate([train_ds, y_hat_train.reshape(-1, 1)], axis=1)
# val_ds = np.concatenate([val_ds, y_hat_val.reshape(-1, 1)], axis=1)
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
    "hidden_dim": 128,
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
num_epochs = 30
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

# def train_pointwise_mlp_ensemble(model_params, num_models, lr_scheduler, save_dir, **kwargs):
#     models = []
#     train_losses = []
#     val_losses = []
#     for i in range(num_models):
#         logging.info(f"Training model {i+1}/{num_models}")
#         model = MLPModel(**model_params)
#         optimizer = optim.Adam(model.parameters(), lr=lr)
#         save_dir_i = save_dir + f'model_{i+1}/'
#         model, train_loss, val_loss = train_pointwise_mlp(model,
#                                                           #num_epochs=num_epochs, 
#                                                           #old_epoch=0,
#                                                           optimizer=optimizer, 
#                                                           lr_scheduler=lr_scheduler,
#                                                           save_model_path=save_dir_i,
#                                                           **kwargs)
#         models.append(model)
#         train_losses.append(train_loss)
#         val_losses.append(val_loss)
#         logging.info(f"Model {i+1}/{num_models} trained")
#     return models, train_losses, val_losses
    


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
# model, train_losses, val_losses = train_pointwise_mlp_ensemble(model_params,
#                                                                num_models=5,
#                                                                num_epochs=num_epochs, 
#                                                                old_epoch=epoch,
#                                                                #optimizer=optimizer, 
#                                                                lr_scheduler=lr_scheduler,
#                                                                train_dataloader=train_dataloader,
#                                                                val_dataloader=val_dataloader,
#                                                                save_model_path=save_dir,
#                                                                rmse_const=train_stats['stds'][0],
#                                                                )


train_losses_old += train_losses
val_losses_old += val_losses

save_losses_and_png(train_losses_old, val_losses_old, save_dir)

logging.info("Completed training")
logging.info("Training losses: %s", train_losses)
logging.info("Validation losses: %s", val_losses)





