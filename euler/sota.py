
from utils import add_src_and_logger

save_dir = f'../models/sota/'
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
from fco2models.umeanest import train_mean_estimator, MLPModel

np.random.seed(1)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

lr = 1e-2
batch_size = 2048

logging.info("------------ Starting training ------------------")

def add_xco2(df):
    selector = df[['lat_005', 'time_1d']].to_xarray()
    # rename the columns to match the xarray dataset
    selector = selector.rename({'lat_005': 'lat', 'time_1d': 'time'})
    xco2mbl = xr.open_dataarray('../data/atmco2/xco2mbl-timeP7D_1D-lat25km.nc')
    matched_xco2 = xco2mbl.sel(**selector, method='nearest').to_series()
    
    df['xco2'] = matched_xco2

    return df 

dfs = []
for year in range(1982, 2022):
    df = pd.read_parquet(f'../data/SOCATv2024-1d_005deg-colloc-r20250224/SOCATv2024_1d_005deg_collocated_{year}-r20250224.pq')
    df.reset_index(inplace=True)
    df['year'] = year
    df = add_xco2(df)
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)
print(df.columns)

print(df.xco2.mean(), df.xco2.std())
print("mean before subtracting xco2")
print(df.fco2rec_uatm.mean(), df.fco2rec_uatm.std())
print("mean after subtracting xco2")
print((df['fco2rec_uatm'] - df['xco2']).mean(), (df['fco2rec_uatm'] - df['xco2']).std())

#add day_of_year column
df['day_of_year'] = df['time_1d'].dt.dayofyear
# remove entres with high ice concentration
df = df[df.ice_cci < 0.8]
# renane lon and lat columns
df = df.rename(columns={'lon_005':'lon', 'lat_005': 'lat'})
df = prep_df(df, logger=logging)[0]



test_months = pd.date_range('1982-01', '2022-01', freq='7MS').values.astype('datetime64[M]')
months = df.time_1d.values.astype('datetime64[M]')
mask_test = np.isin(months, test_months)

df_train = df[~mask_test]
df_val = df[mask_test]

# logging.info("select measurements from the northern hemisphere")
# df_train = df_train[df_train['is_north']]
# df_val = df_val[df_val['is_north']]

# drop nan rows
predictors = ['sst_cci', 'sss_cci', 'chl_globcolour', 'ssh_sla', 
              'mld_dens_soda', 'xco2', 'sin_day_of_year', 'cos_day_of_year', 
              'sin_lat', 'sin_lon_cos_lat', 'cos_lon_cos_lat']
target = 'fco2rec_uatm'
df_train = df_train[[target] + predictors].dropna()
df_val = df_val[[target] + predictors].dropna()



train_ds = df_train.values[:, :, np.newaxis]
val_ds = df_val.values[:, :, np.newaxis]
# shuffle the data
np.random.shuffle(train_ds)

print(f"train_ds shape: {train_ds.shape}")
print(f"val_ds shape: {val_ds.shape}")
logging.info(f"train_ds shape: {train_ds.shape}")
logging.info(f"val_ds shape: {val_ds.shape}")


# normalize the data
mode = 'mean_std'
train_stats = get_stats(train_ds, logger=logging)
train_ds, val_ds = normalize_dss([train_ds, val_ds], train_stats, mode, logger=logging)

train_dataset = TensorDataset(torch.tensor(train_ds))
val_dataset = TensorDataset(torch.tensor(val_ds))
train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
model_params = {
    "input_dim": train_ds.shape[1] - 1,
    "hidden_dim": 128,
    "output_dim": 1,
}

model = MLPModel(**model_params)
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




model, train_losses, val_losses = train_mean_estimator(model,
                                                       num_epochs=num_epochs, 
                                                       old_epoch=epoch,
                                                       optimizer=optimizer, 
                                                       lr_scheduler=lr_scheduler,
                                                       train_dataloader=train_dataloader,
                                                       val_dataloader=val_dataloader,
                                                       save_model_path=save_dir,
                                                       rmse_const=train_stats['stds'][0],
                                                       is_sequence=False)


train_losses_old += train_losses
val_losses_old += val_losses

save_losses_and_png(train_losses_old, val_losses_old, save_dir)

logging.info("Completed training")
logging.info("Training losses: %s", train_losses)
logging.info("Validation losses: %s", val_losses)





