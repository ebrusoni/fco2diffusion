import sys
from pathlib import Path

src_path = Path.home() / "work" / "fco2diffusion" / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
    
renko=False
if renko:
    DATA_PATH = '/home/jovyan/work/datapolybox/'
else:
    DATA_PATH = '../data/training_data/'

import pandas as pd 
import numpy as np
import xarray as xr
import torch
# import dataloader and tensordataset
from torch.utils.data import TensorDataset, DataLoader
import torch.utils.data as data
import torch.optim as optim
import torch.nn as nn

import json
import logging
from tqdm import tqdm

from fco2models.utraining import prep_df, normalize_dss
from fco2models.umeanest import train_mean_estimator



np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

lr = 1e-2
batch_size = 2048
save_dir = f'../models/sota/'
 #create directory if it does not exist
import os
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

logging.basicConfig(
    filename=save_dir+'training.log',
    filemode='a',
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
    )

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


df = prep_df(df, logger=logging)[0]

# get unique value fo expocode column
expocodes = df['expocode'].unique()

# select 0.1 of expecodes for testing
expocode_test = np.random.choice(expocodes, size=int(0.1*len(expocodes)), replace=False)
# select the rows with expocode in expocode_test
df_test = df[df['expocode'].isin(expocode_test)]
# select the rows with expocode not in expocode_test
df_train = df[~df['expocode'].isin(expocode_test)]

# drop nan rows
predictors = ['sst_cci', 'sss_cci', 'chl_globcolour', 'ssh_sla', 
              'mld_dens_soda', 'xco2', 'sin_lat', 'sin_lon_cos_lat', 'cos_lon_cos_lat',
              'sin_day_of_year', 'cos_day_of_year']
target = 'fco2rec_uatm'
df_train = df_train[[target] + predictors].dropna()
df_test = df_test[[target] + predictors].dropna()

train_ds = df_train.values[:, :, np.newaxis]
val_ds = df_test.values[:, :, np.newaxis]
# shuffle the data
np.random.shuffle(train_ds)

print(f"train_ds shape: {train_ds.shape}")
print(f"val_ds shape: {val_ds.shape}")
logging.info(f"train_ds shape: {train_ds.shape}")
logging.info(f"val_ds shape: {val_ds.shape}")

print(f"train_ds shape: {train_ds.shape}")
print(f"val_ds shape: {val_ds.shape}")
logging.info(f"train_ds shape: {train_ds.shape}")
logging.info(f"val_ds shape: {val_ds.shape}")


# normalize the data
mode = 'mean_std'
train_ds, rest_ds, train_means, train_stds, train_mins, train_maxs = normalize_dss([train_ds, val_ds], mode=mode, logger=logging)
vald_ds = rest_ds[0]

# remove the 3rd axis
train_ds = train_ds[:, :, 0]
val_ds = val_ds[:, :, 0]
train_dataset = TensorDataset(torch.tensor(train_ds))
val_dataset = TensorDataset(torch.tensor(val_ds))
train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# define simple mlp model
class  MLPModel(nn.Module):
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
        x = self.fc1(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.batchnorm3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return (x, None)
    
model_params = {
    "input_dim": train_ds.shape[1] - 1,
    "hidden_dim": 256,
    "output_dim": 1,
}
model = MLPModel(**model_params)
def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Number of trainable parameters: {count_trainable_parameters(model)}")
num_epochs = 100
optimizer = optim.Adam(model.parameters(), lr=lr)

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
    "lr_params": None,
    "noise_params": None,
    "batch_size": batch_size,
    "num_epochs": num_epochs,
    "lr": lr,
    "optimizer": optimizer.__class__.__name__,
    "predictors": predictors,
    "train_means": train_means,
    "train_stds": train_stds,
    "train_mins": train_mins,
    "train_maxs": train_maxs,
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
                                                       lr_scheduler=None,
                                                       train_dataloader=train_dataloader,
                                                       val_dataloader=val_dataloader,
                                                       save_model_path=save_dir)

    
train_losses_old += train_losses
val_losses_old += val_losses
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

logging.info("Completed training")
logging.info("Training losses: %s", train_losses)
logging.info("Validation losses: %s", val_losses)



