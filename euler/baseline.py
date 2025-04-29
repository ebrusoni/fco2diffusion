from utils import add_src_and_logger
save_dir = f'../models/baseline/'
is_renkolab = False
DATA_PATH, logging = add_src_and_logger(is_renkolab, save_dir)

import json
import pandas as pd
import numpy as np

import torch
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import TensorDataset
# import wandb
from diffusers.optimization import get_cosine_schedule_with_warmup, get_constant_schedule
from diffusers import DDPMScheduler, UNet1DModel

from fco2models.utraining import prep_df, normalize_dss, prepare_segment_ds, save_losses_and_png, load_checkpoint, get_stats
from fco2models.models import MLP, UNet2DModelWrapper, ConvNet, UNet2DModelWrapper
from fco2models.umeanest import train_mean_estimator

# fix random seed for reproducibility
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

lr = 4e-5
batch_size = 128
num_epochs = 40


logging.info("------------ Starting training ------------------")
logging.info("Training with larger random dataset")

df_train = pd.read_parquet(DATA_PATH+'traindf_100km_xco2.pq')
df_val = pd.read_parquet(DATA_PATH+'valdf_100km_xco2.pq')
df_2021 = pd.read_parquet(DATA_PATH+'df_100km_xco2_2021.pq')

df_train, df_val, df_2021 = prep_df([df_train, df_val, df_2021], index = ['segment', 'bin'], logger=logging)

predictors = ['sst_cci', 'sss_cci', 'chl_globcolour', 'ssh_sla', 'mld_dens_soda', 'xco2']
positional_encoding = ['day_of_year', 'lat', 'lon']
predictors += positional_encoding
train_ds, val_ds = prepare_segment_ds([df_train, df_val], predictors, with_mask=True, logging=logging)
mask_ix = len(predictors) + 1
# val_ds_2021 = prepare_segment_ds(df_2021, predictors, logging=logging)
# val_ds = np.concatenate([val_ds, val_ds_2021], axis = 0)

print(f"train_ds shape: {train_ds.shape}")
print(f"val_ds shape: {val_ds.shape}")
logging.info(f"train_ds shape: {train_ds.shape}")
logging.info(f"val_ds shape: {val_ds.shape}")


mode = 'min_max'
train_stats = get_stats(train_ds, logger=logging)
train_ds, val_ds = normalize_dss([train_ds, val_ds], train_stats, mode, ignore=[7, 8, 9] + [mask_ix], logger=logging)

# print mins and maxs of the data
for i in range(train_ds.shape[1]):
    print(f"train_ds {i} min: {np.nanmin(train_ds[:, i, :])}, max: {np.nanmax(train_ds[:, i, :])}")
    print(f"val_ds {i} min: {np.nanmin(val_ds[:, i, :])}, max: {np.nanmax(val_ds[:, i, :])}")

train_dataset = TensorDataset(torch.tensor(train_ds))
val_dataset = TensorDataset(torch.tensor(val_ds))
train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# layers_per_block = 2
# down_block_types = ('DownBlock2D', 'DownBlock2D')
# up_block_types = ('UpBlock2D', 'UpBlock2D')
# model_params = {
#     "sample_size": (10, 64),
#     "in_channels": 1,
#     "out_channels": 1,
#     "layers_per_block": layers_per_block,
#     "block_out_channels": (32, 64),
#     "down_block_types": down_block_types,
#     "up_block_types": up_block_types,
#     "norm_num_groups": 32
# }

# model = UNet2DModelWrapper(**model_params)

layers_per_block = 2
down_block_types = ('DownBlock1D', 'DownBlock1D')
up_block_types = ('UpBlock1D', 'UpBlock1D')
model_params = {
    "sample_size": 64,
    "in_channels": (len(predictors) + 1),
    "out_channels": 1,
    "layers_per_block": layers_per_block,
    "block_out_channels": (32, 64),
    "down_block_types": down_block_types,
    "up_block_types": up_block_types,
    "norm_num_groups": 32
}
model = UNet1DModel(**model_params)

# model_params = { 
#     "channels_in": train_ds.shape[1] - 1
#     }
# model = ConvNet(**model_params)

def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {count_trainable_parameters(model)}")

optimizer = optim.Adam(model.parameters(), lr=lr)

lr_params = {
    "num_warmup_steps": 0.05 * num_epochs * len(train_dataloader), 
    "num_training_steps": num_epochs * len(train_dataloader)
    }
lr_scheduler = get_constant_schedule(optimizer)

# save all hyperparameters to a json file
param_dict = {
    "model_params": model_params,
    "lr_params": lr_params,
    "optimizer": optimizer.__class__.__name__,
    "noise_params": None,
    "batch_size": batch_size,
    "num_epochs": num_epochs,
    "lr": lr,
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


checkpoint_path = None
if checkpoint_path is not None:
    model, optimizer, lr_scheduler, epoch, train_losses_old, val_losses_old = load_checkpoint(checkpoint_path, model, optimizer, lr_scheduler) 
else:
    epoch = 0
    train_losses_old = []
    val_losses_old = []    

rmse_const = train_stats['stds'] if mode == 'mean_std' else (train_stats['maxs'] - train_stats['mins']) / 2.0
model, train_losses, val_losses = train_mean_estimator(model,
                                                       num_epochs=num_epochs, 
                                                       old_epoch=epoch,
                                                       optimizer=optimizer, 
                                                       lr_scheduler=lr_scheduler,
                                                       train_dataloader=train_dataloader,
                                                       val_dataloader=val_dataloader,
                                                       save_model_path=save_dir, 
                                                       rmse_const=rmse_const)

    
save_losses_and_png(
    train_losses_old + train_losses, 
    val_losses_old + val_losses, 
    save_dir
    )

logging.info("Completed training")
logging.info("Training losses: %s", train_losses)
logging.info("Validation losses: %s", val_losses)