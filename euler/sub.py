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
from fco2dataset.ucruise import filter_nans
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
# import wandb
from tqdm import tqdm
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers import DDPMScheduler, UNet1DModel
from fco2models.utraining import train_diffusion, prep_data
from torch.utils.data import TensorDataset
import json
from fco2models.models import MLP, UNet2DModelWrapper
from scipy.ndimage import gaussian_filter1d 
import logging

# fix random seed for reproducibility
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

lr = 5e-4
batch_size = 128
save_dir = f'../models/unet2d_{batch_size}_{lr}/'
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

def load_checkpoint(path, model, optimizer, scheduler):
    logging.info(f"Loading checkpoint from {path}")
    # Load checkpoint
    checkpoint = torch.load(path)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    epoch = checkpoint['epoch'] + 1  # Continue from next epoch
    train_losses = checkpoint['train_losses']
    val_losses = checkpoint['val_losses']
    logging.info(f'starting from epoch {epoch + 1}')
    return model, optimizer, scheduler, epoch, train_losses, val_losses



# load data and filter out nans in the context variables
# df = pd.read_parquet('../data/training_data/traindf_100km.pq')
logging.info("Training with larger random dataset")
df_train = pd.read_parquet(DATA_PATH+'traindf_100km_random_reshaped.pq')
df_val = pd.read_parquet(DATA_PATH+'valdf_100km_random_reshaped.pq')
df_2021 = pd.read_parquet(DATA_PATH+'df_100km_random_reshaped_2021.pq')
logging.info("Using already separated train and validation datasets")
#df_train = pd.read_parquet('/home/jovyan/work/datapolybox/traindf_100km_random.pq')
#df_val = pd.read_parquet('/home/jovyan/work/datapolybox/valdf_100km_random.pq')
predictors = ['sst_cci', 'sss_cci', 'chl_globcolour']
train_ds = prep_data(df_train, predictors, logging=logging)
val_ds = prep_data(df_val, predictors, logging=logging)
val_ds_2021 = prep_data(df_2021, predictors, logging=logging)
val_ds = np.concatenate([val_ds, val_ds_2021], axis = 0)


print(f"train_ds shape: {train_ds.shape}")
print(f"val_ds shape: {val_ds.shape}")
logging.info(f"train_ds shape: {train_ds.shape}")
logging.info(f"val_ds shape: {val_ds.shape}")


def normalize(x, mean_max, std_min, mode):
    """normalize the data"""
    if mode == 'mean_std':
        x = (x - mean_max) / std_min
    elif mode == 'min_max':
        # normalize -1 to 1
        x = 2 * (x - std_min) / (mean_max - std_min) - 1
    return x

train_means = []
train_stds = []
train_mins = []
train_maxs = []
for i in range(train_ds.shape[1]):
    train_means.append(np.nanmean(train_ds[:, i, :]))
    train_stds.append(np.nanstd(train_ds[:, i, :]))
    train_mins.append(np.nanmin(train_ds[:, i, :]))
    train_maxs.append(np.nanmax(train_ds[:, i, :]))

mode = 'min_max'
logging.info(f"Normalizing data using {mode} normalization")
# normalize the data
for i in range(train_ds.shape[1]):
    train_ds[:, i, :] = normalize(train_ds[:, i, :], train_maxs[i], train_mins[i], mode)
    val_ds[:, i, :] = normalize(val_ds[:, i, :], train_maxs[i], train_mins[i], mode)


# print validation data mean and std

# print mins and maxs of the data
for i in range(train_ds.shape[1]):
    print(f"train_ds {i} min: {np.nanmin(train_ds[:, i, :])}, max: {np.nanmax(train_ds[:, i, :])}")
    print(f"val_ds {i} min: {np.nanmin(val_ds[:, i, :])}, max: {np.nanmax(val_ds[:, i, :])}")

# save the training and validation data
np.save('../train_ds.npy', train_ds)
np.save('../val_ds.npy', val_ds)

print(f"train_ds shape: {train_ds.shape}")
print(f"val_ds shape: {val_ds.shape}")

# timestep_dim = 16
# layers_per_block = 3
# down_block_types = ('DownBlock1DNoSkip', 'AttnDownBlock1D')
# up_block_types = ('AttnUpBlock1D', 'UpBlock1DNoSkip')
# model_params = {
#     "sample_size": 64,
#     "in_channels": timestep_dim + train_ds.shape[1] + 1,
#     "out_channels": 1,
#     "layers_per_block": layers_per_block,
#     "block_out_channels": (64, 128),
#     "down_block_types": down_block_types,
#     "up_block_types": up_block_types
# }
# logging.info("Using UNet1DModel")

# model = UNet1DModel(**model_params)

layers_per_block = 2
down_block_types = ('DownBlock2D', 'AttnDownBlock2D')
up_block_types = ('AttnUpBlock2D', 'UpBlock2D')
model_params = {
    "sample_size": (4, 64),
    "in_channels": 1,
    "out_channels": 1,
    "layers_per_block": layers_per_block,
    "block_out_channels": (32, 64),
    "down_block_types": down_block_types,
    "up_block_types": up_block_types,
    "norm_num_groups": 16
}

model = UNet2DModelWrapper(**model_params)

# model_params = {
#     "input_dim": 64*(ds.shape[1] + 1),
#     "output_dim": 64,
#     "hidden_dims": [64*5, 64*3],
#     "dropout_prob": 0.0
#     }
timesteps = 1000
# model = MLP(**model_params, num_timesteps=timesteps)

num_epochs = 1

optimizer = optim.AdamW(model.parameters(), lr=lr)

train_dataset = TensorDataset(torch.tensor(train_ds))
val_dataset = TensorDataset(torch.tensor(val_ds))
train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

#checkpoint_path = ""
epoch = 0
train_losses_old = []
val_losses_old = []
#model, optimizer, scheduler, epoch, train_losses_old, val_losses_old = load_checkpoint(checkpoint_path, model, optimizer, lr_scheuduler)

lr_params = {
    "num_warmup_steps": 0.05 * num_epochs * len(train_dataloader), 
    "num_training_steps": num_epochs * len(train_dataloader)
    }
lr_scheduler = get_cosine_schedule_with_warmup(optimizer, **lr_params)
noise_params = {
    "num_train_timesteps": timesteps,
    "beta_schedule": 'squaredcos_cap_v2',
    "clip_sample_range": 1.0
    }
noise_scheduler = DDPMScheduler(**noise_params)
# save all hyperparameters to a json file
param_dict = {
    "model_params": model_params,
    "lr_params": lr_params,
    "noise_params": noise_params,
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


model, train_losses, val_losses = train_diffusion(model,
                                                  num_epochs=num_epochs - epoch, 
                                                  optimizer=optimizer, 
                                                  lr_scheduler=lr_scheduler, 
                                                  noise_scheduler=noise_scheduler, 
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
t_tot = noise_scheduler.config.num_train_timesteps
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

logging.info("Completed training")
logging.info("Training losses: %s", train_losses)
logging.info("Validation losses: %s", val_losses)