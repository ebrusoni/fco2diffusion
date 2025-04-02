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
from fco2models.utraining import train_diffusion
from torch.utils.data import TensorDataset
import json
from fco2models.models import MLP, UNet2DModelWrapper

# fix random seed for reproducibility
np.random.seed(0)
torch.manual_seed(0)

# load data and filter out nans in the context variables
df = pd.read_parquet('../data/training_data/traindf_100km.pq')
col_map = dict(zip(df.columns, range(len(df.columns))))
print(col_map)
ds = np.load('../data/training_data/trainds_100km.npy')
y = ds[0]
X, y = filter_nans(ds[:, :, :-1], y[:, :-1], ['sst_cci', 'sss_cci', 'chl_globcolour', 'mld_dens_soda'], col_map)
print(X.shape, y.shape)


assert np.apply_along_axis(lambda x: np.isnan(x).all(), 1, y).sum() == 0

print(X.shape, y[np.newaxis].shape)
assert np.isnan(X).sum() == 0
n_samples = X.shape[1]
n_dims = X.shape[2]
ds = np.zeros((n_samples, X.shape[0] + 1, n_dims))

ds[:, 0, :] = y
for i in range(X.shape[0]):
    ds[:, i + 1, :] = X[i]


# shuffle
np.random.shuffle(ds)
# split into training and validation
train_ds = ds[:int(0.9 * n_samples)]
val_ds = ds[int(0.9 * n_samples):]

def normalize(x, mean, std):
    """normalize the data"""
    x = (x - mean) / std
    return x

train_means = []
train_stds = []
for i in range(X.shape[0] + 1):
    train_means.append(np.nanmean(train_ds[:, i, :]))
    train_stds.append(np.nanstd(train_ds[:, i, :]))

train_means = np.array(train_means)
train_stds = np.array(train_stds)

# normalize training data
train_ds[:, 0, :] = normalize(train_ds[:, 0, :], train_means[0], train_stds[0])
for i in range(X.shape[0]):
    train_ds[:, i + 1, :] = normalize(train_ds[:, i + 1, :], train_means[i + 1], train_stds[i + 1])

# normalize validation data with training data mean and std
val_ds[:, 0, :] = normalize(val_ds[:, 0, :], train_means[0], train_stds[0])
for i in range(X.shape[0]):
    val_ds[:, i + 1, :] = normalize(val_ds[:, i + 1, :], train_means[i + 1], train_stds[i + 1])

# print validation data mean and std

print(f"Validation data mean fco2: {np.nanmean(val_ds[:, 0, :])}, std: {np.nanstd(val_ds[:, 0, :])}")
print(f"Validation data mean sst: {np.nanmean(val_ds[:, 1, :])}, std: {np.nanstd(val_ds[:, 1, :])}")
print(f"Validation data mean sss: {np.nanmean(val_ds[:, 2, :])}, std: {np.nanstd(val_ds[:, 2, :])}")
print(f"Validation data mean chl: {np.nanmean(val_ds[:, 3, :])}, std: {np.nanstd(val_ds[:, 3, :])}")
print(f"Validation data mean mld: {np.nanmean(val_ds[:, 4, :])}, std: {np.nanstd(val_ds[:, 4, :])}")

# save the training and validation data
np.save('../data/training_data/train_ds.npy', train_ds)
np.save('../data/training_data/val_ds.npy', val_ds)

print(f"train_ds shape: {train_ds.shape}")
print(f"val_ds shape: {val_ds.shape}")

# X = np.log(X + 2)
# ds[:, 0, :] = (y - np.nanmean(y)) / np.nanstd(y)
# for i in range(X.shape[0]):
#     ds[:, i + 1, :] = (X[i] - X[i].mean()) / X[i].std()

train_dataset = TensorDataset(torch.from_numpy(train_ds))
val_dataset = TensorDataset(torch.from_numpy(val_ds))


timestep_dim = 16
layers_per_block = 2
down_block_types = ('DownBlock1DNoSkip', 'AttnDownBlock1D')
up_block_types = ('AttnUpBlock1D', 'UpBlock1DNoSkip')
model_params = {
    "sample_size": 64,
    "in_channels": timestep_dim + train_ds.shape[1] + 1,
    "out_channels": 1,
    "layers_per_block": layers_per_block,
    "block_out_channels": (32, 64),
    "down_block_types": down_block_types,
    "up_block_types": up_block_types
}

model = UNet1DModel(**model_params)
# layers_per_block = 2
# down_block_types = ('DownBlock2D', 'AttnDownBlock2D')
# up_block_types = ('AttnUpBlock2D', 'UpBlock2D')
# model_params = {
#     "sample_size": (8, 64),
#     "in_channels": 1,
#     "out_channels": 1,
#     "layers_per_block": layers_per_block,
#     "block_out_channels": (32, 32),
#     "down_block_types": down_block_types,
#     "up_block_types": up_block_types,
#     "norm_num_groups": 16
# }
# model = UNet2DModelWrapper(**model_params)
# model_params = {
#     "input_dim": 64*(ds.shape[1] + 1),
#     "output_dim": 64,
#     "hidden_dims": [64*5, 64*3],
#     "dropout_prob": 0.0
#     }
timesteps = 1000
# model = MLP(**model_params, num_timesteps=timesteps)
batch_size = 128
num_epochs = 50
lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr)
train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
lr_params = {
    "num_warmup_steps": 500, 
    "num_training_steps": num_epochs * len(train_dataloader)
    }
lr_scheduler = get_cosine_schedule_with_warmup(optimizer, **lr_params)
noise_params = {
    "num_train_timesteps": timesteps,
    "beta_schedule": 'squaredcos_cap_v2',
    "clip_sample_range": 3.0
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
    "train_means": train_means.tolist(),
    "train_stds": train_stds.tolist(),}
save_dir = '../models/unet2d/'
with open(save_dir+'hyperparameters.json', 'w') as f:
    param_dict = json.dumps(param_dict, indent=4)
    f.write(param_dict)
model, train_losses, val_losses = train_diffusion(model,
                                                  num_epochs=num_epochs, 
                                                  optimizer=optimizer, 
                                                  lr_scheduler=lr_scheduler, 
                                                  noise_scheduler=noise_scheduler, 
                                                  train_dataloader=train_dataloader,
                                                  val_dataloader=val_dataloader,
                                                  save_model_path=save_dir)


# plot and save the training and validation losses
import matplotlib.pyplot as plt
plt.plot(train_losses, label='train')
plt.plot(val_losses, label='val')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Training and Validation Losses')
plt.savefig(save_dir + 'losses.png')
plt.show()