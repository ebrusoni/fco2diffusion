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

from fco2models.utraining import prep_df, normalize_dss, load_checkpoint, train_diffusion, save_losses_and_png_diffusion
from fco2models.utraining import get_segments_random, get_segments, make_monthly_split, get_stats_df, get_context_mask
from fco2models.models import MLP, UNet2DModelWrapper, ConvNet, UNet2DModelWrapper, ClassEmbedding
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

df = pd.read_parquet(DATA_PATH + "SOCAT_1982_2021_grouped_colloc_augm_bin.pq")
df = prep_df(df, logger=logging)[0]
print(df.fco2rec_uatm.max(), df.fco2rec_uatm.min())

mask_train, mask_val, mask_test = make_monthly_split(df)
df_train = df[df.expocode.map(mask_train)]
df_val = df[df.expocode.map(mask_val)]
df_test = df[df.expocode.map(mask_test)]
print(df_train.fco2rec_uatm.max(), df_train.fco2rec_uatm.min())


target = "fco2rec_uatm"
predictors = ['sst_cci', 'sss_cci', 'chl_globcolour', 'ssh_sla', 'mld_dens_soda', 'xco2']
positional_encoding = ['sin_day_of_year', 'cos_day_of_year', 'sin_lat', 'sin_lon_cos_lat', 'cos_lon_cos_lat']
all_cols = predictors + positional_encoding
cols = [target] + all_cols

train_stats = get_stats_df(df_train, cols, logger=logging)

segment_df_train = df_train.groupby("expocode").apply(
    lambda x: get_segments_random(x, cols, n=3),
    include_groups=False,
)
train_ds = np.concatenate(segment_df_train.values, axis=0)
segment_df_val = df_val.groupby("expocode").apply(
    lambda x: get_segments(x, cols),
    include_groups=False,
)
val_ds = np.concatenate(segment_df_val.values, axis=0)
segment_df_test = df_test.groupby("expocode").apply(
    lambda x: get_segments(x, cols),
    include_groups=False,
)
test_ds = np.concatenate(segment_df_test.values, axis=0)

print(f"train_ds shape: {train_ds.shape}, val_ds shape: {val_ds.shape}, test_ds shape: {test_ds.shape}")
train_context_mask, val_context_mask, test_context_mask = get_context_mask([train_ds, val_ds, test_ds], logger=logging)
train_ds = train_ds[train_context_mask]
val_ds = val_ds[val_context_mask]
test_ds = test_ds[test_context_mask]
print(f"removing context nans")
print(f"train_ds shape: {train_ds.shape}, val_ds shape: {val_ds.shape}, test_ds shape: {test_ds.shape}")

mode = 'mean_std'
train_ds, val_ds, test_ds = normalize_dss([train_ds, val_ds, test_ds], train_stats, mode, ignore=[], logger=logging)

# count nans in first column of the dataset
print(f"train_ds shape: {train_ds.shape}, val_ds shape: {val_ds.shape}, test_ds shape: {test_ds.shape}")
logging.info(f"train_ds shape: {train_ds.shape}, val_ds shape: {val_ds.shape}, test_ds shape: {test_ds.shape}")

# print mins and maxs of the data
for i in range(train_ds.shape[1]):
    print(f"train_ds {i} min: {np.nanmin(train_ds[:, i, :])}, max: {np.nanmax(train_ds[:, i, :])}")
    print(f"val_ds {i} min: {np.nanmin(val_ds[:, i, :])}, max: {np.nanmax(val_ds[:, i, :])}")

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
num_epochs = 1
timesteps = 1000

layers_per_block = 2
down_block_types = ('DownBlock2D', 'DownBlock2D')
up_block_types = ('UpBlock2D', 'UpBlock2D')
model_params = {
    "sample_size": (13, 64),
    "in_channels": 1,
    "out_channels": 1,
    "layers_per_block": layers_per_block,
    "block_out_channels": (16, 32),
    "down_block_types": down_block_types,
    "up_block_types": up_block_types,
    "norm_num_groups": 16,
    # "class_embed_type": "Identity",
    # "num_class_embeds": None, 
}

model = UNet2DModelWrapper(**model_params)
def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {count_trainable_parameters(model)}")

# model_params = {
#     "input_dim": 64*(ds.shape[1] + 1),
#     "output_dim": 64,
#     "hidden_dims": [64*5, 64*3],
#     "dropout_prob": 0.0
#     }
# model = MLP(**model_params, num_timesteps=timesteps)

optimizer = optim.AdamW(model.parameters(), lr=lr)


train_dataset = TensorDataset(torch.tensor(train_ds))
val_dataset = TensorDataset(torch.tensor(val_ds))
train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

lr_params = {
    "num_warmup_steps": 0.05 * num_epochs * len(train_dataloader), 
    "num_training_steps": num_epochs * len(train_dataloader)
    }
lr_scheduler = get_cosine_schedule_with_warmup(optimizer, **lr_params)

checkpoint_path = None
if checkpoint_path is not None:
    model, optimizer, lr_scheduler, epoch, train_losses_old, val_losses_old = load_checkpoint(checkpoint_path, model, optimizer, lr_scheduler) 
else:
    epoch = 0
    train_losses_old = []
    val_losses_old = [] 

noise_params = {
    "num_train_timesteps": timesteps,
    "beta_schedule": 'squaredcos_cap_v2',
    "clip_sample_range": 2.0
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
    "predictors": all_cols,
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

# class_embedder = ClassEmbedding(dim_classes=len(positional_encoding), 
#                                 output_dim=16*4, 
#                                 num_classes=[100]*len(positional_encoding)
#                                 )
model, train_losses, val_losses = train_diffusion(model,
                                                  num_epochs=num_epochs - epoch, 
                                                  optimizer=optimizer, 
                                                  lr_scheduler=lr_scheduler, 
                                                  noise_scheduler=noise_scheduler, 
                                                  train_dataloader=train_dataloader,
                                                  val_dataloader=val_dataloader,
                                                  save_model_path=save_dir,
                                                  #pos_encodings_start=len(predictors) + 1,
                                                  #class_embedder=class_embedder,
                                                  )

    
save_losses_and_png_diffusion(
    train_losses_old + train_losses, 
    val_losses_old+ val_losses, 
    save_dir,
    noise_scheduler.config.num_train_timesteps,
    )

logging.info("Completed training")
logging.info("Training losses: %s", train_losses)
logging.info("Validation losses: %s", val_losses)