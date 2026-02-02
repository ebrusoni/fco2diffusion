from utils import add_src_and_logger
models = ["UNet1D", "Guidance-UNet1D", "UNet2D", "UNet2DL"]
dm = models[0]
save_dir = f'../models/{dm}/'
is_renkolab = True
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
from fco2models.utraining import get_segments_random, get_segments, make_monthly_split, get_stats_df, get_context_mask, replace_with_cruise_data, perturb_fco2
from fco2models.models import MLP, UNet2DModelWrapper, ConvNet, UNet2DModelWrapper, ClassEmbedding, Unet2DClassifierFreeModel, UNet2DShipMix, TSEncoderWrapper, DiffusionEnsemble, UNet1DModelWrapper, Unet1DClassifierFreeModel
from fco2models.umeanest import train_mean_estimator

# fix random seed for reproducibility
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

ds_version = ""
lr = 1e-3
batch_size = 128

logger = logging.getLogger(__name__)
logger.info("------------ Starting training ------------------")
logger.info("Training with larger random dataset")

df = pd.read_parquet(DATA_PATH + f"SOCAT_1982_2021_grouped_colloc_augm_binned-v{ds_version}.pq")
df = prep_df(df, bound=True)[0]

print(df.fco2rec_uatm.max(), df.fco2rec_uatm.min())
print(f"dataset shape: {df.shape}")

mask_train, mask_val, mask_test = make_monthly_split(df) # split expocodes by month as in gregor2024
df_train = df[df.expocode.map(mask_train)]
df_val = df[df.expocode.map(mask_val)]
logger.info(f"Validation seamask ratio: {df_val.seamask.sum()/df_val.shape[0]}")
df_test = df[df.expocode.map(mask_test)]
logger.info(f"Shapes - Train: {df_train.shape}, Val: {df_val.shape}, Test: {df_test.shape}")
assert df_val.expocode.isin(df_test.expocode).sum() == 0, "expocode ids overlap between validation and test sets"
assert df_train.expocode.isin(df_val.expocode).sum() == 0, "expocode ids overlap between train and validation sets"
assert df_train.expocode.isin(df_test.expocode).sum() == 0, "expocode ids overlap between train and test sets"
logger.info(f"training dataset shape: {df_train.shape}")
logger.info(f"validation dataset shape: {df_val.shape}")
logger.info(f"test dataset shape: {df_test.shape}")


target = "fco2rec_uatm"
predictors = ['sst_anom', 'sss_anom', 'chl_anom', 'ssh_anom', 'mld_anom',
              'sst_clim', 'sss_clim', 'chl_clim', 'ssh_clim', 'mld_clim',
              'xco2', 'co2_clim8d']
positional_encoding = ['sin_day_of_year', 'cos_day_of_year', 'sin_lat', 'sin_lon_cos_lat', 'cos_lon_cos_lat']
model_inputs = [target] + predictors + positional_encoding
socat_data =  []#["sst_deg_c", "sal"]
cols = model_inputs + socat_data

# divide the cruise tracks (indexed by expocodes) in segments of length 64
train_stats = get_stats_df(df_train, model_inputs + socat_data) # gets means, stds, maxs and mins for every column, used for normalizing
segment_df_train = df_train.groupby("expocode").apply(
    lambda x: get_segments_random(x, cols, n=4),
    include_groups=False,
)
train_socat_ds = np.concatenate(segment_df_train.values, axis=0)
train_ds = train_socat_ds[:, :, :] # here we remove the socat ship measurements when available
#socat_ds = train_socat_ds[:, -2:, :]
# convert to kelvin
#socat_ds[:, 0, :] += 273.15
#train_ds = replace_with_cruise_data(train_ds, socat_ds, prob=0.5)

segment_df_val = df_val.groupby("expocode").apply(
    lambda x: get_segments(x, model_inputs),
    include_groups=False,
)
val_ds = np.concatenate(segment_df_val.values, axis=0)
segment_df_test = df_test.groupby("expocode").apply(
    lambda x: get_segments(x, model_inputs),
    include_groups=False,
)
test_ds = np.concatenate(segment_df_test.values, axis=0)

print(f"train_ds shape: {train_ds.shape}, val_ds shape: {val_ds.shape}, test_ds shape: {test_ds.shape}")
train_context_mask, val_context_mask, test_context_mask = get_context_mask([train_ds, val_ds, test_ds]) # mask to filter out invalid context containing Nans
train_ds = train_ds[train_context_mask]
#socat_ds = socat_ds[train_context_mask]
val_ds = val_ds[val_context_mask]
test_ds = test_ds[test_context_mask]
print(f"removing context nans")
print(f"train_ds shape: {train_ds.shape}, val_ds shape: {val_ds.shape}, test_ds shape: {test_ds.shape}")

mode = 'mean_std' # z-normalization
#train_ds = np.concatenate([train_ds, socat_ds], axis=1)
train_ds, val_ds, test_ds = normalize_dss([train_ds, val_ds, test_ds], train_stats, mode, ignore=[]) # normalize

# count nans in first column of the dataset
print(f"train_ds shape: {train_ds.shape}, val_ds shape: {val_ds.shape}, test_ds shape: {test_ds.shape}")
logger.info(f"train_ds shape: {train_ds.shape}, val_ds shape: {val_ds.shape}, test_ds shape: {test_ds.shape}")
# print mins and maxs of the data
for i in range(train_ds.shape[1]):
    print(f"train_ds {i} min: {np.nanmin(train_ds[:, i, :])}, max: {np.nanmax(train_ds[:, i, :])}")

print(f"train_ds shape: {train_ds.shape}")
print(f"val_ds shape: {val_ds.shape}")

# timestep_dim = 16
if dm == "UNet1D" or dm == "Guidance-UNet1D":
    layers_per_block = 2
    down_block_types = ("DownResnetBlock1D", "DownResnetBlock1D")
    up_block_types = ("UpResnetBlock1D", "UpResnetBlock1D")
    model_params = {
         "sample_size": 64,
         "in_channels": train_ds.shape[1] + 1, # add nanmask input channel
         "out_channels": 8, # required otherwise code breaks
         "layers_per_block": layers_per_block,
         "block_out_channels": (16, 32),
         "down_block_types": down_block_types,
         "up_block_types": up_block_types,
         "norm_num_groups": None,
         "use_timestep_embedding": True,
         "act_fn": "relu"
    }
    logger.info("Using UNet1DModel")

    if dm == "UNet1D":
        # epoch 200 is the best one usually
        model =  UNet1DModelWrapper(**model_params)
    else:
        model_params = {
            "unet_config": model_params,
            "keep_channels": [18], # do not drop nan-mask channel
            "num_channels": 19
            }
        model = Unet1DClassifierFreeModel(**model_params)

if dm == "UNet2D" or dm == "UNet2DL":
    # epochs 200 work well for both models
    layers_per_block = 2
    down_block_types = ('DownBlock2D', 'DownBlock2D')
    up_block_types = ('UpBlock2D', 'UpBlock2D')
    model_params = {
       # "sample_size": (14, 64),
       "in_channels": 1,
       "out_channels": 1,
       "layers_per_block": layers_per_block,
       "block_out_channels": (16, 32) if dm == "UNet2DL" else (8, 16),
       "down_block_types": down_block_types,
       "up_block_types": up_block_types,
       "norm_num_groups": 16 if dm == "UNet2DL" else 8,
    #    "class_embed_type": "Identity",
    #    "num_class_embeds": None, 
    }
    model = UNet2DModelWrapper(**model_params)

num_epochs = 300
timesteps = 1000

#model_params= {
#   "feat_dim": len(model_inputs) + 1, # add mask and timestep channels
#   "max_len": 65,
#   "d_model": 32,
#   "n_heads": 4,
#   "num_layers": 4,
#   "dim_feedforward": 80,
#   "num_classes":64,
#   "activation": "relu",
#   "pos_encoding": "learnable"
#}

def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {count_trainable_parameters(model)}")

model.to('cuda')
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

checkpoint_path = None #"../models/ts_encoder/final_model_e_400.pt"
if checkpoint_path is not None:
    model, optimizer, lr_scheduler, epoch, train_losses_old, val_losses_old = load_checkpoint(checkpoint_path, model, optimizer, lr_scheduler) 
else:
    epoch = 0
    train_losses_old = []
    val_losses_old = [] 

noise_params = {
    "num_train_timesteps": timesteps,
    "beta_schedule": 'squaredcos_cap_v2',
    "clip_sample_range": 4.0
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
    "predictors": predictors + positional_encoding,
    "train_means": train_stats['means'],
    "train_stds": train_stats['stds'],
    "train_mins": train_stats['mins'],
    "train_maxs": train_stats['maxs'],
    "mode": mode,
    }

logger.info("All parameters: %s", param_dict)
# save the model parameters to a json file
with open(save_dir +'hyperparameters.json', 'w') as f:
    param_dict = json.dumps(param_dict, indent=4)
    f.write(param_dict)

model, train_losses, val_losses = train_diffusion(model,
                                                  num_epochs=num_epochs,
                                                  old_epoch=epoch, 
                                                  optimizer=optimizer, 
                                                  lr_scheduler=lr_scheduler, 
                                                  noise_scheduler=noise_scheduler, 
                                                  train_dataloader=train_dataloader,
                                                  val_dataloader=val_dataloader,
                                                  save_model_path=save_dir,
                                                  )

    
save_losses_and_png_diffusion(
    train_losses_old + train_losses, 
    val_losses_old+ val_losses, 
    save_dir,
    noise_scheduler.config.num_train_timesteps,
    )

logger.info("Completed training")
logger.info("Training losses: %s", train_losses)
logger.info("Validation losses: %s", val_losses)