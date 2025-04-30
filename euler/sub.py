from utils import add_src_and_logger
lr = 5e-4
batch_size = 128
save_dir = f'../models/unet2d_{batch_size}_{lr}/'
is_renkolab = False
DATA_PATH, logging = add_src_and_logger(is_renkolab, save_dir)
    
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import torch.utils.data as data
import numpy as np
# import wandb
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers import DDPMScheduler, UNet1DModel
from fco2models.utraining import train_diffusion, load_checkpoint, prepare_segment_ds, normalize_dss, prep_df, save_losses_and_png_diffusion, get_stats
from torch.utils.data import TensorDataset
import json
from fco2models.models import MLP, UNet2DModelWrapper

# fix random seed for reproducibility
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

 #create directory if it does not exist

logging.info("------------ Starting training ------------------")



# load data and filter out nans in the context variables
# df = pd.read_parquet('../data/training_data/traindf_100km.pq')
logging.info("Training with larger random dataset")
logging.info("Using already separated train and validation datasets")
df_train = pd.read_parquet(DATA_PATH+'traindf_100km_xco2.pq')
df_val = pd.read_parquet(DATA_PATH+'valdf_100km_xco2.pq')
df_2021 = pd.read_parquet(DATA_PATH+'df_100km_xco2_2021.pq')

df_train, df_val, df_2021 = prep_df([df_train, df_val, df_2021], index = ['segment', 'bin'], logger=logging, bound=True)

predictors = ['sst_cci', 'sss_cci', 'chl_globcolour', 'ssh_sla', 'mld_dens_soda', 'xco2']
positional_encoding = ['sin_day_of_year', 'cos_day_of_year', 'sin_lat', 'sin_lon_cos_lat', 'cos_lon_cos_lat']
all_cols = predictors + positional_encoding
train_ds, val_ds, val_ds_2021  = prepare_segment_ds([df_train, df_val, df_2021], all_cols, logging=logging)
val_ds = np.concatenate([val_ds, val_ds_2021], axis = 0)

print(f"train_ds shape: {train_ds.shape}")
print(f"val_ds shape: {val_ds.shape}")
logging.info(f"train_ds shape: {train_ds.shape}")
logging.info(f"val_ds shape: {val_ds.shape}")


# normalize the data
mode = 'min_max'
train_stats = get_stats(train_ds, logger=logging)
train_ds, val_ds = normalize_dss([train_ds, val_ds], train_stats, mode, ignore=[7,8,9,10,11],  logger=logging)

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
down_block_types = ('DownBlock2D', 'AttnDownBlock2D')
up_block_types = ('AttnUpBlock2D', 'UpBlock2D')
model_params = {
    "sample_size": (13, 64),
    "in_channels": 1,
    "out_channels": 1,
    "layers_per_block": layers_per_block,
    "block_out_channels": (16, 32),
    "down_block_types": down_block_types,
    "up_block_types": up_block_types,
    "norm_num_groups": 8,
    #"num_class_embeds": len(positional_encoding)
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


model, train_losses, val_losses = train_diffusion(model,
                                                  num_epochs=num_epochs - epoch, 
                                                  optimizer=optimizer, 
                                                  lr_scheduler=lr_scheduler, 
                                                  noise_scheduler=noise_scheduler, 
                                                  train_dataloader=train_dataloader,
                                                  val_dataloader=val_dataloader,
                                                  save_model_path=save_dir,
                                                  pos_encodings_start=None,
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