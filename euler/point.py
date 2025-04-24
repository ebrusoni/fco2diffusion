from utils import add_src_and_logger
save_dir = f'../models/point/'
is_renkolab = False
DATA_PATH, logging = add_src_and_logger(is_renkolab, save_dir)

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
# import wandb
from diffusers.optimization import get_cosine_schedule_with_warmup
from torch.utils.data import TensorDataset
import json
from fco2models.utraining import prep_df, normalize_dss, save_losses_and_png, load_checkpoint
from fco2models.umeanest import train_mean_estimator
# fix random seed for reproducibility
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)


lr = 5e-4
batch_size = 2048

logging.info("------------ Starting training ------------------")

    
logging.info("Training with larger random dataset")
df_train = pd.read_parquet(DATA_PATH+'traindf_100km_xco2.pq')
df_val = pd.read_parquet(DATA_PATH+'valdf_100km_xco2.pq')
df_2021 = pd.read_parquet(DATA_PATH+'df_100km_xco2_2021.pq')

df_train, df_val, df_2021 = prep_df([df_train, df_val, df_2021], logger=logging)

predictors = ['sst_cci', 'sss_cci', 'chl_globcolour', 'ssh_sla', 
              'mld_dens_soda', 'xco2', 'sin_lat', 'sin_lon_cos_lat', 'cos_lon_cos_lat',
              'sin_day_of_year', 'cos_day_of_year']
target = 'fco2rec_uatm'

df_train = df_train[[target] + predictors].dropna()
df_val = df_val[[target] + predictors].dropna()
df_2021 = df_2021[[target] + predictors].dropna()

# add last axis for compatibility with the helper functions
train_ds = df_train.values[:, :, np.newaxis]
val_ds = df_val.values[:, :, np.newaxis]


print(f"train_ds shape: {train_ds.shape}")
print(f"val_ds shape: {val_ds.shape}")
logging.info(f"train_ds shape: {train_ds.shape}")
logging.info(f"val_ds shape: {val_ds.shape}")

mode = 'mean_std'
train_ds, rest_ds, train_means, train_stds, train_mins, train_maxs = normalize_dss([train_ds, val_ds], mode, ignore=list(range(7, 13)), logger=logging)
vald_ds = rest_ds[0]

# print mins and maxs of the data
for i in range(train_ds.shape[1]):
    print(f"train_ds {i} min: {np.nanmin(train_ds[:, i, :])}, max: {np.nanmax(train_ds[:, i, :])}")
    print(f"val_ds {i} min: {np.nanmin(val_ds[:, i, :])}, max: {np.nanmax(val_ds[:, i, :])}")


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
        # remove last axis
        x = x.squeeze(-1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.batchnorm2(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.batchnorm3(x)
        x = self.fc3(x)
        return (x.unsqueeze(-1), None)
    
model_params = {
    "input_dim": train_ds.shape[1] - 1,
    "hidden_dim": 256,
    "output_dim": 1,
}
model = MLPModel(**model_params)
def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Number of trainable parameters: {count_trainable_parameters(model)}")
num_epochs = 10
optimizer = optim.AdamW(model.parameters(), lr=lr)

checkpoint_path = None
if checkpoint_path is not None:
    model, optimizer, lr_scheduler, epoch, train_losses_old, val_losses_old = load_checkpoint(checkpoint_path, model, optimizer) 
else:
    epoch = 0
    train_losses_old = []
    val_losses_old = []  

lr_params = {
    "num_warmup_steps": 0.05 * num_epochs * len(train_dataloader), 
    "num_training_steps": num_epochs * len(train_dataloader)
    }
lr_scheduler = get_cosine_schedule_with_warmup(optimizer, **lr_params)
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
                                                  lr_scheduler=lr_scheduler,
                                                  train_dataloader=train_dataloader,
                                                  val_dataloader=val_dataloader,
                                                  save_model_path=save_dir)

    
save_losses_and_png(
    train_losses_old + train_losses, 
    val_losses_old + val_losses, 
    save_dir
    )

logging.info("Completed training")
logging.info("Training losses: %s", train_losses)
logging.info("Validation losses: %s", val_losses)