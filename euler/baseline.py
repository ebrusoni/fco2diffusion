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

from fco2models.models import UNet2DModelWrapper
from fco2models.utraining import prep_data
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
from fco2models.utraining import train_diffusion, sinusoidal_day_embedding
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
batch_size = 64
save_dir = f'../models/baseline/'
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

logging.info("Training with larger random dataset")
df_train = pd.read_parquet(DATA_PATH+'traindf_100km_random_reshaped.pq')
df_val = pd.read_parquet(DATA_PATH+'valdf_100km_random_reshaped.pq')
df_2021 = pd.read_parquet(DATA_PATH+'df_100km_random_reshaped_2021.pq')
logging.info("Using already separated train and validation datasets")


predictors = ['sst_cci', 'sss_cci', 'chl_globcolour', 'year', 'lon', 'lat']
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

# print mins and maxs of the data
for i in range(train_ds.shape[1]):
    print(f"train_ds {i} min: {np.nanmin(train_ds[:, i, :])}, max: {np.nanmax(train_ds[:, i, :])}")
    print(f"val_ds {i} min: {np.nanmin(val_ds[:, i, :])}, max: {np.nanmax(val_ds[:, i, :])}")

train_dataset = TensorDataset(torch.tensor(train_ds))
val_dataset = TensorDataset(torch.tensor(val_ds))
train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


layers_per_block = 2
down_block_types = ('DownBlock2D', 'DownBlock2D')
up_block_types = ('UpBlock2D', 'UpBlock2D')
model_params = {
    "sample_size": (4, 64),
    "in_channels": 1,
    "out_channels": 1,
    "layers_per_block": layers_per_block,
    "block_out_channels": (32, 64),
    "down_block_types": down_block_types,
    "up_block_types": up_block_types,
    "norm_num_groups": 32
}

model = UNet2DModelWrapper(**model_params)
num_epochs = 5
optimizer = optim.AdamW(model.parameters(), lr=lr)

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


def train_baseline(model, num_epochs, old_epoch, train_dataloader, val_dataloader, optimizer, lr_scheduler, save_model_path=None):
    """training loop for diffusion model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    model.to(device)
    
    loss_fn = nn.MSELoss()
    
    # Initialize wandb
    # wandb.init(project="conditional-diffusion")
    # wandb.watch(model)
    train_losses = []
    val_losses = []
    for epoch in range(old_epoch, num_epochs):
        model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        # noise = torch.randn((batch_size,1,64)).to(device)
        
        for batch in progress_bar:
            batch = batch[0].to(device)
            target = batch[:, 0:1, :]
            context = batch[:, 1:, :]

            nan_mask = torch.isnan(target)
            # replace nan with zeros
            target = torch.where(nan_mask, torch.zeros_like(target), target).float()
            

            #concatenate the noisy target with the context and the mask
            input = torch.cat([context, (~nan_mask).float()], dim=1)
            input = input.to(device).float()
            
            # Get the model prediction
            mean_pred = model(input, torch.zeros(batch.shape[0], ).to(device).float(), return_dict=False)[0]

            # Calculate the loss
            optimizer.zero_grad()
            loss = loss_fn(mean_pred[~nan_mask], target[~nan_mask])	
            loss.backward(loss)
            epoch_loss += loss.item()
            
            # Update the model parameters with the optimizer
            optimizer.step()

            # Update the learning rate
            lr_scheduler.step()

            #update loss in progress bar
            progress_bar.set_postfix({"Loss": loss.item()})
            # batch.detach()
        train_losses.append(epoch_loss / len(train_dataloader))
        print(f"Epoch {epoch+1} Loss: {epoch_loss / len(train_dataloader):.6f}")
        #check_gradients(model)

        # print validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_dataloader:
                batch = batch[0].to(device)
                target = batch[:, 0:1, :]
                context = batch[:, 1:, :]

                nan_mask = torch.isnan(target)
                # replace nan with zeros
                target = torch.where(nan_mask, torch.zeros_like(target), target).float()

                #concatenate the noisy target with the context and the mask
                input = torch.cat([context, (~nan_mask).float()], dim=1)
                input = input.to(device).float()

                mean_pred = model(input, torch.zeros(batch.shape[0], ).to(device), return_dict=False)[0]

                # Calculate the loss
                val_loss += loss_fn(mean_pred[~nan_mask], target[~nan_mask]).item()
        val_losses.append(val_loss / len(val_dataloader))
        print(f"Validation Loss: {val_loss / len(val_dataloader):.6f}")

        if save_model_path and (epoch+1) % 10 == 0:
            torch.save(model.state_dict(), save_model_path+f"e_{epoch+1}.pt")

    # save model checkpoint
    if save_model_path:
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': lr_scheduler.state_dict() if lr_scheduler else None,
            'rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            'train_losses': train_losses,
            'val_losses': val_losses,
        }, save_model_path+f"final_model_e_{num_epochs}.pt")
    return model, train_losses, val_losses
                   

model, train_losses, val_losses = train_baseline(model,
                                                  num_epochs=num_epochs, 
                                                  old_epoch=epoch,
                                                  optimizer=optimizer, 
                                                  lr_scheduler=lr_scheduler,
                                                  train_dataloader=train_dataloader,
                                                  val_dataloader=val_dataloader,
                                                  save_model_path=save_dir)


# save the model parameters to a json file
with open(save_dir +'hyperparameters.json', 'w') as f:
    param_dict = json.dumps(param_dict, indent=4)
    f.write(param_dict)
    
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