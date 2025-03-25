import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
# import wandb
from tqdm import tqdm
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers import DDPMScheduler, UNet1DModel


# Training function
def train_diffusion(model, dataset, num_epochs=10, batch_size=16, lr=1e-4, warmup_steps=500, num_workers=0, timesteps=1000):
    """training loop for diffusion model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    model.to(device)
    
    dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    # Learning rate scheduler
    # num_training_steps = num_epochs * len(dataloader)
    # lr_scheduler = get_cosine_schedule_with_warmup(
    #     optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps
    # )
    
    noise_scheduler = DDPMScheduler(num_train_timesteps=timesteps)
    loss_fn = nn.MSELoss()
    
    # Initialize wandb
    # wandb.init(project="conditional-diffusion")
    # wandb.watch(model)
    
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        # noise = torch.randn((batch_size,1,64)).to(device)
        
        for batch in progress_bar:
            batch = batch[0].to(device)
            target = batch[:, 0:1, :]
            context = batch[:, 1:, :]
            noise = torch.randn_like(target).to(device).float()
            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (batch.shape[0],), device=device).long()
            nan_mask = torch.isnan(target)
            # replace nan with zeros
            target = torch.where(nan_mask, torch.zeros_like(target), target)
            # Add noise to the clean images according to the noise magnitude at each timestep
            noisy_target = noise_scheduler.add_noise(target, noise, timesteps)

            #concatenate the noisy target with the context and the mask
            noisy_input = torch.cat([noisy_target, context, (~nan_mask).float()], dim=1)
            noisy_input = noisy_input.to(device).float()
            
            # Get the model prediction
            noise_pred = model(noisy_input, timesteps, return_dict=False)[0]

            # Calculate the loss
            loss = loss_fn(noise_pred[~nan_mask], noise[~nan_mask])	
            loss.backward(loss)
            epoch_loss += loss.item()
            
            # Update the model parameters with the optimizer
            optimizer.step()
            optimizer.zero_grad()

            # Update the learning rate
            # lr_scheduler.step()

            #update loss in progress bar
            progress_bar.set_postfix({"Loss": loss.item()})
        print(f"Epoch {epoch+1} Loss: {epoch_loss / len(dataloader):.6f}")
        
    return model, noise_scheduler