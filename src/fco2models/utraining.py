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
    model.to(device)
    
    dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    # Learning rate scheduler
    num_training_steps = num_epochs * len(dataloader)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps
    )
    
    noise_scheduler = DDPMScheduler(num_train_timesteps=timesteps)
    loss_fn = nn.MSELoss()
    
    # Initialize wandb
    # wandb.init(project="conditional-diffusion")
    # wandb.watch(model)
    
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in progress_bar:
            # print(type(batch))
            # print(batch)
            batch = batch[0].to(device)
            target = batch[:, 0:1, :]  # Target (first channel)
            # print(target[0])
            context = batch[:, 1:, :]  # Context (other two channels)
            # print(context[0])

            # inpute nans with zeros in target
            mask = torch.isnan(target)
            target = torch.where(mask, torch.zeros_like(target), target).float()
            # add mask to context
            context = torch.cat([context, (~mask).float()], dim=1)
            # assert that there are no nans in context
            assert torch.isnan(context).sum() == 0
            
            # Sample noise and timestep
            noise = torch.randn_like(target)
            timesteps = torch.randint(0, 
                                      noise_scheduler.config.num_train_timesteps, 
                                      (target.shape[0],), 
                                      device=device).long()
            
            # Add noise using the scheduler
            noisy_input = noise_scheduler.add_noise(target, noise, timesteps)
            noisy_input = torch.cat([noisy_input, context], dim=1).to(device).float()
            # print(noisy_input.shape)
            
            # Forward pass
            optimizer.zero_grad()
            model_output = model(sample=noisy_input, timestep=timesteps)[0]
            
            # Compute loss ignoring nans
            loss = loss_fn(model_output[~mask], target[~mask])
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            
            # Logging
            epoch_loss += loss.item()
            # wandb.log({"loss": loss.item(), "lr": optimizer.param_groups[0]['lr']})
            progress_bar.set_postfix(loss=loss.item())
        
        print(f"Epoch {epoch+1} Loss: {epoch_loss / len(dataloader):.6f}")

    return model, noise_scheduler