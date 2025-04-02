import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
# import wandb
from tqdm import tqdm
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers import DDPMScheduler, UNet1DModel

def check_gradients(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    print(f"Total gradient norm: {total_norm}")

# Training function
def train_diffusion(model, num_epochs, train_dataloader, val_dataloader, noise_scheduler, optimizer, lr_scheduler, save_model_path=None):
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
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
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
            optimizer.zero_grad()
            loss = loss_fn(noise_pred[~nan_mask], noise[~nan_mask])	
            loss.backward(loss)
            epoch_loss += loss.item()
            
            # Update the model parameters with the optimizer
            optimizer.step()

            # Update the learning rate
            lr_scheduler.step()

            #update loss in progress bar
            progress_bar.set_postfix({"Loss": loss.item()})
            batch.detach()
        train_losses.append(epoch_loss / len(train_dataloader))
        print(f"Epoch {epoch+1} Loss: {epoch_loss / len(train_dataloader):.6f}")
        check_gradients(model)

        # print validation loss
        model.eval()
        val_loss = 0.0
        for batch in val_dataloader:
            noisy_input, noise, nan_mask, timesteps = prep_sample(batch, noise_scheduler, device)
            noise_pred = model(noisy_input, timesteps, return_dict=False)[0]
            loss = loss_fn(noise_pred[~nan_mask], noise[~nan_mask])
            val_loss += loss.item()
        val_losses.append(val_loss / len(val_dataloader))
        print(f"Validation Loss: {val_loss / len(val_dataloader):.6f}")

        if save_model_path and (epoch+1) % 10 == 0:
            torch.save(model.state_dict(), save_model_path+f"_{epoch+1}.pt")

    return model, train_losses, val_losses


def prep_sample(batch, noise_scheduler, device):
    batch = batch[0].to(device)
    target = batch[:, 0:1, :]
    context = batch[:, 1:, :]

    noise = torch.randn_like(target).to(device).float()
    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (batch.shape[0],), device=device).long()

    # Replace nan with zeros
    nan_mask = torch.isnan(target)
    target = torch.where(nan_mask, torch.zeros_like(target), target)
    # Add noise to the clean images according to the noise magnitude at each timestep
    noisy_target = noise_scheduler.add_noise(target, noise, timesteps)
    # Concatenate the noisy target with the context
    noisy_input = torch.cat([noisy_target, context, (~nan_mask).float()], dim=1)
    noisy_input = noisy_input.to(device).float()
    return noisy_input, noise, nan_mask, timesteps



def full_denoise(model, noise_scheduler, context_loader, n_samples=10):
    """full denoising loop for diffusion model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    model.to(device)
    samples = []
    for ix in range(n_samples):
        print(ix)
        batch_sample = []
        for context_batch in context_loader:
            context = context_batch.to(device)
            # context = context.unsqueeze(0)
            sample = torch.randn((context.shape[0], 1, context.shape[2])).to(device)
            for i, t in enumerate(noise_scheduler.timesteps):
                mask = torch.ones_like(sample).bool()
                # concat noise, context and mask
                sample_context = torch.cat([sample, context, mask], dim=1)
            
                # Get model pred
                with torch.no_grad():
                    residual = model(sample_context, t, return_dict=False)[0]
            
                # Update sample with step
                sample = noise_scheduler.step(residual, t, sample).prev_sample
            batch_sample.append(sample.detach().cpu())
        batch_sample = torch.cat(batch_sample, dim=0)
        samples.append(batch_sample)

    return np.array(samples)



# plot final samples

