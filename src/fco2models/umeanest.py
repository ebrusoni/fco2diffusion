import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

def train_mean_estimator(model, num_epochs, old_epoch, train_dataloader, val_dataloader, optimizer, lr_scheduler, save_model_path=None, rmse_const=None):
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
            target = batch[:, 0:1]
            context = batch[:, 1:]

            not_nan_mask = context[:, -1:, :].bool()
            # replace nan with zeros
            target = torch.where(~not_nan_mask, torch.zeros_like(target), target).float()
            #assert np.isnan(target.cpu().numpy()).sum() == 0, "target contains nan values"
        
            #concatenate the noisy target with the context and the mask
            input = context
            input = input.to(device).float()
            
            # Get the model prediction
            mean_pred = model(input, torch.zeros(batch.shape[0], ).to(device).float(), return_dict=False)[0]

            # Calculate the loss
            optimizer.zero_grad()
            loss = loss_fn(mean_pred[not_nan_mask], target[not_nan_mask])	
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

                not_nan_mask = context[:, -1:, :].bool()
                # replace nan with zeros
                target = torch.where(~not_nan_mask, torch.zeros_like(target), target).float()

                #concatenate the noisy target with the context and the mask
                input = context
                input = input.to(device).float()

                mean_pred = model(input, torch.zeros(batch.shape[0], ).to(device), return_dict=False)[0]

                # Calculate the loss
                val_loss += loss_fn(mean_pred[not_nan_mask], target[not_nan_mask]).item()
        val_losses.append(val_loss / len(val_dataloader))
        print(f"Validation Loss: {val_loss / len(val_dataloader):.6f}")
        if rmse_const is not None:
            print(f"Validation RMSE: {np.sqrt(val_loss / len(val_dataloader)) * rmse_const:.6f}")

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