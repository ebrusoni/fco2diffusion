import torch
import torch.nn as nn
from diffusers import UNet1DModel, UNet2DModel

class MLP(nn.Module):
    def __init__(self, input_dim=64, output_dim=10, hidden_dims=[64], activation=nn.ReLU, dropout_prob=0.0, num_timesteps=500):
        """
        Parametrizable Multi-Layer Perceptron (MLP)
        
        Args:
            input_dim (int): Dimension of the input features. Default is 64.
            output_dim (int): Dimension of the output. Default is 10.
            hidden_dims (list): List of integers representing the dimensions of each hidden layer.
                               Default is [64] (one hidden layer with 64 neurons).
            activation (torch.nn.Module): Activation function to use. Default is ReLU.
            dropout_prob (float): Dropout probability. Default is 0.0 (no dropout).
        """
        super(MLP, self).__init__()
        
        # Store model parameters
        self.input_dim = input_dim + 1  # +1 for the time dimension
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.num_timesteps = num_timesteps
        
        # Create list of dimensions for all layers (input, hidden, output)
        all_dims = [input_dim + 1] + hidden_dims + [output_dim]
        
        # Construct the network
        layers = []
        for i in range(len(all_dims) - 1):
            # Add linear layer
            layers.append(nn.Linear(all_dims[i], all_dims[i+1]))
            
            # Don't add activation and dropout after the output layer
            if i < len(all_dims) - 2:
                layers.append(activation())
                if dropout_prob > 0:
                    layers.append(nn.Dropout(dropout_prob))
        
        # Combine all layers into a sequential model
        self.model = nn.Sequential(*layers)
    
    def forward(self, x, time, **kwargs):
        """
        Forward pass through the network (add some resizing for compatibility with U-Net model).
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
            time (torch.Tensor): Time tensor
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim)
        """
        # flatten 2nd dimension of x
        x = x.view(x.shape[0], -1)
        # normalize timestamp using num_timesteps
        time = time / self.num_timesteps
        # Ensure time is a column vector
        time = time.view(-1, 1)
        # Concatenate time to input features
        x = torch.cat([x, time], dim=1)
        # Pass through the model
        return [self.model(x).view(x.shape[0], 1, self.output_dim), None]
    


# just a wrapper for the UNet2DModel forward to adjust the input shape
class UNet2DModelWrapper(UNet2DModel):
    
    def forward(self, x, time, **kwargs):
        # print(x.shape)
        # current shape (batch_size, channels, bins)
        # zero array with next greatest power of 2 as channels
        # channels = 16
        # next_log = torch.ceil(torch.log2(torch.tensor(x.shape[1]))).item()
        channels = 16#int(2**next_log)
        temp = torch.zeros((x.shape[0], 1, channels, x.shape[2]), device=x.device)
        temp[:, 0, :x.shape[1], :] = x
        x = temp
        # Pass through the model
        pred = super().forward(x, time, **kwargs)[0]
        # print(pred.squeeze(1)[:, 0:1, :].shape)
        return (pred.squeeze(1)[:, 0:1, :],)
    


    

        
