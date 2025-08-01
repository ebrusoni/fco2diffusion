import torch
import torch.nn as nn
from torch import vmap
import copy
from torch.func import stack_module_state, functional_call
from diffusers import UNet1DModel, UNet2DModel
import importlib

class MLPNaiveEnsemble(nn.Module):
    """
    A naive ensemble whose .forward(x) returns logits with shape
    [E, B, C] — **no reduction is performed**.
    """
    def __init__(self, ensemble_size: int, mlp_class, mlp_kwargs):
        super().__init__()
        self.E = ensemble_size

        # create E real copies, then stack their states
        mlp_module, mlp_class = mlp_class.rsplit('.', 1)
        mlp_module = importlib.import_module(mlp_module)
        mlp_class = getattr(mlp_module, mlp_class)
   
        models  = [mlp_class(**mlp_kwargs) for _ in range(ensemble_size)]
        self.models = nn.ModuleList(models) 


    def forward(self, x, t, **kwargs):                       
        # Vectorise over the leading E dimension of params and buffers
        return (torch.stack([self.models[i](x, t, **kwargs)[0] for i in range(self.E)], axis=0), None)

class MLPEnsemble(nn.Module):
    """
    A vmap-based ensemble whose .forward(x) returns logits with shape
    [E, B, C] — **no reduction is performed**.
    """
    def __init__(self, ensemble_size: int, mlp_class, mlp_kwargs):
        super().__init__()
        self.E = ensemble_size

        # a template network we will call "functionally" (lives on 'meta' device)
        self.base = mlp_class(**mlp_kwargs).to('meta')
        #self.add_module("_base", base)    # appears in the state_dict

        # create E real copies, then stack their states
        models  = [mlp_class(**mlp_kwargs) for _ in range(ensemble_size)]
        params, buffers = stack_module_state(models)          # :contentReference[oaicite:0]{index=0}

        self.params = nn.ParameterList(params)                
        self.buffers = nn.ParameterList(buffers)            

    # ------- Functional forward for a single member --------------------------------
    def _fmodel(self, param_single, buffer_single, x, t):
        return functional_call(self.base, (param_single, buffer_single), (x,t))

    # ------- Public forward ---------------------------------------------------------
    def forward(self, x, t, **kwargs):                       # x: [B, …]
        # Pick up current buffers
        #buffers = {k: getattr(self, k) for k in self._buffers}
        x = x.expand(self.E, *x.shape)  # [E, B, ...]  # replicate x for each ensemble member
        x = t.expand(self.E, *t.shape)  # [E, B, ...]  # replicate t for each ensemble member
        # Vectorise over the leading E dimension of params and buffers
        return vmap(self._fmodel)(self.params, self.buffers, x, t)
        # result: [E, B, C]

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
    
    def forward(self, x, t, **kw):
        fixed_h = 32
        h = x.shape[1]
        pad = fixed_h - h
        if pad < 0:
            raise ValueError(f"More channels ({h}) than fixed_h ({fixed_h})")
        x = F.pad(x, (0, 0, 0, pad)).unsqueeze(1)     # (B, C, bins) → (B, 1, fixed_h, bins)
        pred = super().forward(x, t, **kw)[0]
        return (pred[:, 0, :1, :],)

# just a wrapper for the UNet2DModel forward to adjust the input shape
class UNet1DModelWrapper(UNet1DModel):
    
    def forward(self, x, t, **kw):
        pred = super().forward(x, t, **kw)[0]
        return (pred[:, 0:1, :],)

class ConvNet(nn.Module):
    def __init__(self, channels_in, input_dim=64, kernel_size=3):
        """
        Parametrizable Convolutional Neural Network (ConvNet)
        
        Args:
            channels_in (int): Number of input channels.
            input_dim (int): Dimension of the input features. Default is 64.
        """
        super(ConvNet, self).__init__()
        
        # Store model parameters
        self.input_dim = input_dim
        self.channels_in = channels_in
        self.channels_out = 1

        self.channels1 = 128
        self.channels2 = 64
        
        #self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv1d(channels_in, self.channels1, kernel_size=kernel_size, padding='same')
        self.conv2 = nn.Conv1d(self.channels1, self.channels2, kernel_size=kernel_size, padding='same')
        self.conv3 = nn.Conv1d(self.channels2, 1, kernel_size=kernel_size, padding='same')
        self.dropout = nn.Dropout(0.1)
        self.batchnorm1 = nn.BatchNorm1d(self.channels1)
        self.batchnorm2 = nn.BatchNorm1d(self.channels2)

    def forward(self, x, t, return_dict=False):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv3(x)

        return (x, None)


class ClassEmbedding(nn.Module):
    def __init__(self, num_condition_dimensions, output_dim, num_classes_per_dimension):
        """
        Class embedding layer for the UNet model.
        
        Args:
            num_condition_dimensions (int): Number of condition dimensions (5 in your case)
            output_dim (int): Dimension of the output embedding
            num_classes_per_dimension (list): List of number of classes for each dimension
        """
        super(ClassEmbedding, self).__init__()
        
        self.num_condition_dimensions = num_condition_dimensions
        self.output_dim = output_dim
        
        # Create an embedding layer for each condition dimension
        embedding_nets = []
        for i in range(num_condition_dimensions):
            embedding_nets.append(nn.Embedding(num_classes_per_dimension[i], output_dim))
        self.embedding = nn.ModuleList(embedding_nets)
        
        # Optional projection to match UNet's expected embedding dimension
        # Uncomment if needed
        # self.projection = nn.Linear(output_dim, block_out_channels[0] * 4)
    
    def forward(self, class_labels):
        """
        Forward pass through the class embedding layer.
        
        Args:
            class_labels (torch.Tensor): Class labels tensor of shape (batch_size, num_condition_dimensions)
            
        Returns:
            torch.Tensor: Combined class embeddings tensor of shape (batch_size, output_dim)
        """
        # Ensure class_labels is a 2D tensor
        if len(class_labels.shape) == 1:
            class_labels = class_labels.unsqueeze(1)

        # just take first entry of each segment for embedding
        class_labels = class_labels[:, :, 0]
        
        # Get the embeddings for each condition dimension
        embeddings_sum = torch.zeros((class_labels.shape[0], self.output_dim), device=class_labels.device)
        for i in range(self.num_condition_dimensions):
            embeddings_sum += self.embedding[i](class_labels[:, i])
        
        return embeddings_sum
    
class UNet2DWithClassEmbedding(UNet2DModel):
    def __init__(self, unet_config, class_embedding_config, position_feature_start):
        """
        UNet2D model with class embedding.
        
        Args:
            unet_config (dict): Configuration for the UNet2D model.
            class_embedding_config (dict): Configuration for the class embedding layer.
        """
        super(UNet2DWithClassEmbedding, self).__init__(**unet_config)
        
        # Initialize the class embedding layer
        self.my_class_embedding = ClassEmbedding(**class_embedding_config)
        self.ix = position_feature_start

    def forward(self, x, time, **kwargs):
        mask = x[:, -1:, :] # last channel is always the mask
        position_features = x[:, self.ix:-1, :]
        unet_features = torch.cat([x[:, :self.ix, :], mask], dim=1)
        # Get the class labels from the input tensor
        class_labels = self.my_class_embedding(position_features.int())
        
        batch_size, channels, bins = x.shape
        height = 16 # must be power of 2
        temp = torch.zeros((batch_size, 1, height, bins), device=unet_features.device)
        temp[:, 0, :channels, :] = x
        x = temp
        
        # Pass through the model
        pred = super().forward(x, time, **kwargs, class_labels=class_labels)[0]
        return (pred[:, 0, 0:1, :],)

import torch.nn.functional as F
class Unet2DClassifierFreeModel(UNet2DModel):
    def __init__(self, unet_config, keep_channels, num_channels, w=2.0):
        """
        UNet2D model with class embedding.
        
        Args:
            unet_config (dict): Configuration for the UNet2D model.
            class_embedding_config (dict): Configuration for the class embedding layer.
        """
        super(Unet2DClassifierFreeModel, self).__init__(**unet_config)
        self.w = w
        self.fixed_h = 16  # must be power of 2
        self.channel_mask = torch.full((self.fixed_h,), True, dtype=torch.bool)
        self.channel_mask[keep_channels] = False
        self.channel_mask[0] = False  # always keep the first channel (fCO2)
    
    def forward(self, x, time, **kwargs):
        h = x.shape[1]
        pad = self.fixed_h - h
        if pad < 0:
            raise ValueError(f"More channels ({h}) than fixed_h ({self.fixed_h})")
        x = F.pad(x, (0, 0, 0, pad))    # (B, C, bins) → (B, 1, fixed_h, bins)
        
        if self.training:
            uncond = torch.rand(x.shape[0], device=x.device) < 0.5
            x_uncond = x.clone()
            x_uncond[uncond.nonzero(as_tuple=True)[0].unsqueeze(1), self.channel_mask.nonzero(as_tuple=True)[0], :] = 0
            pred = super().forward(x_uncond.unsqueeze(1), time, **kwargs)[0]
            return (pred[:, 0, 0:1, :],)
            
        x_uncond = x.clone()
        x_uncond[:, self.channel_mask, :] = 0.0 # keep target, temperature, salinity, and mask, but set all other channels to 0
        pred = super().forward(x.unsqueeze(1), time, **kwargs)[0] if self.w != 0 else 0
        uncond_pred =  super().forward(x_uncond.unsqueeze(1), time, **kwargs)[0] if self.w != 1 else 0
        # classifier free prediction
        pred = uncond_pred + self.w * (pred - uncond_pred)
        return (pred[:, 0, 0:1, :],)
    
    def set_w(self, new_w):
        self.w = new_w

class UNet2DShipMix(UNet2DModel):
    def __init__(self, unet_config, ship_mix_cols):
        """
        UNet2D model with class embedding.
        
        Args:
            unet_config (dict): Configuration for the UNet2D model.
            class_embedding_config (dict): Configuration for the class embedding layer.
        """
        super(UNet2DShipMix, self).__init__(**unet_config)
        self.mixcols = ship_mix_cols
        self.fixed_h = 16
        self.mixmask = torch.full((self.fixed_h,), True, dtype=torch.bool)
        self.mixmask[ship_mix_cols] = False

    def forward(self, x, time, **kwargs):
        h = x.shape[1]
        pad = self.fixed_h - h
        if pad < 0:
            raise ValueError(f"More channels ({h}) than fixed_h ({self.fixed_h})")
        x = F.pad(x, (0, 0, 0, pad))    # (B, C, bins) → (B, 1, fixed_h, bins)
        
        if not self.training:
            pred = super().forward(x.unsqueeze(1), time, **kwargs)[0]
            return (pred[:, 0, 0:1, :],)
            
        ship_data = x[:, self.mixcols, :]
        model_input = x[:, self.mixmask, :].clone()
        lambda_mix = torch.rand((model_input.shape[0], 2, 1), device=model_input.device)
        temp_sal = model_input[:, [1,2], :]  # temperature and salinity
        temp_sal = temp_sal * (1-lambda_mix) + ship_data * lambda_mix
        nan_mask = ~torch.isnan(temp_sal)
        model_tempsal = model_input[:, [1,2], :]
        model_tempsal[nan_mask] = temp_sal[nan_mask]
        
        pred = super().forward(model_input.unsqueeze(1), time, **kwargs)[0]
        return (pred[:, 0, 0:1, :],)

from fco2models.time_models import TSTransformerEncoderClassiregressor
class TSEncoderWrapper(TSTransformerEncoderClassiregressor):
    def __init__(self, **TSconfig):
        """
        Wrapper for the TSTransformerEncoderClassiregressor to use it as a model.
        
        Args:
            **TSconfig: Configuration parameters for the TSTransformerEncoderClassiregressor.
        """
        super().__init__(**TSconfig)

    def forward(self, x, t, return_dict=False, **kwargs):
        b, c, s = x.shape
        #print(t.shape)
        #if t.ndim == 0:
        #    t = torch.full((b,), t.item()).to(x.device)
        #print(t.shape)
        #t = torch.stack([t] * s, dim=1).unsqueeze(1) # this is a time vector of shape (B, 1, S)
        #x = torch.cat([x, t], dim=1)  # concatenate time as a feature

        out = super().forward(x.permute(0, 2, 1), t, torch.ones((b, s + 1)).bool().to(x.device), **kwargs) # (B, S, C) → (B, C, S)

        return (out.unsqueeze(1), None) # return shape (B, 1, S)


class DiffusionEnsemble(nn.Module):
    def __init__(self, ensemble_size: int, diffusion_class, diffusion_kwargs):
        super().__init__()
        self.E = ensemble_size
        # create E real copies, then stack their states
        diffusion_module, diffusion_class = diffusion_class.rsplit('.', 1)
        diffusion_module = importlib.import_module(diffusion_module)
        diffusion_class = getattr(diffusion_module, diffusion_class)

        models  = [diffusion_class(**diffusion_kwargs) for _ in range(ensemble_size)]
        self.models = nn.ModuleList(models)

    def forward(self, x, t, **kwargs):
        # Vectorise over the leading E dimension of params and buffers
        return (torch.stack([self.models[i](x, t, **kwargs)[0] for i in range(self.E)], axis=0), None)