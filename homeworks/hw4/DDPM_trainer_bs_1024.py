# %% [markdown]
# <a href="https://colab.research.google.com/github/rll/deepul/blob/master/homeworks/hw4/hw4.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown]
# # Getting Started
# 
# ## Overview
# This semester, all homeworks will be conducted through Google Colab notebooks. All code for the homework assignment will be written and run in this notebook. Running in Colab will automatically provide a GPU, but you may also run this notebook locally by following [these instructions](https://research.google.com/colaboratory/local-runtimes.html) if you wish to use your own GPU.
# 
# You will save images in the notebooks to use and fill out a given LaTeX template which will be submitted to Gradescope, along with your notebook code.
# 
# ## Using Colab
# On the left-hand side, you can click the different icons to see a Table of Contents of the assignment, as well as local files accessible through the notebook.
# 
# Make sure to go to **Runtime -> Change runtime type** and select **GPU** as the hardware accelerator. This allows you to use a GPU. Run the cells below to get started on the assignment. Note that a session is open for a maximum of 12 hours, and using too much GPU compute may result in restricted access for a short period of time. Please start the homework early so you have ample time to work.
# 
# **If you loaded this notebook from clicking "Open in Colab" from github, you will need to save it to your own Google Drive to keep your work.**
# 
# ## General Tips
# In each homework problem, you will implement and train various diffusion models.
# 
# Feel free to print whatever output (e.g. debugging code, training code, etc) you want, as the graded submission will be the submitted pdf with images.
# 
# After you complete the assignment, download all of the images outputted in the results/ folder and upload them to the figure folder in the given latex template.
# 
# Run the cells below to download and load up the starter code.

# %%
# !if [ -d deepul ]; then rm -Rf deepul; fi
# !git clone https://github.com/rll/deepul.git
# !pip install ./deepul
# !pip install scikit-learn

# %%
from deepul.hw4_helper import *
import warnings
warnings.filterwarnings('ignore')
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"



# %% [markdown]
# # Question 1: Toy Dataset [30pt]
# 
# In this question, we will train a simple diffusion models a toy 2D dataset.
# 
# Execute the cell below to visualize our datasets.

# %%
visualize_q1_dataset()

# %% [markdown]
# For code simplicity, we will train a continuous-time variant of the diffusion prompt. In practice training objectives and code between discrete-time and continuous-time diffusion models are similar.
# 
# Given a data element $x$ and neural net $f_\theta(x, t)$, implement the following diffusion training steps:
# 1. Sample the diffusion timestep: $t \sim \text{Uniform}(0, 1)$
# 2. Compute the noise-strength following a cosine schedule: $\alpha_t = \cos\left(\frac{\pi}{2}t\right), \sigma_t = \sin\left(\frac{\pi}{2}t\right)$
# 3. Apply the forward process - Sample noise $\epsilon \sim N(0,I)$ (same shape as $x$) and compute noised $x_t = \alpha_t x + \sigma_t \epsilon$
# 4. Estimate $\hat{\epsilon} = f_\theta(x_t, t)$
# 5. Optimize the loss $L = \lVert \epsilon - \hat{\epsilon} \rVert_2^2$. Here, it suffices to just take the mean over all dimensions.
# 
# Note that for the case of continuous-time diffusion, the forward process is $x_{0\to1}$ and reverse process is $x_{1\to0}$
# 
# Use an MLP for $f_\theta$ to optimize the loss. You may find the following details helpful.
# * Normalize the data using mean and std computed from the train dataset
# * Train 100 epochs, batch size 1024, Adam with LR 1e-3 (100 warmup steps, cosine decay to 0)
# * MLP with 4 hidden layers and hidden size 64
# * Condition on t by concatenating it with input x (i.e. 2D x + 1D t = 3D cat(x, t))
# 
# To sample, implement the standard DDPM sampler. You may find the equation from the [DDIM paper](https://arxiv.org/pdf/2010.02502.pdf) helpful, rewritten and re-formatted here for convenience.
# $$x_{t-1} = \alpha_{t-1}\left(\frac{x_t - \sigma_t\hat{\epsilon}}{\alpha_t}\right) + \sqrt{\sigma_{t-1}^2 - \eta_t^2}\hat{\epsilon} + \eta_t\epsilon_t$$
# where $\epsilon_t \sim N(0, I)$ is random Gaussian noise. For DDPM, let
# $$\eta_t = \sigma_{t-1}/\sigma_t\sqrt{1 - \alpha_t^2/\alpha_{t-1}^2}$$
# To run the reverse process, start from $x_1 \sim N(0, I)$ and perform `num_steps` DDPM updates (a hyperparameter), pseudocode below.
# ```
# ts = linspace(1 - 1e-4, 1e-4, num_steps + 1)
# x = sample_normal
# for i in range(num_steps):
#     t = ts[i]
#     tm1 = ts[i + 1]
#     eps_hat = model(x, t)
#     x = DDPM_UPDATE(x, eps_hat, t, tm1)
# return x
# ```
# Note: If you encounter NaNs, you may need to clip $\sigma_{t-1}^2 - \eta_t^2$ to 0 if it goes negative, as machine precision issues can make it a very small negative number (e.g. -1e-12) if its too close to 0

# %%
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Optimizer

import math
def get_cosine_schedule_with_warmup(
    optimizer:Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

# %%

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import math
from torch.utils.data import DataLoader

class net(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(3, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 2)

    def forward(self, x,t):
        x = torch.cat((x,t),-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        out = self.fc4(x)
        return out
    
class Trainner():
    def __init__(self, model, device, train_loader, test_loader, epochs,learning_rate,warmup_steps):
         
         self.model = model
         self.model.to(device)
         self.device = device
         self.train_loader = train_loader
         self.test_loader = test_loader 
         self.epochs = epochs
         self.lr = learning_rate
         
         self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
         self.schedular = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps = warmup_steps, num_training_steps = epochs * len(train_loader))
         self.train_losses = []
         self.test_losses = []
        
    def alpha(self,t):
        return torch.cos((math.pi/2)*t)
    
    def sigma(self,t):
        return torch.sin((math.pi/2)*t)
     
    def train(self):
        
        for epoch in range(self.epochs):
            total_loss = []
            total_val_loss = []
            
            self.model.train(True)
            
            for batch in self.train_loader:
                x = batch.to(self.device)
                
                batch_size = x.shape[0]
                
                t = torch.FloatTensor(batch_size).uniform_(0,1).reshape(-1,1).to(self.device) # (batch_size)
                eps = torch.randn_like(x).to(self.device)
                # add noise
                x_t = self.alpha(t) * x + self.sigma(t) * eps
                pred_eps = self.model(x_t, t)
                
                L = F.mse_loss(pred_eps, eps)
                
                self.optimizer.zero_grad()
                
                L.backward()
                self.optimizer.step()
                self.schedular.step()
                
                total_loss.append(L.item())
            # eval
            self.model.eval()
            with torch.no_grad():
                for val_batch in self.test_loader:
                    x = val_batch.to(self.device)
                    batch_size = x.shape[0]
                    
                    t = torch.FloatTensor(batch_size).uniform_(0,1).reshape(-1,1).to(self.device) # (batch_size)
                    eps = torch.randn_like(x).to(self.device)
                    
                    x_t = self.alpha(t) * x + self.sigma(t) * eps
                    
                    pred_eps = self.model(x_t, t)
                    
                    L = F.mse_loss(pred_eps, eps)
                    
                    total_val_loss.append(L.item())
            train_loss = np.mean(total_loss)
            test_loss = np.mean(total_val_loss)
            
            self.train_losses.append(train_loss)
            self.test_losses.append(test_loss)
            if epoch % 10 ==0:
                print(f"Epoch {epoch}; train Loss {train_loss}; val loss {test_loss}; lr: {self.schedular.get_lr()}")                    
             
    def DDPM_update(self, x, eps_hat,t,tm1):
        eps = torch.randn_like(x)
        n_t = (self.sigma(tm1) / self.sigma(t)) * torch.sqrt(1 - (self.alpha(t)**2 / self.alpha(tm1)**2))
        
        x_tm1 = self.alpha(tm1) * ((x - self.sigma(t)*eps_hat ) / self.alpha(t)) + torch.sqrt(torch.clip((self.sigma(tm1)**2 - n_t**2), min = 0))*eps_hat + n_t * eps
        return x_tm1

    def draw_samples(self, num_samples = 2000):
        num_steps = np.power(2, np.linspace(0, 9, 9)).astype(int)
        samples = []
        for step in num_steps:
            x = torch.normal(mean = 0,std = 1,size = (num_samples, 2)).to(self.device)
            
            ts = torch.linspace(1 - 1e-4, 1e-4, step +1 ).to(self.device)
            for i in range(step):
                t= ts[i].expand(num_samples, 1)
                tm1 = ts[i+1].expand(num_samples, 1)
                with torch.no_grad():
                    eps_hat = self.model(x,t)
                x = self.DDPM_update(x,eps_hat, t, tm1)
                
            samples.append(x.cpu())
        
        return np.array(samples)
                
def q1(train_data, test_data):
    """
    train_data: A (100000, 2) numpy array of 2D points
    test_data: A (10000, 2) numpy array of 2D points

    Returns
    - a (# of training iterations,) numpy array of train losses evaluated every minibatch
    - a (# of num_epochs + 1,) numpy array of test losses evaluated at the start of training and the end of every epoch
    - a numpy array of size (9, 2000, 2) of samples drawn from your model.
      Draw 2000 samples for each of 9 different number of diffusion sampling steps
      of evenly logarithmically spaced integers 1 to 512
      hint: np.power(2, np.linspace(0, 9, 9)).astype(int)
    """
    # Training configuration
    epochs = 100
    batch_size = 1024
    learning_rate = 1e-3
    warmup_steps = 100
    hidden_size = 64
    
    train_data = torch.tensor(train_data, dtype = torch.float32)
    test_data = torch.tensor(test_data, dtype = torch.float32)
    
    train_loader = DataLoader(train_data, batch_size= batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size = batch_size, shuffle = False)
    
    model = net(hidden_size=hidden_size)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = Trainner(model, device, train_loader, test_loader, epochs , learning_rate, warmup_steps)
    trainer.train()
    
    train_losses = trainer.train_losses
    test_losses = trainer.test_losses
    samples = trainer.draw_samples(2000)
    return train_losses, test_losses, samples

# %%
# q1_save_results(q1)

# %% [markdown]
# # Question 2: Pixel-Space Diffusion on CIFAR-10 [30pt]
# 
# In this question, we will train pixel-space UNet diffusion model on CIFAR-10
# 
# Execute the cell below to visualize our datasets.

# %%
visualize_q2_data()

# %% [markdown]
# We'll use a UNet architecture similar to the original [DDPM](https://arxiv.org/abs/2006.11239) paper. We provide the following pseudocode for each part of the model:
# ```
# def timestep_embedding(timesteps, dim, max_period=10000):
#     half = dim // 2
#     freqs = np.exp(-np.log(max_period) * np.arange(0, half, dtype=float32) / half)
#     args = timesteps[:, None].astype(float32) * freqs[None]
#     embedding = cat([np.cos(args), np.sin(args)], axis=-1)
#     if dim % 2:
#         embedding = cat([embedding, np.zeros_like(embedding[:, :1])], axis=-1)
#     return embedding
# 
# ResidualBlock(in_channels, out_channels, temb_channels)
#     Given x, temb
#     h = Conv2d(in_channels, out_channels, 3, padding=1)(x)
#     h = GroupNorm(num_groups=8, num_channels=out_channels)(h)
#     h = SiLU()(h)
#     
#     temb = Linear(temb_channels, out_channels)(temb)
#     h += temb[:, :, None, None] # h is BxDxHxW, temb is BxDx1x1
#     
#     h = Conv2d(out_channels, out_channels, 3, padding=1)(h)
#     h = GroupNorm(num_groups=8, num_channels=out_channels)(h)
#     h = SiLU()(h)
#     
#     if in_channels != out_channels:
#         x = Conv2d(in_channels, out_channels, 1)(x)
#     return x + h
#     
# Downsample(in_channels)
#     Given x
#     return Conv2d(in_channels, in_channels, 3, stride=2, padding=1)(x)
# 
# Upsample(in_channels)
#     Given x
#     x = interpolate(x, scale_factor=2)
#     x = Conv2d(in_channels, in_channels, 3, padding=1)(x)
#     return x
#     
# UNet(in_channels, hidden_dims, blocks_per_dim)
#     Given x, t
#     temb_channels = hidden_dims[0] * 4
#     emb = timestep_embedding(t, hidden_dims[0])
#     emb = Sequential(Linear(hidden_dims[0], temb_channels), SiLU(), Linear(temb_channels, temb_channels))(emb)
#     
#     h = Conv2d(in_channels, hidden_dims[0], 3, padding=1)(x)
#     hs = [h]
#     prev_ch = hidden_dims[0]
#     down_block_chans = [prev_ch]
#     for i, hidden_dim in enumerate(hidden_dims):
#         for _ in range(blocks_per_dim):
#             h = ResidualBlock(prev_ch, hidden_dim, temb_channels)(h, emb)
#             hs.append(h)
#             prev_ch = hidden_dim
#             down_block_chans.append(prev_ch)
#         if i != len(hidden_dims) - 1:
#             h = Downsample(prev_ch)(h)
#             hs.append(h)
#             down_block_chans.append(prev_ch)
#     
#     h = ResidualBlock(prev_ch, prev_ch, temb_channels)(h, emb)
#     h = ResidualBlock(prev_ch, prev_ch, temb_channels)(h, emb)
#     
#     for i, hidden_dim in list(enumerate(hidden_dims))[::-1]:
#         for j in range(blocks_per_dim + 1):
#             dch = down_block_chans.pop()
#             h = ResidualBlock(prev_ch + dch, hidden_dim, temb_channels)(cat(h, hs.pop()), emb)
#             prev_ch = hidden_dim
#             if i and j == blocks_per_dim:
#                 h = Upsample(prev_ch)(h)
#     
#     h = GroupNorm(num_groups=8, num_channels=prev_ch)(h)
#     h = SiLU()(h)
#     out = Conv2d(prev_ch, in_channels, 3, padding=1)(h)
#     return out
# ```

# %% [markdown]
# **Hyperparameter details**
# * Normalize data to [-1, 1]
# * UNET with hidden_dims as [64, 128, 256, 512] and 2 blocks_per_dim
# * Train 60 epochs, batch size 256, Adam with LR 1e-3 (100 warmup steps, cosine decay to 0)
# * For diffusion schedule, sampling and loss, use the same setup as Q1
# 
# You may also find it helpful to clip $\hat{x} = \frac{x_t - \sigma_t \hat{\epsilon}}{\alpha_t}$ to [-1, 1] during each sampling step.

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
import math

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, temb_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        
        self.temb_proj = nn.Linear(temb_channels, out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Shortcut connection for dimension matching
        if self.in_channels != self.out_channels:
            self.shortcut = nn.Conv2d(self.in_channels, self.out_channels, 1)
        else:
            self.shortcut = nn.Identity()
        
    def forward(self, x, temb):
        # First conv + norm + activation
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h)
        
        # Add time embedding
        temb_proj = self.temb_proj(temb)
        h += temb_proj[:, :, None, None]  # h is BxDxHxW, temb is BxDx1x1
        
        # Second conv + norm + activation
        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)
        
        # Skip connection
        return self.shortcut(x) + h

class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1)
    
    def forward(self, x):
        return self.conv(x)
    
class Upsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2)
        x = self.conv(x)
        return x
    
class Unet(nn.Module):
    def __init__(self, in_channels, hidden_dims, blocks_per_dim):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dims = hidden_dims
        self.blocks_per_dim = blocks_per_dim
        temb_channels = hidden_dims[0] * 4
        
        # Time embedding layers
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dims[0], temb_channels),
            nn.SiLU(),
            nn.Linear(temb_channels, temb_channels)
        )
        
        # Initial convolution
        self.init_conv = nn.Conv2d(in_channels, hidden_dims[0], 3, padding=1)
        
        # Build the architecture following the pseudocode exactly
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        
        # Track channel progression for upsampling path
        self.down_block_channels = []
        
        # Build downsampling path
        prev_ch = hidden_dims[0]
        self.down_block_channels.append(prev_ch)  # for initial conv
        
        for i, hidden_dim in enumerate(hidden_dims):
            level_blocks = nn.ModuleList()
            for _ in range(blocks_per_dim):
                level_blocks.append(ResidualBlock(prev_ch, hidden_dim, temb_channels))
                prev_ch = hidden_dim
                self.down_block_channels.append(prev_ch)
            self.down_blocks.append(level_blocks)
            
            if i != len(hidden_dims) - 1:
                self.down_samples.append(Downsample(prev_ch))
                self.down_block_channels.append(prev_ch)  # for downsample
        
        # Middle blocks
        self.mid_block1 = ResidualBlock(prev_ch, prev_ch, temb_channels)
        self.mid_block2 = ResidualBlock(prev_ch, prev_ch, temb_channels)
        
        # Build upsampling path - reverse order of hidden_dims
        up_dims = list(reversed(hidden_dims))
        for i, hidden_dim in enumerate(up_dims):
            level_blocks = nn.ModuleList()
            for j in range(blocks_per_dim + 1):
                # Calculate input channels: current + skip connection
                if j == 0:
                    # First block of each level gets concatenated with skip
                    skip_ch = self.down_block_channels[-(1 + j + i * (blocks_per_dim + 1))]
                    input_ch = prev_ch + skip_ch
                else:
                    # Subsequent blocks also get skip connections according to pseudocode
                    skip_ch = self.down_block_channels[-(1 + j + i * (blocks_per_dim + 1))]
                    input_ch = prev_ch + skip_ch
                
                level_blocks.append(ResidualBlock(input_ch, hidden_dim, temb_channels))
                prev_ch = hidden_dim
            
            self.up_blocks.append(level_blocks)
            
            # Add upsampling layer for all but the last level (i=0 corresponds to the first level in original order)
            if i < len(up_dims) - 1:
                self.up_samples.append(Upsample(prev_ch))
        
        # Final layers
        self.final_norm = nn.GroupNorm(num_groups=8, num_channels=prev_ch)
        self.final_conv = nn.Conv2d(prev_ch, in_channels, 3, padding=1)
        
    def forward(self, x, t):
        # Time embedding
        emb = timestep_embedding(t, self.hidden_dims[0])
        emb = self.time_mlp(emb) # [B,256]
        
        # Initial convolution
        h = self.init_conv(x) # [B,64,32,32]
        hs = [h] 
        down_block_chans = [self.hidden_dims[0]]  # Start with initial conv channels
        
        # Downsampling path - exactly following pseudocode
        prev_ch = self.hidden_dims[0]
        for i, hidden_dim in enumerate(self.hidden_dims):
            for block in self.down_blocks[i]:
                h = block(h, emb)
                hs.append(h)
                prev_ch = hidden_dim
                down_block_chans.append(prev_ch)
            
            if i != len(self.hidden_dims) - 1:
                h = self.down_samples[i](h)
                hs.append(h)
                down_block_chans.append(prev_ch)
        
        # Middle blocks
        h = self.mid_block1(h, emb)
        h = self.mid_block2(h, emb)
        # print("h shape:", h.shape)
        # Upsampling path - exactly following pseudocode
        up_dims = list(reversed(self.hidden_dims))
        for i, hidden_dim in enumerate(up_dims):
            for j in range(self.blocks_per_dim + 1):
                dch = down_block_chans.pop()
                skip_h = hs.pop()
                # print("before concat h shape:", h.shape)
                # print("skip_h shape:", skip_h.shape)
                h = torch.cat([h, skip_h], dim=1)
                h = self.up_blocks[i][j](h, emb)
                prev_ch = hidden_dim
                # print("after concat h shape:", h.shape)
                # Upsample after the last block of each level (except the final level)
                if j == self.blocks_per_dim and i < len(up_dims) - 1:
                    h = self.up_samples[i](h)
                    # print("after upsample h shape:", h.shape)
        
        # Final layers
        # print("final h shape:", h.shape)
        h = self.final_norm(h)
        h = F.silu(h)
        out = self.final_conv(h)
        
        return out

def broadcast(values, broadcast_to):
    """Ensure values tensor is on the same device and properly broadcasted"""
    values = values.flatten()
    while len(values.shape) < len(broadcast_to.shape):
        values = values.unsqueeze(-1)
    return values.to(broadcast_to.device)

def forward_diffusion(images, timesteps) -> tuple[torch.Tensor, torch.Tensor]:
    """
    https://arxiv.org/pdf/2006.11239.pdf, equation (14), the term inside epsilon_theta
    """
    device = images.device
    gaussian_noise = torch.randn(images.shape, device=device)
    betas = torch.linspace(1e-4, 1e-2, 1000, device=device)
    alphas = 1 - betas
    alphas_hat = torch.cumprod(alphas, dim=0)
    alpha_hat = alphas_hat[timesteps]
    alpha_hat = broadcast(alpha_hat, images)
    
    return torch.sqrt(alpha_hat) * images + torch.sqrt(1 - alpha_hat) * gaussian_noise, gaussian_noise

def reverse_diffusion(model, device, num_timesteps=1000, shape=(10, 3, 32, 32)):
    """
    Sample from the model using reverse diffusion
    """
    model.eval()
    with torch.no_grad():
        # Start from pure noise
        x = torch.randn(shape, device=device)
        
        betas = torch.linspace(1e-4, 1e-2, num_timesteps, device=device)
        alphas = 1 - betas
        alphas_hat = torch.cumprod(alphas, dim=0)
        
        # Reverse diffusion process
        for t in reversed(range(num_timesteps)):
            t_tensor = torch.full((x.shape[0],), t, device=device, dtype=torch.long)
            
            # Predict noise
            predicted_noise = model(x, t_tensor)
            
            # Compute alpha values
            alpha_t = alphas[t]
            alpha_hat_t = alphas_hat[t]
            beta_t = betas[t]
            
            # Remove predicted noise
            x = (1 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1 - alpha_hat_t)) * predicted_noise)
            
            # Optional: clip to [-1, 1] as mentioned in instructions
            x = torch.clamp(x, -1, 1)
            
            # Add noise if not the last step
            if t > 0:
                noise = torch.randn_like(x)
                x = x + torch.sqrt(beta_t) * noise
        
        return x
def q2(train_data, test_data):
    """
    train_data: A (50000, 32, 32, 3) numpy array of images in [0, 1]
    test_data: A (10000, 32, 32, 3) numpy array of images in [0, 1]

    Returns
    - a (# of training iterations,) numpy array of train losses evaluated every minibatch
    - a (# of num_epochs + 1,) numpy array of test losses evaluated at the start of training and the end of every epoch
    - a numpy array of size (10, 10, 32, 32, 3) of samples in [0, 1] drawn from your model.
      The array represents a 10 x 10 grid of generated samples. Each row represents 10 samples generated
      for a specific number of diffusion timesteps. Do this for 10 evenly logarithmically spaced integers
      1 to 512, i.e. np.power(2, np.linspace(0, 9, 10)).astype(int)
    """
    
    train_losses = []
    test_losses = []
    
    # Normalize data to [-1, 1]
    train_data = train_data * 2 - 1
    test_data = test_data * 2 - 1
    
    # Convert to PyTorch tensors and change to channel first
    train_data = torch.tensor(train_data, dtype=torch.float32)
    test_data = torch.tensor(test_data, dtype=torch.float32)
    train_data = torch.permute(train_data, (0, 3, 1, 2)).contiguous()
    test_data = torch.permute(test_data, (0, 3, 1, 2)).contiguous()
    
    batch_size = 1024
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    hidden_dims = [64, 128, 256, 512]
    model = Unet(in_channels=3, hidden_dims=hidden_dims, blocks_per_dim=2)
    
    # Move model to device BEFORE creating optimizer
    model = model.to(device)
    
    # Compile the model for better performance
    model = torch.compile(model)
    
    epochs = 60
    learning_rate = 8e-4
    warmup_steps = 100
    num_timesteps = 1000
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=epochs * len(train_loader)
    )
    
    # Initialize best model tracking
    best_test_loss = float('inf')
    best_model_state = None
    
    # Evaluate initial test loss
    model.eval()
    with torch.no_grad():
        total_test_loss = []
        for batch in test_loader:
            x = batch.to(device)
            timesteps = torch.randint(0, num_timesteps, (x.size(0),), device=device).long()
            noisy_images, noise = forward_diffusion(x, timesteps)
            pred_noise = model(noisy_images, timesteps)
            loss = F.mse_loss(pred_noise, noise)
            total_test_loss.append(loss.item())
        initial_test_loss = np.mean(total_test_loss)
        test_losses.append(initial_test_loss)
        
        # Save initial model as best if it's the first
        if initial_test_loss < best_test_loss:
            best_test_loss = initial_test_loss
            best_model_state = model.state_dict().copy()
    
    for epoch in range(epochs):
        model.train()
        total_loss = []
        
        for batch in train_loader:
            x = batch.to(device)
            optimizer.zero_grad()
            timesteps = torch.randint(0, num_timesteps, (x.size(0),), device=device).long()
            
            noisy_images, noise = forward_diffusion(x, timesteps)
            pred_noise = model(noisy_images, timesteps)
            
            loss = F.mse_loss(pred_noise, noise)
            loss.backward()
            
  
            
            optimizer.step()
            scheduler.step()
            train_losses.append(loss.item())
            total_loss.append(loss.item())
        
        # Evaluate test loss at end of epoch
        model.eval()
        with torch.no_grad():
            total_test_loss = []
            for batch in test_loader:
                x = batch.to(device)
                timesteps = torch.randint(0, num_timesteps, (x.size(0),), device=device).long()
                noisy_images, noise = forward_diffusion(x, timesteps)
                pred_noise = model(noisy_images, timesteps)
                loss = F.mse_loss(pred_noise, noise)
                total_test_loss.append(loss.item())
            current_test_loss = np.mean(total_test_loss)
            test_losses.append(current_test_loss)
            
            # Save best model
            if current_test_loss < best_test_loss:
                best_test_loss = current_test_loss
                best_model_state = model.state_dict().copy()
        
        if epoch % 1 == 0:
            print(f"Epoch {epoch}; train loss: {np.mean(total_loss):.4f}; test loss: {test_losses[-1]:.4f}")
    
    # Load best model for sampling and save it
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        torch.save(best_model_state, 'best_diffusion_model_bs_1024.pth')
    # load model from best_diffusion_model_bs_1024.pth
    model.load_state_dict(torch.load('best_diffusion_model_bs_1024.pth'))
    
    # Generate samples
    timestep_values = np.power(2, np.linspace(0, 9, 10)).astype(int)
    samples = np.zeros((10, 10, 32, 32, 3))
    
    for i, num_steps in enumerate(timestep_values):
        # Generate 10 samples with num_steps diffusion steps
        generated = reverse_diffusion(model, device, num_timesteps=min(num_steps, 1000), shape=(10, 3, 32, 32))
        
        # Convert back to [0, 1] and change to channel last
        generated = torch.clamp((generated + 1) / 2, 0, 1)
        generated = generated.permute(0, 2, 3, 1).cpu().numpy()
        samples[i] = generated
    
    return np.array(train_losses), np.array(test_losses), samples

# %%
q2_save_results(q2)

# %% [markdown]
# # Question 3: Class-Conditional Latent-Space Diffusion on CIFAR-10 with DiT [60pt]
# 
# In this question, we will train latent-space [Diffusion Transformer (DiT)](https://arxiv.org/abs/2212.09748) model on CIFAR-10 **with class conditioning.**
# 
# Execute the cell below to visualize our datasets.

# %%
visualize_q3_data()

# %% [markdown]
# ## Part 3(a) VAE reconstructions and Scale Factor [10pt]
# 
# Similar to how we learned a AR model in VQGAN latent space for homework 1, in this question, you will train a diffusion model in the latent space of a VAE. Note that since diffusion models can model continuous distributions, we do not need a discretization bottleneck in the VAE, and the latent space itself is continuous.
# 
# Below, we specify each of the relevant properties or functions that you may need.

# %%
# @property
# def latent_shape(self) -> Tuple[int, int, int]:
#     """Size of the encoded representation"""
#
# def encode(self, x: np.ndarray) -> np.ndarray:
#     """Encode an image x. Note: Channel dim is in dim 1
#
#     Args:
#         x (np.ndarray, dtype=float32): Image to encode. shape=(batch_size, 3, 32, 32). Values in [-1, 1]
#
#     Returns:
#         np.ndarray: Encoded image. shape=(batch_size, 4, 8, 8). Unbounded values
#     """
#
# def decode(self, z: np.ndarray) -> np.ndarray:
#     """Decode an encoded image.
#
#     Args:
#         z (np.ndarray, dtype=float32): Encoded image. shape=(batch_size, 4, 8, 8). Unbounded values.
#
#     Returns:
#         np.ndarray: Decoded image. shape=(batch_size, 3, 32, 32). Values in [-1, 1]
#     """
#

# %%
# @property
# def latent_shape(self) -> Tuple[int, int, int]:
#     """Size of the encoded representation"""
#
# def encode(self, x: np.ndarray) -> np.ndarray:
#     """Encode an image x. Note: Channel dim is in dim 1
#
#     Args:
#         x (np.ndarray, dtype=float32): Image to encode. shape=(batch_size, 3, 32, 32). Values in [-1, 1]
#
#     Returns:
#         np.ndarray: Encoded image. shape=(batch_size, 4, 8, 8). Unbounded values
#     """
#
# def decode(self, z: np.ndarray) -> np.ndarray:
#     """Decode an encoded image.
#
#     Args:
#         z (np.ndarray, dtype=float32): Encoded image. shape=(batch_size, 4, 8, 8). Unbounded values.
#
#     Returns:
#         np.ndarray: Decoded image. shape=(batch_size, 3, 32, 32). Values in [-1, 1]
#     """
#

# %%
# @property
# def latent_shape(self) -> Tuple[int, int, int]:
#     """Size of the encoded representation"""
#
# def encode(self, x: np.ndarray) -> np.ndarray:
#     """Encode an image x. Note: Channel dim is in dim 1
#
#     Args:
#         x (np.ndarray, dtype=float32): Image to encode. shape=(batch_size, 3, 32, 32). Values in [-1, 1]
#
#     Returns:
#         np.ndarray: Encoded image. shape=(batch_size, 4, 8, 8). Unbounded values
#     """
#
# def decode(self, z: np.ndarray) -> np.ndarray:
#     """Decode an encoded image.
#
#     Args:
#         z (np.ndarray, dtype=float32): Encoded image. shape=(batch_size, 4, 8, 8). Unbounded values.
#
#     Returns:
#         np.ndarray: Decoded image. shape=(batch_size, 3, 32, 32). Values in [-1, 1]
#     """
#

# %% [markdown]
# In this part, feed the given images through the VAE to compute and visualize reconstructions. In addition, you will compute a scale factor that will be needed during diffusion training to help normalize the data.
# 
# To estimate the scale factor, encode 1000 images into the VAE latent space, flatten the entire tensor along all dimensions, and compute the standard deviation.

# %%
def q3_a(images, vae):
    """
    images: (1000, 32, 32, 3) numpy array in [0, 1], the images to pass through the encoder and decoder of the vae
    vae: a vae model, trained on the relevant dataset

    Returns
    - a numpy array of size (50, 2, 32, 32, 3) of the decoded image in [0, 1] consisting of pairs
      of real and reconstructed images
    - a float that is the scale factor
    """

    """ YOUR CODE HERE """

    return autoencoded_images, scale_factor

# %%
q3a_save_results(q3_a)

# %% [markdown]
# ## Part 3(b) Diffusion Transformer [30pt]
# In this part, you will train a Diffusion Transformer (Dit) on the latent space of the above pretrained VAE. You can use your Transformer implementation from HW1 as the core part of the DiT implementation.
# 
# Below, we outline the key modifications needed on top of the standard Transformer for DiT.
# ```
# def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
#     assert embed_dim % 2 == 0
# 
#     # use half of dimensions to encode grid_h
#     emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
#     emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
# 
#     emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
#     return emb
# 
# 
# def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
#     assert embed_dim % 2 == 0
#     omega = np.arange(embed_dim // 2, dtype=np.float64)
#     omega /= embed_dim / 2.
#     omega = 1. / 10000**omega  # (D/2,)
# 
#     pos = pos.reshape(-1)  # (M,)
#     out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product
# 
#     emb_sin = np.sin(out) # (M, D/2)
#     emb_cos = np.cos(out) # (M, D/2)
# 
#     emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
#     return emb
# 
# def get_2d_sincos_pos_embed(embed_dim, grid_size):
#     grid_h = np.arange(grid_size, dtype=np.float32)
#     grid_w = np.arange(grid_size, dtype=np.float32)
#     grid = np.meshgrid(grid_w, grid_h)  # here w goes first
#     grid = np.stack(grid, axis=0)
# 
#     grid = grid.reshape([2, 1, grid_size, grid_size])
#     pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
#     return pos_embed
# 
# def modulate(x, shift, scale):
#     return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
# 
# DiTBlock(hidden_size, num_heads)
#     Given x (B x L x D), c (B x D)
#     c = SiLU()(c)
#     c = Linear(hidden_size, 6 * hidden_size)(c)
#     shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = c.chunk(6, dim=1)
#     
#     h = LayerNorm(hidden_size, elementwise_affine=False)(x)
#     h = modulate(h, shift_msa, scale_msa)
#     x = x + gate_msa.unsqueeze(1) * Attention(hidden_size, num_heads)(h)
#     
#     h = LayerNorm(hidden_size, elementwise_affine=False)(x)
#     h = modulate(h, shift_mlp, scale_mlp)
#     x = x + gate_mlp.unsqueeze(1) * MLP(hidden_size)(h)
#     
#     return x
#     
# FinalLayer(hidden_size, patch_size, out_channels)
#     Given x (B x L x D), c (B x D)
#     c = SiLU()(c)
#     c = Linear(hidden_size, 2 * hidden_size)(c)
#     shift, scale = c.chunk(2, dim=1)
#     x = LayerNorm(hidden_size, elementwise_affine=False)(x)
#     x = modulate(x, shift, scale)
#     x = Linear(hidden_size, patch_size * patch_size * out_channels)(x)
#     return x
#     
# DiT(input_shape, patch_size, hidden_size, num_heads, num_layers, num_classes, cfg_dropout_prob)
#     Given x (B x C x H x W) - image, y (B) - class label, t (B) - diffusion timestep
#     x = patchify_flatten(x) # B x C x H x W -> B x (H // P * W // P) x D, P is patch_size
#     x += pos_embed # see get_2d_sincos_pos_embed
#     
#     t = compute_timestep_embedding(t) # Same as in UNet
#     if training:
#         y = dropout_classes(y, cfg_dropout_prob) # Randomly dropout to train unconditional image generation
#     y = Embedding(num_classes + 1, hidden_size)(y)
#     c = t + y
#     
#     for _ in range(num_layers):
#         x = DiTBlock(hidden_size, num_heads)(x, c)
#     
#     x = FinalLayer(hidden_size, patch_size, out_channels)(x)
#     x = unpatchify(x) # B x (H // P * W // P) x (P * P * C) -> B x C x H x W
#     return x
# ```

# %% [markdown]
# **Hyperparameter details**
# * Normalize image to [-1, 1], (2) Encode using the VAE, (3) divide latents by the scale_factor compute in part (a)
# * Transformer with patch_size 2, hidden_size 512, num_heads 8, num_layers 12
# * Train 60 epochs, batch size 256, Adam with LR 1e-3 (100 warmup steps, cosine decay to 0)
# * When sampling, remember to multiple the final generated latents by the scale_factor before feeding it through the decoder
# * For diffusion schedule, sampling and loss, use the same setup as Q1
# 
# For class conditioning, learn an embedding for each class, and an extra embedding to represent the null class. To condition, add the class embedding to the timestep embedding before feeding it into the transformer blocks (see pseudocode). **Train your class conditional diffusion models while dropping out the class (replace with null class) 10% of the time. This will be necessary for part (c).**
# 
# **Remember to save your model parameters after training, as you will need them for part (c)**

# %%
def q3_b(train_data, train_labels, test_data, test_labels, vae):
    """
    train_data: A (50000, 32, 32, 3) numpy array of images in [0, 1]
    train_labels: A (50000,) numpy array of class labels
    test_data: A (10000, 32, 32, 3) numpy array of images in [0, 1]
    test_labels: A (10000,) numpy array of class labels
    vae: a pretrained VAE

    Returns
    - a (# of training iterations,) numpy array of train losses evaluated every minibatch
    - a (# of num_epochs + 1,) numpy array of test losses evaluated at the start of training and the end of every epoch
    - a numpy array of size (10, 10, 32, 32, 3) of samples in [0, 1] drawn from your model.
      The array represents a 10 x 10 grid of generated samples. Each row represents 10 samples generated
      for a specific class (i.e. row 0 is class 0, row 1 class 1, ...). Use 512 diffusion timesteps
    """

    """ YOUR CODE HERE """

    return train_losses, test_losses, samples

# %%
q3b_save_results(q3_b)

# %% [markdown]
# ## Part 3(c) Classifier-Free Guidance [20pt]
# In this part, you will implement [Classifier-Free Guidance](https://arxiv.org/abs/2207.12598) (CFG). CFG is a widely used method during diffusion model sampling to push samples towards more accurately aligning with the conditioning information (e.g. class, text caption).
# 
# Implement CFG requires a small modification to the diffusion sampling code. Given a CIFAR-10 class label, instead of using $\hat{\epsilon} = f_\theta(x_t, t, y)$ to sample, use:
# $$\hat{\epsilon} = f_\theta(x_t, t, \varnothing) + w(f_\theta(x_t, t, y) - f_\theta(x_t, t, \varnothing))$$
# where $w$ is a sampling hyperparameter that controls the strength of CFG. $\varnothing$ indicates the unconditional model with the class label dropped out, which your pre-trained UNet from 3(b) should support. Note that $w = 1$ recovers standard sampling.
# 
# Note: It may be expected to see worse samples (e.g. sautrated images) when CFG value is too high. Generation quality is closer to a U-shape when increasing CFG values (gets better, then worse)

# %%
def q3_c(vae):
    """
    vae: a pretrained vae

    Returns
    - a numpy array of size (4, 10, 10, 32, 32, 3) of samples in [0, 1] drawn from your model.
      The array represents a 4 x 10 x 10 grid of generated samples - 4 10 x 10 grid of samples
      with 4 different CFG values of w = {1.0, 3.0, 5.0, 7.5}. Each row of the 10 x 10 grid
      should contain samples of a different class. Use 512 diffusion sampling timesteps.
    """

    """ YOUR CODE HERE """

    return samples

# %%
q3c_save_results(q3_c)

# %%



