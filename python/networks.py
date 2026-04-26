import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from utils import NUM_CHANNELS

SCALAR_DIM = 7

class DepthWiseSeparableConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channel, in_channel,  3, stride, 1, groups=in_channel, bias=False)
        self.pointwise = nn.Conv2d(in_channel, out_channel, 1, bias=False)
        self.bn_depthwise = nn.BatchNorm2d(in_channel)
        self.bn_pointwise = nn.BatchNorm2d(out_channel)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn_depthwise(x)
        x = self.relu1(x)

        x = self.pointwise(x)
        x = self.bn_pointwise(x)

        return self.relu2(x)


class MobileNetEncoder(nn.Module):
    def __init__(self, in_channels=NUM_CHANNELS, embed_dim=128, width_mult=0.25):
        super().__init__()

        self.width_mult = width_mult

        self.conv1 = nn.Conv2d(in_channels, self.scale_channel(32), 3, 1, 1, bias=False) 
        self.bn1 = nn.BatchNorm2d(self.scale_channel(32))
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        
        self.depth_wise_conv_blocks = nn.Sequential(
            DepthWiseSeparableConvBlock(self.scale_channel(32),  self.scale_channel(64),  1),
            DepthWiseSeparableConvBlock(self.scale_channel(64),  self.scale_channel(128), 2),  
            DepthWiseSeparableConvBlock(self.scale_channel(128), self.scale_channel(128), 1),
            DepthWiseSeparableConvBlock(self.scale_channel(128), self.scale_channel(256), 2),   
            DepthWiseSeparableConvBlock(self.scale_channel(256), self.scale_channel(256), 1),
            DepthWiseSeparableConvBlock(self.scale_channel(256), self.scale_channel(512), 1),
            DepthWiseSeparableConvBlock(self.scale_channel(512), self.scale_channel(512), 1),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.proj = nn.Linear(self.scale_channel(512), embed_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.depth_wise_conv_blocks(x)

        x = self.pool(x)

        x = x.flatten(1)

        x = self.proj(x)
        x = self.relu2(x)

        return x

    def scale_channel(self, scaling_factor):
        return max(1, int(self.width_mult * scaling_factor))
    

class Actor(nn.Module):
    def __init__(self, in_channels=NUM_CHANNELS, hidden_dim=128, action_dim=6, width_mult=0.25):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mobilenet = MobileNetEncoder(in_channels, hidden_dim, width_mult)

        # Scalar encoder
        self.scalar_linear1 = nn.Linear(SCALAR_DIM, 32)
        self.scalar_relu1 = nn.ReLU()
        self.scalar_linear2 = nn.Linear(32, 32)
        self.scalar_relu2 = nn.ReLU()

        self.fusion_linear1 = nn.Linear(hidden_dim + 32, hidden_dim)

        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, action_dim)
        self.fusion_relu = nn.ReLU()

        self._init_weights()

    def forward(self, grid, scalars, hidden):
        grid_enc = self.mobilenet(grid)

        scalar_enc = self.scalar_linear1(scalars)
        scalar_enc = self.scalar_relu1(scalar_enc)
        scalar_enc = self.scalar_linear2(scalar_enc)
        scalar_enc = self.scalar_relu2(scalar_enc)

        enc_cat = torch.cat([grid_enc, scalar_enc], dim=1)

        fusion_enc = self.fusion_linear1(enc_cat)
        fusion_enc = self.fusion_relu(fusion_enc)

        new_hidden = self.gru(fusion_enc, hidden)
        logits = self.out(new_hidden)

        return logits, new_hidden

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.out.weight, gain=0.01)
        nn.init.zeros_(self.out.bias)

    def init_hidden(self, batch_size, device):
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def get_action_logprob_entropy(self, grid, scalars, hidden, action=None):
        logits, new_hidden = self.forward(grid, scalars, hidden)

        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()

        return action, dist.log_prob(action), dist.entropy(), new_hidden


class Critic(nn.Module):
    def __init__(self, global_grid_channels=NUM_CHANNELS, num_agents=2, hidden_dim=256, width_mult=0.25):
        super().__init__()

        self.mobilenet = MobileNetEncoder(global_grid_channels, hidden_dim, width_mult)

        # Scalar encoder
        self.scalar_linear1 = nn.Linear(num_agents * SCALAR_DIM, SCALAR_DIM + num_agents)
        self.scalar_linear2 = nn.Linear(SCALAR_DIM + num_agents, 32)
        self.scalar_linear3 = nn.Linear(32, 32)

        self.fusion_linear1 = nn.Linear(hidden_dim + 32, hidden_dim)
        
        self.out = nn.Linear(hidden_dim, 1)
        self.fusion_relu = nn.ReLU()

        self.scalar_relu1 = nn.ReLU()
        self.scalar_relu2 = nn.ReLU()
        self.scalar_relu3 = nn.ReLU()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, grid, scalars):
        grid_enc = self.mobilenet(grid)

        scalar_enc = self.scalar_linear1(scalars)
        scalar_enc = self.scalar_relu1(scalar_enc)
        scalar_enc = self.scalar_linear2(scalar_enc)
        scalar_enc = self.scalar_relu2(scalar_enc)
        scalar_enc = self.scalar_linear3(scalar_enc)
        scalar_enc = self.scalar_relu3(scalar_enc)

        enc_cat = torch.cat([grid_enc, scalar_enc], dim=1)

        fusion_enc = self.fusion_linear1(enc_cat)
        fusion_enc = self.fusion_relu(fusion_enc)

        return self.out(fusion_enc).squeeze(-1)
