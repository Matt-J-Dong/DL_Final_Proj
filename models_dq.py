# models.py

from typing import List
import numpy as np
from torch import nn
from torch.nn import functional as F
import torch


def build_mlp(layers_dims: List[int]):
    layers = []
    for i in range(len(layers_dims) - 2):
        layers.append(nn.Linear(layers_dims[i], layers_dims[i + 1]))
        layers.append(nn.BatchNorm1d(layers_dims[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(layers_dims[-2], layers_dims[-1]))
    return nn.Sequential(*layers)


class Encoder(nn.Module):
    def __init__(self, output_dim=256):
        super(Encoder, self).__init__()
        # Define the CNN encoder
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, stride=2, padding=1)  # Output: [B, 32, 32, 32] if input is 64x64
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # Output: [B, 64, 16, 16]
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)  # Output: [B, 128, 8, 8]
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  # Output: [B, 256, 4, 4]
        self.bn4 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()

        # Initialize self.fc with correct input size
        self.fc_input_dim = output_dim * 5 * 5  #channels = 256, width = 5, height = 5
        self.fc = nn.Linear(self.fc_input_dim, output_dim)

    def forward(self, x):
        # x: [B, 2, H, W]
        x = self.relu(self.bn1(self.conv1(x)))  # [B, 32, H/2, W/2]
        x = self.relu(self.bn2(self.conv2(x)))  # [B, 64, H/4, W/4]
        x = self.relu(self.bn3(self.conv3(x)))  # [B, 128, H/8, W/8]
        x = self.relu(self.bn4(self.conv4(x)))  # [B, 256, H/16, W/16]
        x = x.view(x.size(0), -1)  # [B, C * H * W]
        x = self.fc(x)  # [B, output_dim]
        return x  # [B, D]


class Predictor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Predictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.bn1 = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(output_dim, output_dim)

    def forward(self, s_prev, u_prev):
        # s_prev: [B, D], u_prev: [B, action_dim]
        x = torch.cat([s_prev, u_prev], dim=-1)  # [B, D + action_dim]
        x = self.fc1(x)  # [B, output_dim]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)  # [B, output_dim]
        #print(f"Predictor output shape: {x.shape}")  # Debugging
        return x  # [B, D]


class JEPA_Model(nn.Module):
    def __init__(self, repr_dim=256, action_dim=2):
        super(JEPA_Model, self).__init__()
        self.encoder = Encoder(output_dim=repr_dim)
        self.predictor = Predictor(input_dim=repr_dim + action_dim, output_dim=repr_dim)
        # For simplicity, using the same architecture for target encoder
        self.target_encoder = Encoder(output_dim=repr_dim)
        # Initialize target encoder with same weights as encoder
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        # Freeze target encoder parameters
        for param in self.target_encoder.parameters():
            param.requires_grad = False

    def forward(self, init_state, actions):
        """
        Args:
            init_state: [B, C, H, W]
            actions: [B, T-1, action_dim]

        Output:
            predictions: [B, T, D]
        """
        B, T_minus_one, _ = actions.shape
        
        # Initialize list to store predicted representations
        pred_encs = []

        # Get initial representation s_0
        s_prev = self.encoder(init_state)  # [B, D]
        pred_encs.append(s_prev)

        for t in range(T_minus_one):
            u_prev = actions[:, t]  # [B, action_dim]
            s_pred = self.predictor(s_prev, u_prev)  # [B, D]
            pred_encs.append(s_pred)
            s_prev = s_pred

        # Stack pred_encs into [B, T, D]
        pred_encs = torch.stack(pred_encs, dim=1)  # [B, T, D]

        return pred_encs


class Prober(torch.nn.Module):
    def __init__(
        self,
        embedding: int,
        arch: str,
        output_shape: List[int],
    ):
        super().__init__()
        self.output_dim = np.prod(output_shape)
        self.output_shape = output_shape
        self.arch = arch

        arch_list = list(map(int, arch.split("-"))) if arch != "" else []
        f = [embedding] + arch_list + [self.output_dim]
        layers = []
        for i in range(len(f) - 2):
            layers.append(torch.nn.Linear(f[i], f[i + 1]))
            layers.append(torch.nn.ReLU(True))
        layers.append(torch.nn.Linear(f[-2], f[-1]))
        self.prober = torch.nn.Sequential(*layers)

    def forward(self, e):
        output = self.prober(e)
        return output