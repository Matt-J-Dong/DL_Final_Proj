from typing import List
import numpy as np
from torch import nn
from torch.nn import functional as F
import torch


def build_mlp(layers_dims: List[int]):
    layers = []
    for i in range(len(layers_dims) - 2):
        layers.append(nn.Linear(layers_dims[i], layers_dims[i + 1]))
        layers.append(nn.LayerNorm(layers_dims[i + 1]))  # Replaced BatchNorm1d
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(layers_dims[-2], layers_dims[-1]))
    return nn.Sequential(*layers)


from torch import nn
import torch

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class ResNetEncoder(nn.Module):
    def __init__(self, output_dim=256, input_channels=2, input_shape=(2, 64, 64)):
        super(ResNetEncoder, self).__init__()
        
        # Initial convolution layer to reduce spatial dimensions
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual layers
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)

        # Dynamically compute `fc_input_dim` based on input shape
        dummy_input = torch.zeros(1, *input_shape)
        with torch.no_grad():
            x = self._forward_features(dummy_input)
        self.fc_input_dim = x.numel()

        # Fully connected layer for output
        self.fc = nn.Linear(self.fc_input_dim, output_dim)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = [ResNetBlock(in_channels, out_channels, stride, downsample)]
        for _ in range(1, blocks):
            layers.append(ResNetBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def _forward_features(self, x):
        # Forward pass through convolution and pooling layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Forward pass through residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

    def forward(self, x):
        # Forward through feature extractor
        x = self._forward_features(x)
        
        # Flatten and pass through fully connected layer
        x = x.view(x.size(0), -1)  # Dynamically flatten based on feature map size
        x = self.fc(x)
        return x



class JEPA_Model(nn.Module):
    def __init__(self, device="cuda", repr_dim=256, action_dim=2):
        super(JEPA_Model, self).__init__()
        self.device = device
        self.repr_dim = repr_dim
        self.action_dim = action_dim
        self.encoder = ResNetEncoder(output_dim=repr_dim, input_channels=2).to(device)
        self.target_encoder = ResNetEncoder(output_dim=repr_dim, input_channels=2).to(device)

        # Copy weights
        self.target_encoder.load_state_dict(self.encoder.state_dict(), strict=False)

        # Freeze target encoder parameters
        for param in self.target_encoder.parameters():
            param.requires_grad = False


    def forward(self, states, actions):
        B, T, C, H, W = states.shape
        pred_encs = []

        o_0 = states[:, 0].to(self.device)
        s_0 = self.encoder(o_0)
        pred_encs.append(s_0)

        s_prev = s_0
        for t in range(1, T):
            u_prev = actions[:, t - 1].to(self.device)
            s_pred = self.predictor(s_prev, u_prev)
            pred_encs.append(s_pred)
            s_prev = s_pred

        pred_encs = torch.stack(pred_encs, dim=1)
        return pred_encs

    def train_step(self, states, actions, criterion, optimizer, momentum=0.99):
        B, T, C, H, W = states.shape
        total_loss = 0

        o_0 = states[:, 0].to(self.device)
        s_0 = self.encoder(o_0)
        s_prev = s_0

        for t in range(1, T):
            o_n = states[:, t].to(self.device)
            s_target = self.target_encoder(o_n)

            u_prev = actions[:, t - 1].to(self.device)
            s_pred = self.predictor(s_prev, u_prev)

            loss = criterion(s_pred, s_target)
            total_loss += loss

            s_prev = s_pred

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            for param_q, param_k in zip(self.encoder.parameters(), self.target_encoder.parameters()):
                param_k.data = momentum * param_k.data + (1 - momentum) * param_q.data

        return total_loss.item()


class Prober(torch.nn.Module):
    def __init__(self, embedding: int, arch: str, output_shape: List[int]):
        super(Prober, self).__init__()
        self.output_dim = np.prod(output_shape)
        arch_list = list(map(int, arch.split("-"))) if arch else []
        layers_dims = [embedding] + arch_list + [self.output_dim]
        self.prober = build_mlp(layers_dims)

    def forward(self, e):
        output = self.prober(e)
        return output.view(-1, *self.output_shape)  # Reshape output to match output_shape
