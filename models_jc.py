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
    def __init__(self, output_dim=256, input_channels=2):
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

        # Fully connected layer for output
        self.fc_input_dim = None
        self.fc = nn.Linear(256 * 8 * 8, output_dim)  # Default for input (64x64)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = []
        layers.append(ResNetBlock(in_channels, out_channels, stride, downsample))
        for _ in range(1, blocks):
            layers.append(ResNetBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        # Initial convolution and pooling
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # Flatten and fully connected layer
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x



class Encoder(nn.Module):
    def __init__(self, output_dim=256, input_shape=(2, 64, 64)):
        super(Encoder, self).__init__()
        # Define the CNN encoder
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, stride=2, padding=1)  # Output: [B, 32, 32, 32]
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # Output: [B, 64, 16, 16]
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)  # Output: [B, 128, 8, 8]
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  # Output: [B, 256, 4, 4]
        self.bn4 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()

        # Compute fc_input_dim during initialization using input_shape
        dummy_input = torch.zeros(1, *input_shape)
        with torch.no_grad():
            x = self.relu(self.bn1(self.conv1(dummy_input)))
            x = self.relu(self.bn2(self.conv2(x)))
            x = self.relu(self.bn3(self.conv3(x)))
            x = self.relu(self.bn4(self.conv4(x)))
        self.fc_input_dim = x.numel()
        self.fc = nn.Linear(self.fc_input_dim, output_dim)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Predictor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Predictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.ln1 = nn.LayerNorm(output_dim)  # Replaced BatchNorm1d
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(output_dim, output_dim)

    def forward(self, s_prev, u_prev):
        x = torch.cat([s_prev, u_prev], dim=-1)
        x = self.fc1(x)
        x = self.ln1(x)
        x = self.relu(x)
        x = self.fc2(x)
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