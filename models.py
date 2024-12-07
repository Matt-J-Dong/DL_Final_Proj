# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import numpy as np

def build_mlp(layers_dims: List[int]):
    layers = []
    for i in range(len(layers_dims) - 2):
        layers.append(nn.Linear(layers_dims[i], layers_dims[i + 1]))
        layers.append(nn.BatchNorm1d(layers_dims[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(layers_dims[-2], layers_dims[-1]))
    return nn.Sequential(*layers)


class BasicBlock(nn.Module):
    """A simple ResNet Basic Block."""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
        
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class SimpleResNet(nn.Module):
    """
    A simple ResNet-like model that takes an image and outputs a feature vector.
    We'll assume the input is [B, C, H, W] with C=2, H=W=64 (as given).
    """

    def __init__(self, repr_dim=256):
        super(SimpleResNet, self).__init__()
        self.in_planes = 32
        # Initial conv
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, stride=2, padding=1, bias=False) # [B, 32, 32, 32]
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(True)

        # ResNet layers
        self.layer1 = self._make_layer(32, 2, stride=2) # [B,32,16,16]
        self.layer2 = self._make_layer(64, 2, stride=2) # [B,64,8,8]
        self.layer3 = self._make_layer(128, 2, stride=2) # [B,128,4,4]
        # Now we have [B,128,4,4] ~ 128*4*4=2048 features

        # Increase features to 256 with one more layer
        self.conv2 = nn.Conv2d(128, 256, kernel_size=1, stride=1, bias=False) # [B,256,4,4]
        self.bn2 = nn.BatchNorm2d(256)

        # Flatten and FC
        # 256 * 4 *4 = 4096 features
        self.fc = nn.Linear(256*4*4, repr_dim)

    def _make_layer(self, planes, blocks, stride):
        layers = []
        layers.append(BasicBlock(self.in_planes, planes, stride))
        self.in_planes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_planes, planes, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        # x: [B,2,64,64]
        x = self.relu(self.bn1(self.conv1(x)))  # [B,32,32,32]
        x = self.layer1(x) # [B,32,16,16]
        x = self.layer2(x) # [B,64,8,8]
        x = self.layer3(x) # [B,128,4,4]
        x = self.bn2(self.conv2(x)) # [B,256,4,4]

        x = x.view(x.size(0), -1) # [B, 4096]
        x = self.fc(x) # [B,256]
        return x


class SimpleResNetModel(nn.Module):
    """
    This model ignores actions and just returns the same embedding for each timestep.
    It takes (init_state, actions) and returns embeddings [B,T,D].
    """
    def __init__(self, device="cuda", repr_dim=256, action_dim=2):
        super(SimpleResNetModel, self).__init__()
        self.device = device
        self.repr_dim = repr_dim
        self.action_dim = action_dim
        self.encoder = SimpleResNet(repr_dim=repr_dim).to(device)

    def forward(self, init_state, actions):
        # init_state: [B,C,H,W]
        # actions: [B,T-1,action_dim]
        # We will produce T = (T-1)+1 steps of embeddings
        B = init_state.size(0)
        if actions.ndim == 3:
            T_minus_one = actions.size(1)
        else:
            T_minus_one = 0
        T = T_minus_one + 1

        s_0 = self.encoder(init_state) # [B,D]

        # Just repeat s_0 for all T steps
        s_all = s_0.unsqueeze(1).repeat(1, T, 1) # [B,T,D]
        return s_all

    @property
    def repr_dim(self):
        return 256


class Prober(nn.Module):
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
            layers.append(nn.Linear(f[i], f[i + 1]))
            layers.append(nn.ReLU(True))
        layers.append(nn.Linear(f[-2], f[-1]))
        self.prober = nn.Sequential(*layers)

    def forward(self, e):
        output = self.prober(e)
        return output