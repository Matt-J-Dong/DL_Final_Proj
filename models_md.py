from typing import List
import numpy as np
from torch import nn
import torch
from torchvision.models.resnet import BasicBlock


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
        self.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet-like layers
        self.layer1 = self._make_layer(BasicBlock, in_channels=64, out_channels=64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, in_channels=64, out_channels=128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, in_channels=128, out_channels=256, num_blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, output_dim)

    def _make_layer(self, block, in_channels, out_channels, num_blocks, stride=1):
        downsample = None
        # Add a downsample layer if channel or spatial dimensions differ
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = []
        layers.append(block(in_channels, out_channels, stride=stride, downsample=downsample))
        for _ in range(1, num_blocks):
            layers.append(block(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class Predictor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Predictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.bn1 = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(output_dim, output_dim)

    def forward(self, s_prev, u_prev):
        x = torch.cat([s_prev, u_prev], dim=-1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class JEPA_Model(nn.Module):
    def __init__(self, device="cuda", repr_dim=256, action_dim=2):
        super(JEPA_Model, self).__init__()
        self.device = device
        self.repr_dim = repr_dim
        self.action_dim = action_dim
        self.encoder = Encoder(output_dim=repr_dim).to(device)
        self.predictor = Predictor(input_dim=repr_dim + action_dim, output_dim=repr_dim).to(device)
        self.target_encoder = Encoder(output_dim=repr_dim).to(device)
        self.target_encoder.load_state_dict(self.encoder.state_dict())
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
        T = T_minus_one + 1
        pred_encs = []

        s_prev = self.encoder(init_state)
        pred_encs.append(s_prev)

        for t in range(T_minus_one):
            u_prev = actions[:, t]
            s_pred = self.predictor(s_prev, u_prev)
            pred_encs.append(s_pred)
            s_prev = s_pred

        pred_encs = torch.stack(pred_encs, dim=1)  # [B,T,D]
        return pred_encs

    def variance_regularization(self, states, epsilon=1e-4):
        """
        Prevent representation collapse by enforcing variance in each embedding dimension.

        Args:
            states: [B,T,D] Predicted embeddings.
            epsilon: Minimum variance threshold.

        Returns:
            Regularization loss value.
        """
        # Reshape states from [B,T,D] to [B*T,D]
        states = states.view(-1, states.size(-1))  # [B*T, D]

        std = torch.sqrt(states.var(dim=0) + 1e-10)
        return torch.mean(torch.relu(epsilon - std))

    def covariance_regularization(self, states):
        """
        Reduces redundancy by decorrelating dimensions of embeddings.

        Args:
            states: [B,T,D] Predicted embeddings.

        Returns:
            Regularization loss value.
        """
        # Reshape states from [B,T,D] to [B*T,D]
        states = states.view(-1, states.size(-1))  # [B*T,D]

        batch_size, dim = states.size()
        norm_states = states - states.mean(dim=0, keepdim=True)
        cov_matrix = (norm_states.T @ norm_states) / (batch_size - 1)
        off_diag = cov_matrix - torch.diag(torch.diag(cov_matrix))
        return torch.sum(off_diag ** 2)

    def compute_energy(self, predicted_encs, target_encs, distance_function="l2"):
        """
        Compute the energy function.

        Args:
            predicted_encs: [B,T,D]
            target_encs: [B,T,D]
            distance_function: str ("l2" or "cosine")

        Returns:
            energy: Scalar
        """
        if distance_function == "l2":
            energy = torch.sum((predicted_encs - target_encs) ** 2)
        elif distance_function == "cosine":
            cos = nn.CosineSimilarity(dim=-1)
            energy = -torch.sum(cos(predicted_encs, target_encs))
        else:
            raise ValueError(f"Unknown distance function: {distance_function}")
        return energy

    def train_step(self, states, actions, optimizer, momentum=0.99, distance_function="l2"):
        """
        Perform a single training step.

        Args:
            states: [B,T,Ch,H,W]
            actions: [B,T-1,2]
            optimizer: Optimizer
            momentum: float
            distance_function: str
        """
        B, T, C, H, W = states.shape
        init_state = states[:, 0]
        pred_encs = self.forward(init_state, actions)  # [B,T,D]

        target_encs = []
        for t in range(T):
            o_t = states[:, t]
            s_target = self.target_encoder(o_t)
            target_encs.append(s_target)
        target_encs = torch.stack(target_encs, dim=1)  # [B,T,D]

        loss = self.compute_energy(pred_encs, target_encs, distance_function)

        # Add regularization
        lambda_var, lambda_cov = 1.0, 1.0
        loss += lambda_var * self.variance_regularization(pred_encs)
        loss += lambda_cov * self.covariance_regularization(pred_encs)

        optimizer.zero_grad()
        loss.backward()

        max_grad_norm = 1.0
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)

        optimizer.step()

        # Momentum update target encoder
        with torch.no_grad():
            for param_q, param_k in zip(self.encoder.parameters(), self.target_encoder.parameters()):
                param_k.data = momentum * param_k.data + (1 - momentum) * param_q.data

        return loss.item()
