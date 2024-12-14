from typing import List
import torch
import torch.nn as nn
import numpy as np

def build_mlp(layers_dims: List[int], dropout=0.0):
    """Utility function to build an MLP with optional dropout."""
    layers = []
    for i in range(len(layers_dims) - 2):
        layers.append(nn.Linear(layers_dims[i], layers_dims[i + 1]))
        layers.append(nn.BatchNorm1d(layers_dims[i + 1]))
        layers.append(nn.ReLU(True))
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
    layers.append(nn.Linear(layers_dims[-2], layers_dims[-1]))
    return nn.Sequential(*layers)

class SmallCNN(nn.Module):
    """
    A smaller convolutional neural network to replace ResNet-18.
    This network consists of three convolutional blocks followed by a fully connected layer
    to produce the desired output embedding.
    """
    def __init__(self, input_channels=2, output_dim=256):
        super(SmallCNN, self).__init__()
        self.features = nn.Sequential(
            # First Convolutional Block
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Uncomment and modify these blocks as needed for deeper networks
            # Second Convolutional Block
            # nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(32),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2),

            # Third Convolutional Block
            # nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(64),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
        self.fc = nn.Linear(16, output_dim)  # Fully connected layer to output_dim

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class Encoder(nn.Module):
    def __init__(self, output_dim=256, input_channels=2):
        super(Encoder, self).__init__()
        # Initialize the custom SmallCNN without pretrained weights
        self.features = SmallCNN(input_channels=input_channels, output_dim=output_dim)

    def forward(self, x):
        x = self.features(x)
        return x

class RecurrentPredictor(nn.Module):
    """
    A simple recurrent predictor using GRU to maintain hidden state across time steps.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.0):
        super(RecurrentPredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, s_prev, u_prev, hidden=None):
        """
        s_prev: (B, repr_dim)
        u_prev: (B, action_dim)
        hidden: (1, B, hidden_dim) or None
        Returns:
            s_pred: (B, output_dim)
            hidden: (1, B, hidden_dim)
        """
        # Concatenate state and action
        x = torch.cat([s_prev, u_prev], dim=-1).unsqueeze(1)  # (B, 1, input_dim)
        out, hidden = self.gru(x, hidden)  # out: (B, 1, hidden_dim)
        out = self.dropout(out)
        s_pred = self.fc(out.squeeze(1))  # (B, output_dim)
        s_pred = self.relu(s_pred)
        return s_pred, hidden

class RecurrentJEPA_Model(nn.Module):
    def __init__(self, device="cuda", repr_dim=256, action_dim=2, hidden_dim=512, dropout=0.0):
        super(RecurrentJEPA_Model, self).__init__()
        self.device = device
        self.repr_dim = repr_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.encoder = Encoder(output_dim=repr_dim, input_channels=2).to(device)
        # The input dimension to the predictor is repr_dim + action_dim
        self.predictor = RecurrentPredictor(input_dim=repr_dim + action_dim,
                                           hidden_dim=hidden_dim,
                                           output_dim=repr_dim,
                                           dropout=dropout).to(device)
        self.target_encoder = Encoder(output_dim=repr_dim, input_channels=2).to(device)
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        for param in self.target_encoder.parameters():
            param.requires_grad = False

    def forward(self, init_state, actions):
        """
        init_state: (B, C, H, W)
        actions: (B, T_minus_one, action_dim)
        Returns:
            pred_encs: (B, T, repr_dim)
        """
        B, T_minus_one, _ = actions.shape
        T = T_minus_one + 1
        pred_encs = []

        # Encode the initial state
        s_prev = self.encoder(init_state)  # (B, repr_dim)
        pred_encs.append(s_prev)

        hidden = None  # Initialize hidden state

        for t in range(T_minus_one):
            u_prev = actions[:, t]  # (B, action_dim)
            s_pred, hidden = self.predictor(s_prev, u_prev, hidden)  # (B, repr_dim), hidden
            pred_encs.append(s_pred)
            s_prev = s_pred  # Update for next step

        pred_encs = torch.stack(pred_encs, dim=1)  # (B, T, repr_dim)
        return pred_encs

    def variance_regularization(self, states, epsilon=1e-4):
        std = torch.sqrt(states.var(dim=0) + 1e-10)
        return torch.mean(torch.relu(epsilon - std))

    def covariance_regularization(self, states):
        if states.ndim == 3:
            states = states.view(-1, states.size(-1))
        elif states.ndim != 2:
            raise ValueError(f"Expected states to have 2 or 3 dimensions, got {states.ndim}")
        batch_size, dim = states.size()
        norm_states = states - states.mean(dim=0, keepdim=True)
        cov_matrix = (norm_states.T @ norm_states) / (batch_size - 1)
        off_diag = cov_matrix - torch.diag(torch.diag(cov_matrix))
        return torch.sum(off_diag ** 2)

    def compute_energy(self, predicted_encs, target_encs, distance_function="l2"):
        if distance_function == "l2":
            energy = torch.sum((predicted_encs - target_encs) ** 2) / (predicted_encs.size(0) * predicted_encs.size(1))
        elif distance_function == "cosine":
            cos = nn.CosineSimilarity(dim=-1)
            energy = -torch.sum(cos(predicted_encs, target_encs)) / (predicted_encs.size(0) * predicted_encs.size(1))
        else:
            raise ValueError(f"Unknown distance function: {distance_function}")
        return energy

    def train_step(self, states, actions, optimizer, momentum=0.99, distance_function="l2", add_noise=False, lambda_cov=0.5):
        """
        Perform a single training step.
        Args:
            states: (B, T, C, H, W)
            actions: (B, T_minus_one, action_dim)
            optimizer: Optimizer to update the model
            momentum: Momentum for updating the target encoder
            distance_function: "l2" or "cosine"
            add_noise: If True, add noise to the predictions (not implemented here)
            lambda_cov: Weight for covariance regularization
        Returns:
            loss.item(): The scalar loss value
            pred_encs: (B, T, repr_dim)
        """
        B, T, C, H, W = states.shape

        init_state = states[:, 0]  # (B, C, H, W)
        pred_encs = self.forward(init_state, actions)  # (B, T, repr_dim)

        target_encs = []
        for t in range(T):
            o_t = states[:, t]  # (B, C, H, W)
            s_target = self.target_encoder(o_t)  # (B, repr_dim)
            target_encs.append(s_target)
        target_encs = torch.stack(target_encs, dim=1)  # (B, T, repr_dim)

        loss = self.compute_energy(pred_encs, target_encs, distance_function)
        lambda_var = 0.1
        loss += lambda_var * self.variance_regularization(pred_encs)
        loss += lambda_cov * self.covariance_regularization(pred_encs)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)

        optimizer.step()

        # Update the target encoder with momentum
        with torch.no_grad():
            for param_q, param_k in zip(self.encoder.parameters(), self.target_encoder.parameters()):
                param_k.data = momentum * param_k.data + (1 - momentum) * param_q.data

        return loss.item(), pred_encs

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
