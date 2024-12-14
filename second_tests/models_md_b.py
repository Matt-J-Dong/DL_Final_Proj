from typing import List
import torch
import torch.nn as nn
from torchvision.models import resnet50

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

class Encoder(nn.Module):
    def __init__(self, output_dim=256, input_channels=2):
        super(Encoder, self).__init__()
        # Load ResNet-50 without pretrained weights
        resnet = resnet50(pretrained=False)
        # Modify the first convolutional layer to accept the required input channels
        resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Remove the final fully connected layer and average pooling
        self.features = nn.Sequential(*list(resnet.children())[:-2])  # Exclude avgpool and fc
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
        self.fc = nn.Linear(resnet.fc.in_features, output_dim)  # Replace with custom FC layer

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class Predictor(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=512, dropout=0.3):  # Increased dropout to 0.3
        super(Predictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.residual = nn.Linear(input_dim, output_dim)  # Residual connection

    def forward(self, s_prev, u_prev):
        x = torch.cat([s_prev, u_prev], dim=-1)
        res = self.residual(x)  # Residual connection
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)  # Apply dropout after activation
        x = self.fc2(x)
        return x + res

class JEPA_Model(nn.Module):
    def __init__(self, device="cuda", repr_dim=256, action_dim=2, dropout=0.3):  # Pass increased dropout
        super(JEPA_Model, self).__init__()
        self.device = device
        self.repr_dim = repr_dim
        self.action_dim = action_dim
        self.encoder = Encoder(output_dim=repr_dim, input_channels=2).to(device)
        self.predictor = Predictor(input_dim=repr_dim + action_dim, output_dim=repr_dim, dropout=dropout).to(device)
        self.target_encoder = Encoder(output_dim=repr_dim, input_channels=2).to(device)
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        for param in self.target_encoder.parameters():
            param.requires_grad = False

    def forward(self, init_state, actions):
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

        pred_encs = torch.stack(pred_encs, dim=1)
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
        B, T, C, H, W = states.shape

        # Add noise
        if add_noise:
            states = states + 0.01 * torch.randn_like(states)

        # (Method 4) Random horizontal flip augmentation:
        # With 50% chance, flip the states horizontally. This simulates data augmentation.
        if torch.rand(1).item() < 0.5:
            states = torch.flip(states, dims=[3])  # flip width dimension

        init_state = states[:, 0]
        pred_encs = self.forward(init_state, actions)

        target_encs = []
        for t in range(T):
            o_t = states[:, t]
            s_target = self.target_encoder(o_t)
            target_encs.append(s_target)
        target_encs = torch.stack(target_encs, dim=1)

        loss = self.compute_energy(pred_encs, target_encs, distance_function)
        lambda_var = 0.1
        loss += lambda_var * self.variance_regularization(pred_encs)
        loss += lambda_cov * self.covariance_regularization(pred_encs)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        # optimizer.step is called outside

        with torch.no_grad():
            for param_q, param_k in zip(self.encoder.parameters(), self.target_encoder.parameters()):
                param_k.data = momentum * param_k.data + (1 - momentum) * param_q.data

        return loss.item()