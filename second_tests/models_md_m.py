# models_md_h.py
from typing import List
import torch
import torch.nn as nn
import numpy as np

class Encoder(nn.Module):
    def __init__(self, output_dim=256, input_channels=2):
        super(Encoder, self).__init__()
        # Simple CNN architecture inspired by BYOL (you can adjust as needed)
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, output_dim),
            nn.LayerNorm(output_dim)
        )
    
    def forward(self, x):
        return self.encoder(x)

class Predictor(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=512, output_dim=256, dropout=0.3):
        super(Predictor, self).__init__()
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.predictor(x)

class JEPA_Model(nn.Module):
    def __init__(self, device="cuda", repr_dim=256, action_dim=2, dropout=0.3):
        super(JEPA_Model, self).__init__()
        self.device = device
        self.repr_dim = repr_dim
        self.action_dim = action_dim
        self.encoder = Encoder(output_dim=repr_dim, input_channels=2).to(device)
        self.predictor = Predictor(input_dim=repr_dim + action_dim, hidden_dim=512, output_dim=repr_dim, dropout=dropout).to(device)
        self.target_encoder = Encoder(output_dim=repr_dim, input_channels=2).to(device)
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        for param in self.target_encoder.parameters():
            param.requires_grad = False

    def forward(self, init_state, actions):
        # Assuming actions shape: (B, T-1, action_dim)
        B, T_minus_one, _ = actions.shape
        T = T_minus_one + 1
        pred_encs = []

        s_prev = self.encoder(init_state)
        pred_encs.append(s_prev)

        for t in range(T_minus_one):
            u_prev = actions[:, t]
            x = torch.cat([s_prev, u_prev], dim=-1)
            s_pred = self.predictor(x)
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
        # Compute per-sample energy
        energy = ((predicted_encs - target_encs) ** 2).sum(dim=-1).mean(dim=-1)  # (B)
        return energy

    def compute_margin_contrastive_loss(self, energy, labels, margin=1.0):
        """
        Compute margin-based contrastive loss.
        Args:
            energy (torch.Tensor): Energy loss per sample (B)
            labels (torch.Tensor): Binary labels (B), 0 for good, 1 for bad
            margin (float): Margin for contrastive loss
        Returns:
            torch.Tensor: Contrastive loss
        """
        # For good inputs (label=0), minimize energy
        loss_pos = (1 - labels) * energy  # (B)

        # For bad inputs (label=1), ensure energy > margin
        loss_neg = labels * torch.relu(margin - energy)  # (B)

        # Total contrastive loss
        contrastive_loss = loss_pos + loss_neg  # (B)
        return contrastive_loss.mean()

    def train_step(self, states, actions, labels, optimizer, momentum=0.99, distance_function="l2", add_noise=False, lambda_cov=0.5, margin=1.0):
        """
        Modified train_step to handle margin-based contrastive loss.
        Args:
            labels (torch.Tensor): Binary labels indicating good (0) or bad (1) inputs.
        """
        B, T, C, H, W = states.shape

        # Add noise if requested
        if add_noise:
            states = states + 0.01 * torch.randn_like(states)

        # Random horizontal flip with 50% chance for data augmentation
        if torch.rand(1).item() < 0.5:
            states = torch.flip(states, dims=[3])

        init_state = states[:, 0]
        pred_encs = self.forward(init_state, actions)  # (B, T, D)

        target_encs = []
        for t in range(T):
            o_t = states[:, t]
            s_target = self.target_encoder(o_t)
            target_encs.append(s_target)
        target_encs = torch.stack(target_encs, dim=1)  # (B, T, D)

        # Compute energy loss
        energy = self.compute_energy(pred_encs, target_encs, distance_function)  # (B)

        # Compute margin-based contrastive loss
        contrastive_loss = self.compute_margin_contrastive_loss(energy, labels, margin)

        # Total loss with regularizations
        lambda_var = 0.1
        loss = contrastive_loss + lambda_var * self.variance_regularization(pred_encs) + lambda_cov * self.covariance_regularization(pred_encs)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)

        with torch.no_grad():
            for param_q, param_k in zip(self.encoder.parameters(), self.target_encoder.parameters()):
                param_k.data = momentum * param_k.data + (1 - momentum) * param_q.data

        # Return energy loss and contrastive loss
        return energy.mean().item(), contrastive_loss.item(), loss.item(), pred_encs
