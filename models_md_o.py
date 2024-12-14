# models_md_o.py

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

class SimpleEncoder(nn.Module):
    def __init__(self, input_dim=8450, hidden_dim=128, output_dim=256, dropout=0.1):
        """
        A simple encoder using MLP layers.
        
        Args:
            input_dim (int): Number of input features after flattening.
            hidden_dim (int): Number of neurons in the hidden layer.
            output_dim (int): Dimension of the output representation.
            dropout (float): Dropout probability.
        """
        super(SimpleEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input: (B, C, H, W) -> (B, C*H*W)
        x = self.encoder(x)
        return x

class SimplePredictor(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128, dropout=0.1):
        """
        A simple predictor using MLP layers with a residual connection.
        
        Args:
            input_dim (int): Number of input features (representation + action).
            output_dim (int): Dimension of the output representation.
            hidden_dim (int): Number of neurons in the hidden layer.
            dropout (float): Dropout probability.
        """
        super(SimplePredictor, self).__init__()
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        self.residual = nn.Linear(input_dim, output_dim)  # Residual connection

    def forward(self, s_prev, u_prev):
        """
        Forward pass for the predictor.
        
        Args:
            s_prev (torch.Tensor): Previous state representation.
            u_prev (torch.Tensor): Previous action.

        Returns:
            torch.Tensor: Predicted next state representation.
        """
        x = torch.cat([s_prev, u_prev], dim=-1)  # Concatenate along feature dimension
        res = self.residual(x)  # Residual connection
        x = self.predictor(x)
        return x + res

class JEPA_Model(nn.Module):
    def __init__(self, device="cuda", repr_dim=256, action_dim=2, dropout=0.1):
        """
        Joint Embedding Predictive Architecture (JEPA) model.
        
        Args:
            device (str): Device to run the model on.
            repr_dim (int): Dimension of the representation space.
            action_dim (int): Dimension of the action space.
            dropout (float): Dropout probability.
        """
        super(JEPA_Model, self).__init__()
        self.device = device
        self.repr_dim = repr_dim
        self.action_dim = action_dim
        self.encoder = SimpleEncoder(input_dim=8450, hidden_dim=128, output_dim=repr_dim, dropout=dropout).to(device)
        self.predictor = SimplePredictor(input_dim=repr_dim + action_dim, output_dim=repr_dim, hidden_dim=128, dropout=dropout).to(device)
        self.target_encoder = SimpleEncoder(input_dim=8450, hidden_dim=128, output_dim=repr_dim, dropout=dropout).to(device)
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
        # Compute per-sample energy
        if distance_function == "l2":
            energy = ((predicted_encs - target_encs) ** 2).sum(dim=-1).mean(dim=-1)  # (B)
        elif distance_function == "cosine":
            energy = 1 - torch.cosine_similarity(predicted_encs, target_encs, dim=-1)  # (B)
        else:
            raise ValueError(f"Unsupported distance function: {distance_function}")
        return energy

    def compute_energy_regularization(self, energy, target_average=1.0, lambda_reg=0.5):
        """
        Compute regularization loss to prevent energy collapse.
        Encourages the average energy to be close to target_average.
        
        Args:
            energy (torch.Tensor): Energy values for each sample.
            target_average (float): Target average energy.
            lambda_reg (float): Weight for the regularization term.
        
        Returns:
            torch.Tensor: Regularization loss.
        """
        reg_loss = torch.mean(torch.relu(target_average - energy))
        return lambda_reg * reg_loss

    def train_step(self, states, actions, labels, optimizer, momentum=0.99, distance_function="l2", add_noise=False, lambda_cov=0.5, target_average=1.0):
        """
        Perform a single training step.
        
        Args:
            states (torch.Tensor): Input states.
            actions (torch.Tensor): Actions taken.
            labels (torch.Tensor): Binary labels indicating good (0) or bad (1) inputs.
            optimizer (torch.optim.Optimizer): Optimizer.
            momentum (float): Momentum for the target encoder update.
            distance_function (str): Distance function to compute energy ('l2' or 'cosine').
            add_noise (bool): Whether to add noise to the states.
            lambda_cov (float): Weight for covariance regularization.
            target_average (float): Target average energy for regularization.
        
        Returns:
            Tuple[float, float, float, torch.Tensor]: Energy loss, energy regularization loss, total loss, predicted encodings.
        """
        B, T, C, H, W = states.shape

        # Add noise if requested
        if add_noise:
            states = states + 0.01 * torch.randn_like(states)

        # Random horizontal flip with 50% chance for data augmentation
        if torch.rand(1).item() < 0.5:
            states = torch.flip(states, dims=[3])  # Assuming H is dim 3

        init_state = states[:, 0]
        pred_encs = self.forward(init_state, actions)

        target_encs = []
        for t in range(T):
            o_t = states[:, t]
            s_target = self.target_encoder(o_t)
            target_encs.append(s_target)
        target_encs = torch.stack(target_encs, dim=1)

        # Compute energy loss
        energy = self.compute_energy(pred_encs, target_encs, distance_function)  # (B)

        # Compute energy-based regularization
        energy_reg = self.compute_energy_regularization(energy, target_average=target_average)

        # Total loss
        lambda_var = 0.1
        loss = energy.mean() + energy_reg + lambda_var * self.variance_regularization(pred_encs) + lambda_cov * self.covariance_regularization(pred_encs)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)

        with torch.no_grad():
            for param_q, param_k in zip(self.encoder.parameters(), self.target_encoder.parameters()):
                param_k.data = momentum * param_k.data + (1 - momentum) * param_q.data

        # Return energy loss and regularization loss
        return energy.mean().item(), energy_reg.item(), loss.item(), pred_encs

class Prober(torch.nn.Module):
    def __init__(
        self,
        embedding: int,
        arch: str,
        output_shape: List[int],
    ):
        """
        Prober module for evaluating representations.
        
        Args:
            embedding (int): Dimension of the embedding.
            arch (str): Architecture specification (e.g., "256-128").
            output_shape (List[int]): Shape of the output.
        """
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
