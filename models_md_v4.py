# models_md_v3.py

from typing import List
import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50
from torchvision.models.resnet import BasicBlock
import numpy as np

def build_mlp(layers_dims: List[int]):
    """Utility function to build an MLP."""
    layers = []
    for i in range(len(layers_dims) - 2):
        layers.append(nn.Linear(layers_dims[i], layers_dims[i + 1]))
        layers.append(nn.BatchNorm1d(layers_dims[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(layers_dims[-2], layers_dims[-1]))
    return nn.Sequential(*layers)


class Encoder(nn.Module):
    """
    Recurrent Neural Network (RNN) based Encoder.
    Replaces the previous CNN-based Encoder to process image data as sequences.
    """
    def __init__(self, output_dim=128, dropout_prob=0.0):
        super(Encoder, self).__init__()
        self.output_dim = output_dim
        self.dropout_prob = dropout_prob

        # Define RNN parameters
        self.input_size = 2  # Number of channels in input (C=2)
        self.hidden_size = 64  # Hidden size for LSTM
        self.num_layers = 2  # Number of LSTM layers

        # Define LSTM layer
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout_prob if self.num_layers > 1 else 0.0
        )

        # Fully connected layer to map hidden state to output_dim
        self.fc = nn.Linear(self.hidden_size, self.output_dim)

    def forward(self, x):
        """
        Forward pass for the RNN Encoder.

        Args:
            x: Tensor of shape [B, C, H, W]

        Returns:
            encoding: Tensor of shape [B, output_dim]
        """
        B, C, H, W = x.shape

        # Reshape and permute to [B, H*W, C] to treat spatial dimensions as sequence
        x = x.view(B, C, H * W).permute(0, 2, 1)  # [B, T, C] where T = H*W

        # Pass through LSTM
        lstm_out, (hn, cn) = self.lstm(x)  # lstm_out: [B, T, hidden_size]

        # Take the last hidden state from the last LSTM layer
        hn_last = hn[-1]  # [B, hidden_size]

        # Pass through fully connected layer to get the final encoding
        encoding = self.fc(hn_last)  # [B, output_dim]

        return encoding


class Predictor(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=512, dropout_prob=0.0):
        super(Predictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_prob)  # Add dropout layer
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.residual = nn.Linear(input_dim, output_dim)  # Residual connection

    def forward(self, s_prev, u_prev):
        x = torch.cat([s_prev, u_prev], dim=-1)
        res = self.residual(x)  # Residual connection
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)  # Apply dropout after ReLU
        x = self.fc2(x)
        return x + res


class Expander(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=512, output_dim=128, dropout_prob=0.0):
        super(Expander, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=dropout_prob)
        
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.bn2 = nn.BatchNorm1d(output_dim)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        return x


class JEPA_Model(nn.Module):
    def __init__(self, device="cuda", repr_dim=128, action_dim=2, dropout_prob=0.0, margin=1.0):
        super(JEPA_Model, self).__init__()
        self.device = device
        self.repr_dim = repr_dim
        self.action_dim = action_dim
        self.dropout_prob = dropout_prob
        self.margin = margin  # Margin for contrastive loss

        self.encoder = Encoder(output_dim=repr_dim, dropout_prob=dropout_prob).to(device)

        self.expander = Expander(input_dim=repr_dim, hidden_dim=512, output_dim=repr_dim, dropout_prob=dropout_prob).to(device)

        self.predictor = Predictor(input_dim=repr_dim + action_dim, output_dim=repr_dim, hidden_dim=512, dropout_prob=dropout_prob).to(device)

        self.target_encoder = Encoder(output_dim=repr_dim).to(device)

        # BatchNorm for embeddings
        self.batch_norm = nn.BatchNorm1d(repr_dim)

        # Load encoder weights into target encoder
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
        s_prev = self.expander(s_prev)  # Pass encoder output through expander
        s_prev = self.batch_norm(s_prev)  # Apply BatchNorm to embeddings
        pred_encs.append(s_prev)

        for t in range(T_minus_one):
            u_prev = actions[:, t]
            s_pred = self.predictor(s_prev, u_prev)
            s_pred = self.batch_norm(s_pred)  # Apply BatchNorm to embeddings
            pred_encs.append(s_pred)
            s_prev = s_pred

        pred_encs = torch.stack(pred_encs, dim=1)
        return pred_encs

    def variance_regularization(self, pred_encs, target_encs, epsilon=1e-4, min_variance=1.0):
        # Ensure consistent 3D shape [B, T, D]
        if pred_encs.ndim == 2:
            pred_encs = pred_encs.unsqueeze(0)  # Add batch dimension
        if target_encs.ndim == 2:
            target_encs = target_encs.unsqueeze(0)  # Add batch dimension

        B, T, D = pred_encs.shape

        # Reshape to 2D for computation: [B*T, D]
        pred_encs = pred_encs.reshape(-1, D)
        target_encs = target_encs.reshape(-1, D)

        # Apply BatchNorm
        pred_encs = self.batch_norm(pred_encs)
        target_encs = self.batch_norm(target_encs)

        # Compute variance separately
        pred_std_x = torch.sqrt(pred_encs.var(dim=0) + epsilon)
        target_std_x = torch.sqrt(target_encs.var(dim=0) + epsilon)

        # Penalize low variance for both predicted and target encodings
        pred_variance_loss = torch.mean(torch.relu(min_variance - pred_std_x))
        target_variance_loss = torch.mean(torch.relu(min_variance - target_std_x))

        # Sum the variance losses
        variance_loss = pred_variance_loss + target_variance_loss
        
        return variance_loss

    def covariance_regularization(self, pred_encs, target_encs, epsilon=1e-4):
        def off_diagonal(matrix):
            # Helper to zero out diagonal and retain off-diagonal elements
            return matrix - torch.diag_embed(torch.diagonal(matrix))

        # Ensure consistent 3D shape [B, T, D]
        if pred_encs.ndim == 2:
            pred_encs = pred_encs.unsqueeze(0)  # Add batch dimension
        if target_encs.ndim == 2:
            target_encs = target_encs.unsqueeze(0)  # Add batch dimension

        B, T, D = pred_encs.shape

        # Reshape to 2D for computation: [B*T, D]
        pred_encs = pred_encs.reshape(-1, D)
        target_encs = target_encs.reshape(-1, D)

        # Apply BatchNorm
        pred_encs = self.batch_norm(pred_encs)
        target_encs = self.batch_norm(target_encs)

        # Center embeddings (zero mean)
        pred_encs = pred_encs - pred_encs.mean(dim=0)
        target_encs = target_encs - target_encs.mean(dim=0)

        # Compute covariance matrices
        cov_pred = torch.matmul(pred_encs.T, pred_encs) / (pred_encs.size(0) - 1)  # Covariance for predictions
        cov_target = torch.matmul(target_encs.T, target_encs) / (target_encs.size(0) - 1)  # Covariance for targets

        # Compute off-diagonal penalties
        off_diag_pred = off_diagonal(cov_pred).pow(2).sum() / pred_encs.size(1)  # Normalize by embedding dim
        off_diag_target = off_diagonal(cov_target).pow(2).sum() / target_encs.size(1)  # Normalize by embedding dim

        # Combine covariance loss for both predicted and target embeddings
        cov_loss = off_diag_pred + off_diag_target
        return cov_loss

    def compute_energy(self, predicted_encs, target_encs, distance_function="l2"):
        """
        Compute the energy function.

        Args:
            predicted_encs: [B, T, D] - Predicted latent representations
            target_encs: [B, T, D] - Target latent representations
            distance_function: str - Distance metric ("l2" or "cosine")

        Returns:
            energy: Scalar energy value
        """
        if distance_function == "l2":
            energy = torch.sum((predicted_encs - target_encs) ** 2) / (predicted_encs.size(0) * predicted_encs.size(1))
        elif distance_function == "cosine":
            cos = nn.CosineSimilarity(dim=-1)
            energy = -torch.sum(cos(predicted_encs, target_encs)) / (predicted_encs.size(0) * predicted_encs.size(1))
        else:
            raise ValueError(f"Unknown distance function: {distance_function}")
        return energy

    def compute_negative_energy(self, predicted_encs, negative_encs, distance_function="l2"):
        """
        Compute the energy for negative (bad) pairs.

        Args:
            predicted_encs: [B, T, D] - Predicted latent representations
            negative_encs: [B, T, D] - Negative latent representations
            distance_function: str - Distance metric ("l2" or "cosine")

        Returns:
            negative_energy: Scalar energy value for negative pairs
        """
        if distance_function == "l2":
            negative_energy = torch.sum((predicted_encs - negative_encs) ** 2) / (predicted_encs.size(0) * predicted_encs.size(1))
        elif distance_function == "cosine":
            cos = nn.CosineSimilarity(dim=-1)
            negative_energy = -torch.sum(cos(predicted_encs, negative_encs)) / (predicted_encs.size(0) * predicted_encs.size(1))
        else:
            raise ValueError(f"Unknown distance function: {distance_function}")
        return negative_energy

    def train_step(self, 
                   states, 
                   actions, 
                   optimizer,
                   scheduler, 
                   momentum=0.99, 
                   distance_function="l2", 
                   lambda_energy=1.0, 
                   lambda_var=1.0, 
                   lambda_cov=1.0,
                   lambda_contrastive=0.1,
                   lambda_negative=0.5,  # Weight for negative loss
                   margin=1.0,           # Margin for contrastive loss
                   *args, **kwargs):
        """
        Perform a single training step.

        Args:
            states: [B, T, Ch, H, W]
            actions: [B, T-1, action_dim]
            optimizer: Optimizer
            momentum: Momentum for target encoder update
            distance_function: Distance metric for energy computation
        """
        B, T, C, H, W = states.shape
        init_state = states[:, 0]
        pred_encs = self.forward(init_state, actions)  # Predicted embeddings

        # Generate target embeddings using the target encoder
        target_encs = []
        for t in range(T):
            o_t = states[:, t]
            s_target = self.target_encoder(o_t)
            target_encs.append(s_target)
        target_encs = torch.stack(target_encs, dim=1)

        # Generate negative embeddings by shuffling target encodings
        negative_encs = target_encs[torch.randperm(B)].to(self.device)

        # Compute the loss function
        loss, energy, var, cov, contrastive, negative = self.compute_loss(
            pred_encs, 
            target_encs, 
            negative_encs, 
            distance_function, 
            lambda_energy, 
            lambda_var, 
            lambda_cov, 
            lambda_contrastive, 
            lambda_negative, 
            margin
        )

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        # Update target encoder using momentum
        with torch.no_grad():
            for param_q, param_k in zip(self.encoder.parameters(), self.target_encoder.parameters()):
                param_k.data = momentum * param_k.data + (1 - momentum) * param_q.data

        return loss.item(), energy.item(), var.item(), cov.item(), contrastive.item(), negative.item()

    def contrastive_loss(self, pred_encs, target_encs, margin=1.0):
        """
        Compute contrastive loss for embeddings.

        Args:
            pred_encs: [B, T, D] - Predicted embeddings
            target_encs: [B, T, D] - Target embeddings
            margin: float - Margin for contrastive loss

        Returns:
            loss: Contrastive loss value
        """
        # Flatten sequences for batch-wise comparison
        pred_encs = pred_encs.reshape(-1, pred_encs.size(-1))  # [B*T, D]
        target_encs = target_encs.reshape(-1, target_encs.size(-1))  # [B*T, D]

        # Pairwise L2 distances
        distances = torch.cdist(pred_encs, target_encs, p=2)  # [B*T, B*T]

        # Positive pairs are on the diagonal
        positive_pairs = distances.diag()  # [B*T]

        # Negative pairs are all off-diagonal elements
        negative_pairs = distances + torch.eye(distances.size(0), device=distances.device) * 1e9
        closest_negative_pairs, _ = torch.min(negative_pairs, dim=1)  # Nearest negatives

        # Contrastive loss with margin
        loss = torch.mean(torch.relu(margin - positive_pairs + closest_negative_pairs))
        return loss

    def compute_loss(self, 
                 pred_encs, 
                 target_encs, 
                 negative_encs,
                 distance_function="l2", 
                 lambda_energy=1.0, 
                 lambda_var=1.0, 
                 lambda_cov=1.0, 
                 lambda_contrastive=0.1, 
                 lambda_negative=0.5,
                 margin=1.0,
                 debug=False, 
                 min_variance=1.0,
                 *args, **kwargs):
        """
        Compute the loss function with contrastive loss and negative sampling.

        Args:
            pred_encs: [B, T, D] - Predicted embeddings
            target_encs: [B, T, D] - Target embeddings
            negative_encs: [B, T, D] - Negative embeddings
            distance_function: str - Distance metric ("l2" or "cosine")
            lambda_energy: float - Weight for energy loss
            lambda_var: float - Weight for variance regularization
            lambda_cov: float - Weight for covariance regularization
            lambda_contrastive: float - Weight for contrastive loss
            lambda_negative: float - Weight for negative sampling loss
            margin: float - Margin for contrastive loss
            debug: bool - If True, return individual loss components

        Returns:
            loss: Total loss
            energy: Energy loss
            var: Variance regularization loss
            cov: Covariance regularization loss
            contrastive: Contrastive loss
            negative: Negative sampling loss
        """
        # Compute the energy function (distance between predicted and target states)
        energy = lambda_energy * self.compute_energy(pred_encs, target_encs, distance_function)

        # Add regularization terms
        var = lambda_var * self.variance_regularization(pred_encs, target_encs, min_variance=min_variance)
        cov = lambda_cov * self.covariance_regularization(pred_encs, target_encs)

        # Compute contrastive loss
        contrastive = lambda_contrastive * self.contrastive_loss(pred_encs, target_encs, margin=margin)

        # Compute negative sampling loss
        negative = lambda_negative * self.compute_energy(pred_encs, negative_encs, distance_function)

        # Total loss
        loss = energy + var + cov + contrastive + negative
        if not debug:
            return loss, energy, var, cov, contrastive, negative
        else:
            return (loss, energy, var, cov, contrastive, negative)

