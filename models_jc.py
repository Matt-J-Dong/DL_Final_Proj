from typing import List
import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50


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
    def __init__(self, output_dim=256, input_channels=2, dropout_prob=0.1):
        super(Encoder, self).__init__()
        # Load ResNet-18 without pretrained weights
        resnet = resnet18(pretrained=False)

        # Modify the first convolutional layer to accept the required input channels
        resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Remove the final fully connected layer and average pooling
        self.features = nn.Sequential(*list(resnet.children())[:-2])  # Exclude avgpool and fc
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
        self.fc = nn.Linear(resnet.fc.in_features, output_dim)  # Replace with custom FC layer
        self.dropout = nn.Dropout(p=dropout_prob)  # Add dropout for regularization

    def forward(self, x):
        x = self.features(x)  # Extract features
        x = self.avgpool(x)  # Global average pooling
        x = torch.flatten(x, 1)  # Flatten to 1D
        x = self.fc(x)  # Fully connected layer for output
        x = self.dropout(x)  # Apply dropout
        return x


class Predictor(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=512, dropout_prob=0.1):
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



class JEPA_Model(nn.Module):
    def __init__(self, device="cuda", repr_dim=256, action_dim=2, dropout_prob=0.1):
        super(JEPA_Model, self).__init__()
        self.device = device
        self.encoder = Encoder(output_dim=repr_dim, input_channels=2, dropout_prob=dropout_prob).to(device)
        self.predictor = Predictor(input_dim=repr_dim + action_dim, output_dim=repr_dim, dropout_prob=dropout_prob).to(device)
        self.target_encoder = Encoder(output_dim=repr_dim, input_channels=2).to(device)
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

        pred_encs = torch.stack(pred_encs, dim=1)
        return pred_encs

    def variance_regularization(self, states, epsilon=1e-4):
        """
        Prevents representation collapse by enforcing variance in each embedding dimension.

        Args:
            states: [B, D] or [B, T, D] Predicted embeddings.
            epsilon: Small positive constant to stabilize variance calculation.

        Returns:
            Regularization loss value.
        """
        if states.ndim == 3:  # [B, T, D]
            states = states.view(-1, states.size(-1))  # Flatten to [B*T, D]
        
        # Compute standard deviation across the batch dimension
        std_x = torch.sqrt(states.var(dim=0, unbiased=False) + epsilon)
        
        # Penalize dimensions with standard deviation below 1.0
        variance_loss = torch.mean(torch.relu(1.0 - std_x))
        
        return variance_loss


    def covariance_regularization(self, states):
        """
        Reduces redundancy by decorrelating dimensions of embeddings.

        Args:
            states: [B, D] or [B, T, D] Predicted embeddings.

        Returns:
            Regularization loss value.
        """
        if states.ndim == 3:  # [B, T, D]
            states = states.view(-1, states.size(-1))  # Flatten to [B*T, D]
        
        batch_size, dim = states.size()
        
        # Center the embeddings
        states = states - states.mean(dim=0, keepdim=True)
        
        # Compute the covariance matrix
        cov_matrix = (states.T @ states) / (batch_size - 1)
        
        # Extract off-diagonal elements
        cov_loss = cov_matrix - torch.diag(torch.diag(cov_matrix))  # Remove diagonal
        cov_loss = (cov_loss ** 2).sum() / dim  # Penalize off-diagonal elements
        
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

    def train_step(self, states, actions, optimizer, momentum=0.99, distance_function="l2"):
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

        # Compute the loss function
        lambda_energy, lambda_var, lambda_cov = 6.25, 6.25, 0.25  # Tunable hyperparameters
        loss = self.compute_loss(pred_encs, target_encs, distance_function, lambda_energy, lambda_var, lambda_cov)


        # Backpropagation
        optimizer.zero_grad()
        loss.backward()

        max_grad_norm = 0.5  # Set the maximum norm for gradients
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)

        optimizer.step()

        # Update target encoder using momentum
        with torch.no_grad():
            for param_q, param_k in zip(self.encoder.parameters(), self.target_encoder.parameters()):
                param_k.data = momentum * param_k.data + (1 - momentum) * param_q.data

        return loss.item()
    
    def compute_loss(self, pred_encs, target_encs, distance_function="l2", lambda_energy=1.0, lambda_var=1.0, lambda_cov=1.0, debug=False):
        """
        Compute the loss function.
        """
        # Compute the energy function (distance between predicted and target states)
        energy = lambda_energy * self.compute_energy(pred_encs, target_encs, distance_function)

        # Add regularization terms
        var = lambda_var * self.variance_regularization(pred_encs)
        cov = lambda_cov * self.covariance_regularization(pred_encs)
        loss = energy + var + cov
        return loss if not debug else (loss, energy, var, cov)
