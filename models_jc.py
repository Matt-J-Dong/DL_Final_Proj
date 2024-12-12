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
    def __init__(self, input_dim, output_dim, hidden_dim=1024, dropout_prob=0.1):
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
    def __init__(self, input_dim=256, hidden_dim=1024, output_dim=256, dropout_prob=0.1):
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
    def __init__(self, device="cuda", repr_dim=256, action_dim=2, dropout_prob=0.1):
        super(JEPA_Model, self).__init__()
        self.device = device
        self.encoder = Encoder(output_dim=repr_dim, input_channels=2, dropout_prob=dropout_prob).to(device)
        
        # Add the expander after the encoder
        self.expander = Expander(input_dim=repr_dim, hidden_dim=1024, output_dim=repr_dim, dropout_prob=dropout_prob).to(device)
        
        self.predictor = Predictor(input_dim=repr_dim + action_dim, output_dim=repr_dim, hidden_dim=1024, dropout_prob=dropout_prob).to(device)
        
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
        s_prev = self.expander(s_prev)  # Pass encoder output through expander
        pred_encs.append(s_prev)

        for t in range(T_minus_one):
            u_prev = actions[:, t]
            s_pred = self.predictor(s_prev, u_prev)
            pred_encs.append(s_pred)
            s_prev = s_pred

        pred_encs = torch.stack(pred_encs, dim=1)
        return pred_encs

    def variance_regularization(self, states, epsilon=1e-4, min_variance=1.0):
        if states.ndim == 3:
            states = states.view(-1, states.size(-1))
        
        std_x = torch.sqrt(states.var(dim=0) + epsilon)
        
        # Modify with a minimum variance threshold
        min_variance = min_variance
        variance_loss = torch.mean(torch.relu(min_variance - std_x))

        
        return variance_loss

    def covariance_regularization(self, states, epsilon=1e-4):
        if states.ndim == 3:
            states = states.view(-1, states.size(-1))
        
        # Center the states
        states = states - states.mean(dim=0)
        
        # Compute covariance matrix
        cov_matrix = torch.matmul(states.t(), states) / (states.size(0) - 1)
        
        # Remove diagonal (we only care about off-diagonal correlations)
        off_diagonal = cov_matrix.clone()
        torch.diagonal(off_diagonal)[:] = 0
        
        # Compute off-diagonal covariance loss
        # Use a more aggressive penalty for off-diagonal elements
        cov_loss = torch.sum(off_diagonal.pow(2))
        
        # Normalize by the number of off-diagonal elements
        dim = states.size(1)
        num_off_diagonal = dim * (dim - 1)
        
        # Add small epsilon to prevent division by zero
        cov_loss = cov_loss / (num_off_diagonal + epsilon)
        
        # Additional clipping to prevent extreme values
        return torch.clamp(cov_loss, min=epsilon, max=5.0)




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

    def train_step(self, 
                   states, 
                   actions, 
                   optimizer, 
                   momentum=0.99, 
                   distance_function="l2", 
                   lambda_energy=1.0, 
                   lambda_var=1.0, 
                   lambda_cov=1.0,
                   debug=False,
                   max_grad_norm=0.5,
                   min_variance = 1.0):
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
        if not debug:
            loss = self.compute_loss(pred_encs, target_encs, distance_function, lambda_energy, lambda_var, lambda_cov, min_variance=min_variance)
        else:
            loss, energy, var, cov = self.compute_loss(pred_encs, target_encs, distance_function, lambda_energy, lambda_var, lambda_cov, min_variance=min_variance, debug=True)


        # Backpropagation
        optimizer.zero_grad()
        loss.backward()

        # max_grad_norm = 0.1  # Set the maximum norm for gradients
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)

        optimizer.step()

        # Update target encoder using momentum
        with torch.no_grad():
            for param_q, param_k in zip(self.encoder.parameters(), self.target_encoder.parameters()):
                param_k.data = momentum * param_k.data + (1 - momentum) * param_q.data

        return loss.item() if not debug else (loss.item(), energy.item(), var.item(), cov.item())
    

    def contrastive_loss(pred_encs, target_encs, margin=1.0):
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
        pred_encs = pred_encs.view(-1, pred_encs.size(-1))  # [B*T, D]
        target_encs = target_encs.view(-1, target_encs.size(-1))  # [B*T, D]

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
                 distance_function="l2", 
                 lambda_energy=1.0, 
                 lambda_var=1.0, 
                 lambda_cov=1.0, 
                 lambda_contrastive=0.1, 
                 margin=1.0, 
                 debug=False, 
                 min_variance=1.0):
        """
        Compute the loss function with contrastive loss added.
        """
        # Compute the energy function (distance between predicted and target states)
        energy = lambda_energy * self.compute_energy(pred_encs, target_encs, distance_function)

        # Add regularization terms
        var = lambda_var * self.variance_regularization(pred_encs, min_variance=min_variance)
        cov = lambda_cov * self.covariance_regularization(pred_encs)

        # Compute contrastive loss
        contrastive = lambda_contrastive * contrastive_loss(pred_encs, target_encs, margin=margin)

        # Total loss
        loss = energy + var + cov + contrastive
        if not debug:
            return loss
        else:
            return (loss, energy, var, cov, contrastive)

