from typing import List
import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50
from torchvision.models.resnet import BasicBlock
import torch.nn.functional as F


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
    def __init__(self, output_dim=256, dropout_prob=0.1):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual block 1
        self.res_block1_conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.res_block1_bn1 = nn.BatchNorm2d(64)
        self.res_block1_conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.res_block1_bn2 = nn.BatchNorm2d(64)

        # Residual block 2
        self.res_block2_conv1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.res_block2_bn1 = nn.BatchNorm2d(128)
        self.res_block2_conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.res_block2_bn2 = nn.BatchNorm2d(128)
        self.res_block2_downsample = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(128)
        )

        self.dropout = nn.Dropout(p=dropout_prob)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, output_dim)

    def forward(self, x):
        # Initial convolution and pooling
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Residual block 1
        identity = x
        out = self.res_block1_conv1(x)
        out = self.res_block1_bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.res_block1_conv2(out)
        out = self.res_block1_bn2(out)
        out += identity  # Residual connection
        x = self.relu(out)

        # Residual block 2
        identity = self.res_block2_downsample(x)  # Downsample identity
        out = self.res_block2_conv1(x)
        out = self.res_block2_bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.res_block2_conv2(out)
        out = self.res_block2_bn2(out)
        out += identity  # Residual connection
        x = self.relu(out)

        # Global average pooling and projection
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x




class Predictor(nn.Module):
    def __init__(self, input_dim=258, output_dim=256, hidden_dim=512):
        super(Predictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

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
    def __init__(self, device='cuda', repr_dim=256,action_dim=2, dropout_prob=0.1):

        self.device = device
        self.repr_dim = repr_dim
        self.action_dim = action_dim
        self.dropout_prob = dropout_prob

        super(JEPA_Model, self).__init__()
        self.encoder = Encoder(output_dim=repr_dim, dropout_prob=dropout_prob).to(device)
        self.predictor = Predictor(input_dim=repr_dim + action_dim, output_dim=repr_dim).to(device)
        self.target_encoder = Encoder(output_dim=repr_dim).to(device)

        # Initialize target encoder with online encoder weights
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        for param in self.target_encoder.parameters():
            param.requires_grad = False

    def forward(self, init_state, actions):
        """
        Args:
            init_state: Initial observation (B, 2, H, W)
            actions: Sequence of actions (B, T-1, 2)

        Returns:
            pred_encs: Predicted embeddings (B, T, repr_dim)
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

    def update_target_encoder(self, momentum=0.99):
        """
        Update target encoder with exponential moving average (EMA).
        """
        with torch.no_grad():
            for param_q, param_k in zip(self.encoder.parameters(), self.target_encoder.parameters()):
                param_k.data = momentum * param_k.data + (1 - momentum) * param_q.data


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
    
    def compute_loss(self, pred_encs, target_encs):
        """
        Compute the L2 distance loss between predicted and target embeddings.

        Args:
            pred_encs: Predicted embeddings (B, T, D).
            target_encs: Target embeddings (B, T, D).

        Returns:
            loss: Mean L2 distance between predicted and target embeddings.
        """
        # Flatten embeddings to compute loss across all timesteps
        pred_encs_flat = pred_encs.view(-1, pred_encs.size(-1))  # [B*T, D]
        target_encs_flat = target_encs.view(-1, target_encs.size(-1))  # [B*T, D]

        # Compute L2 distance
        l2_distance = torch.norm(pred_encs_flat - target_encs_flat, p=2, dim=1)  # [B*T]

        # Return mean L2 distance as loss
        return l2_distance.mean()

    
    def compute_variance(self, pred_encs):
        """
        Compute the mean variance of predicted embeddings across dimensions.

        Args:
            pred_encs: Predicted embeddings (B, T, D).

        Returns:
            variance: Mean variance across all dimensions as a scalar.
        """
        B, T, D = pred_encs.shape
        pred_encs_flat = pred_encs.view(-1, D)  # Flatten to [B*T, D]
        variance = pred_encs_flat.var(dim=0, unbiased=False)  # Per-dimension variance [D]
        return variance.mean()  # Scalar: mean variance across dimensions

    

    def compute_dimensional_covariance(self, pred_encs):
        """
        Compute the mean off-diagonal covariance of predicted embeddings.

        Args:
            pred_encs: Predicted embeddings of shape (B, T, D).

        Returns:
            covariance: Mean off-diagonal covariance as a scalar.
        """
        B, T, D = pred_encs.shape
        pred_encs_flat = pred_encs.view(-1, D)  # Flatten to [B*T, D]

        # Compute mean for centering
        mean = pred_encs_flat.mean(dim=0, keepdim=True)  # Shape: (1, D)

        # Center embeddings
        centered_encs = pred_encs_flat - mean

        # Compute covariance matrix
        cov_matrix = torch.matmul(centered_encs.T, centered_encs) / (pred_encs_flat.size(0) - 1)  # [D, D]

        # Extract off-diagonal elements
        off_diag_cov = cov_matrix - torch.diag(torch.diag(cov_matrix))  # Zero out diagonal

        # Return mean of off-diagonal elements as a scalar
        return off_diag_cov.mean()



    
    

    def train_step(self, 
                   states, 
                   actions, 
                   optimizer,
                   scheduler, 
                   momentum=0.99,
                   target_decay=0.99, 
                   distance_function="l2", 
                   debug=False,
                   max_grad_norm=0.5,
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


        loss = self.compute_loss(pred_encs, 
                                target_encs,)
        
        if debug:
            energy = loss.detach()  # Just reuse loss as a rough energy metric
            variance = self.compute_variance(pred_encs) # Quick variance calculation
            covariance = self.compute_dimensional_covariance(pred_encs).detach()  # Quick covariance calculation

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()


        # max_grad_norm = 0.1  # Set the maximum norm for gradients
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)

        optimizer.step()
        self.update_target_encoder(momentum=target_decay)

        scheduler.step()

        # # Update target encoder using momentum
        # with torch.no_grad():
        #     for param_q, param_k in zip(self.encoder.parameters(), self.target_encoder.parameters()):
        #         param_k.data = momentum * param_k.data + (1 - momentum) * param_q.data

        return loss.item() if not debug else (loss.item(), energy.item(), variance.item(), covariance.item())


