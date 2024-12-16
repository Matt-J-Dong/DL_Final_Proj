from typing import List
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torchvision.models as models
from torch import optim

# Helper functions from BYOL snippet
def cosine_similarity_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return 2 - 2 * (x * y).sum(dim=-1).mean()

def update_target_network(online_net, target_net, momentum=0.99):
    for param_o, param_t in zip(online_net.parameters(), target_net.parameters()):
        param_t.data = param_t.data * momentum + param_o.data * (1 - momentum)

class BYOL(nn.Module):
    def __init__(self, base_encoder, projection_dim=256, hidden_dim=4096):
        super().__init__()
        # base encoder
        self.online_encoder = base_encoder(pretrained=False)
        # remove fc
        dim_mlp = self.online_encoder.fc.in_features
        self.online_encoder.fc = nn.Identity()

        # projectors
        self.online_projector = nn.Sequential(
            nn.Linear(dim_mlp, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, projection_dim)
        )

        self.online_predictor = nn.Sequential(
            nn.Linear(projection_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, projection_dim)
        )

        # target network (same architecture)
        self.target_encoder = base_encoder(pretrained=False)
        self.target_encoder.fc = nn.Identity()
        self.target_projector = nn.Sequential(
            nn.Linear(dim_mlp, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, projection_dim)
        )

        # initialize target weights
        self.target_encoder.load_state_dict(self.online_encoder.state_dict())
        self.target_projector.load_state_dict(self.online_projector.state_dict())
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        for p in self.target_projector.parameters():
            p.requires_grad = False

    def forward_online(self, x):
        # Encodes x and projects
        y = self.online_encoder(x)
        z = self.online_projector(y)
        p = self.online_predictor(z)
        return z, p

    def forward_target(self, x):
        with torch.no_grad():
            y = self.target_encoder(x)
            z = self.target_projector(y)
        return z

class RandomAugmentations:
    def __init__(self):
        # Minimal augmentation: random horizontal flip, maybe slight jitter
        pass

    def __call__(self, x):
        # x: [N, C, H, W]
        # Simple random horizontal flip
        if torch.rand(1).item() < 0.5:
            x = torch.flip(x, dims=[2])
        return x

class BYOL_Model(nn.Module):
    def __init__(self, device="cuda", repr_dim=256, action_dim=2, dropout=0.0):
        super(BYOL_Model, self).__init__()
        self.device = device
        self.repr_dim = repr_dim
        self.action_dim = action_dim

        # Use a ResNet18 as base encoder in BYOL
        self.byol = BYOL(base_encoder=models.resnet18, projection_dim=repr_dim, hidden_dim=4096).to(device)

        # We will define our own augmentation pipeline here for BYOL:
        self.augmentations = nn.Sequential()  # Will define actual augmentations below
        # We'll do augmentations inline in train_step for simplicity.

    def forward(self, init_state, actions):
        # For evaluation or embedding extraction:
        # init_state: [B, C, H, W] single frame
        # But we want embeddings for all timesteps. This function should mimic JEPA forward:
        # Actually, for probing, we need embeddings for all timesteps states[:, t].
        # We'll just encode all frames at once. The caller (ProbingEvaluator) might pass states as a whole sequence.
        # We'll handle that similarly to JEPA_Model: we expect something like [B, T, C, H, W].

        if init_state.ndim == 5:
            # init_state is actually full states: [B, T, C, H, W]
            states = init_state
        else:
            # If only a single frame is passed
            states = init_state.unsqueeze(1)

        B, T, C, H, W = states.shape
        states_flat = states.view(B*T, C, H, W)

        # Just encode with online encoder to get embeddings
        with torch.no_grad():
            y = self.byol.online_encoder(states_flat)
        # y: [B*T, dim_mlp]
        # Project to repr_dim if needed
        # The byol.online_projector maps to repr_dim
        with torch.no_grad():
            z = self.byol.online_projector(y)  # [B*T, repr_dim]

        pred_encs = z.view(B, T, self.repr_dim)
        return pred_encs

    def train_step(self, states, actions, optimizer, momentum=0.99, distance_function="l2", add_noise=False, lambda_cov=0.5):
        # This method replaces JEPA_Model's train_step with BYOL logic.
        # states: [B, T, C, H, W]
        # actions: [B, T-1, action_dim] (not needed for BYOL, we ignore it)

        B, T, C, H, W = states.shape

        if add_noise:
            states = states + 0.01 * torch.randn_like(states)

        # Random horizontal flip (as original code did)
        if torch.rand(1).item() < 0.5:
            states = torch.flip(states, dims=[2])

        # Flatten to apply BYOL (treat all frames as separate samples)
        states_flat = states.view(B*T, C, H, W)

        # Create two augmented views for BYOL:
        # We'll do simple augmentations: random flip again + normalization
        # For minimal changes, let's just do normalization here.
        # Mean and std for 2-channel input:
        mean = torch.tensor([0.5, 0.5], device=states_flat.device) if C == 2 else torch.tensor([0.5]*C, device=states_flat.device)
        std = torch.tensor([0.5]*C, device=states_flat.device)
        states_flat = (states_flat - mean[None,:,None,None]) / std[None,:,None,None]

        # Two augmented views (for BYOL we need two):
        # We'll just do random horizontal flip again to simulate difference:
        x1 = states_flat.clone()
        x2 = states_flat.clone()
        if torch.rand(1).item() < 0.5:
            x1 = torch.flip(x1, dims=[2])
        if torch.rand(1).item() < 0.5:
            x2 = torch.flip(x2, dims=[2])

        # Forward pass online
        z1_online, p1 = self.byol.forward_online(x1)
        z2_online, p2 = self.byol.forward_online(x2)

        # Forward pass target
        with torch.no_grad():
            z1_target = self.byol.forward_target(x1)
            z2_target = self.byol.forward_target(x2)

        # BYOL loss
        loss = cosine_similarity_loss(p1, z2_target) + cosine_similarity_loss(p2, z1_target)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 1.0)

        # Update is done outside (optimizer step done in the training loop)
        # Here we do momentum update of target network
        update_target_network(self.byol.online_encoder, self.byol.target_encoder, momentum=momentum)
        update_target_network(self.byol.online_projector, self.byol.target_projector, momentum=momentum)

        # pred_encs: return embeddings from z1_online as predicted embeddings
        # z1_online: [B*T, repr_dim]
        pred_encs = z1_online.view(B, T, self.repr_dim)

        return loss.item(), pred_encs

def build_mlp(layers_dims: List[int]):
    """Utility function to build an MLP."""
    layers = []
    for i in range(len(layers_dims) - 2):
        layers.append(nn.Linear(layers_dims[i], layers_dims[i + 1]))
        layers.append(nn.BatchNorm1d(layers_dims[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(layers_dims[-2], layers_dims[-1]))
    return nn.Sequential(*layers)

class Prober(torch.nn.Module):
    def __init__(
        self,
        embedding: int,
        arch: str,
        output_shape: List[int],
    ):
        super().__init__()
        
        # Debugging: Print the type and value of output_shape
        print(f"Prober __init__: Received output_shape type: {type(output_shape)}, value: {output_shape}")
        
        # Convert output_shape to list if it's a tuple or torch.Size
        if isinstance(output_shape, (torch.Size, tuple)):
            output_shape = list(output_shape)
            print(f"Prober __init__: Converted output_shape to list: {output_shape}")
        elif isinstance(output_shape, list):
            print(f"Prober __init__: output_shape is already a list: {output_shape}")
        else:
            raise TypeError(f"Prober __init__: output_shape must be a list, tuple, or torch.Size, got {type(output_shape)}")
        
        # Assert that all elements in output_shape are integers
        if not all(isinstance(x, int) for x in output_shape):
            raise TypeError("Prober __init__: All elements in output_shape must be integers.")
        
        self.output_dim = int(np.prod(output_shape))  # Ensure output_dim is integer
        print(f"Prober __init__: Calculated output_dim={self.output_dim}")
        self.output_shape = output_shape
        self.arch = arch

        arch_list = list(map(int, arch.split("-"))) if arch != "" else []
        f = [embedding] + arch_list + [self.output_dim]
        print(f"Prober __init__: Architecture dimensions: {f}")
        
        layers = []
        for i in range(len(f) - 2):
            in_features = f[i]
            out_features = f[i + 1]
            print(f"Prober __init__: Adding Linear layer with in_features={in_features}, out_features={out_features}")
            layers.append(torch.nn.Linear(in_features, out_features))
            layers.append(torch.nn.ReLU(True))
        # Final Linear layer
        in_features = f[-2]
        out_features = f[-1]
        print(f"Prober __init__: Adding final Linear layer with in_features={in_features}, out_features={out_features}")
        layers.append(torch.nn.Linear(in_features, out_features))
        self.prober = torch.nn.Sequential(*layers)

    def forward(self, e):
        output = self.prober(e)
        return output