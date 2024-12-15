import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

class BYOL(nn.Module):
    def __init__(self, encoder_fn=resnet50, projection_dim=256, hidden_dim=4096, target_decay=0.996):
        super(BYOL, self).__init__()
        
        # Online network
        self.online_encoder = encoder_fn(pretrained=False)
        self.online_projector = self._build_mlp(self.online_encoder.fc.in_features, hidden_dim, projection_dim)
        self.online_predictor = self._build_mlp(projection_dim, hidden_dim, projection_dim)
        
        # Replace the classifier head in the encoder
        self.online_encoder.fc = nn.Identity()

        # Target network (EMA of online network)
        self.target_encoder = encoder_fn(pretrained=False)
        self.target_projector = self._build_mlp(self.target_encoder.fc.in_features, hidden_dim, projection_dim)
        self.target_encoder.fc = nn.Identity()

        # EMA initialization
        self._initialize_target_network()
        self.target_decay = target_decay

    def _initialize_target_network(self):
        """Initialize the target network with the online network's weights."""
        for target_param, online_param in zip(self.target_encoder.parameters(), self.online_encoder.parameters()):
            target_param.data.copy_(online_param.data)
            target_param.requires_grad = False

        for target_param, online_param in zip(self.target_projector.parameters(), self.online_projector.parameters()):
            target_param.data.copy_(online_param.data)
            target_param.requires_grad = False

    def _build_mlp(self, input_dim, hidden_dim, output_dim):
        """Utility function to build an MLP."""
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

    @torch.no_grad()
    def _update_target_network(self):
        """Update the target network using exponential moving average."""
        for target_param, online_param in zip(self.target_encoder.parameters(), self.online_encoder.parameters()):
            target_param.data = self.target_decay * target_param.data + (1 - self.target_decay) * online_param.data

        for target_param, online_param in zip(self.target_projector.parameters(), self.online_projector.parameters()):
            target_param.data = self.target_decay * target_param.data + (1 - self.target_decay) * online_param.data

    def forward(self, x1, x2):
        """Forward pass for BYOL.
        Args:
            x1, x2: Two augmented views of the same batch of images.
        Returns:
            Loss for BYOL.
        """
        # Online network forward pass
        z1_online = self.online_projector(self.online_encoder(x1))
        z2_online = self.online_projector(self.online_encoder(x2))
        p1 = self.online_predictor(z1_online)
        p2 = self.online_predictor(z2_online)

        # Target network forward pass (no gradients)
        with torch.no_grad():
            z1_target = self.target_projector(self.target_encoder(x1))
            z2_target = self.target_projector(self.target_encoder(x2))

        # Loss computation
        loss = self._loss_fn(p1, z2_target) + self._loss_fn(p2, z1_target)
        
        # Update target network
        self._update_target_network()

        return loss

    def _loss_fn(self, p, z):
        """Mean squared error loss between normalized predictions and targets."""
        p = F.normalize(p, dim=-1)
        z = F.normalize(z, dim=-1)
        return 2 - 2 * (p * z).sum(dim=-1).mean()
