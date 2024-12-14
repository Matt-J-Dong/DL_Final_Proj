import torch
import torch.nn as nn

def build_mlp(input_dim, hidden_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(True),
        nn.Linear(hidden_dim, output_dim)
    )

class BYOL_Encoder(nn.Module):
    def __init__(self, input_dim=128, output_dim=32):
        super().__init__()
        # A tiny MLP encoder
        self.net = build_mlp(input_dim, 64, output_dim)

    def forward(self, x):
        return self.net(x)

class BYOL_Predictor(nn.Module):
    def __init__(self, input_dim=32, output_dim=32):
        super().__init__()
        self.net = build_mlp(input_dim, 64, output_dim)

    def forward(self, x):
        return self.net(x)

class JEPA_Model(nn.Module):
    def __init__(self, input_dim=128, repr_dim=32, action_dim=16, device='cpu'):
        super().__init__()
        self.device = device
        self.repr_dim = repr_dim
        self.action_dim = action_dim

        # Online encoder
        self.encoder = BYOL_Encoder(input_dim=input_dim, output_dim=repr_dim)
        # Target encoder
        self.target_encoder = BYOL_Encoder(input_dim=input_dim, output_dim=repr_dim)
        self.target_encoder.load_state_dict(self.encoder.state_dict())

        for param in self.target_encoder.parameters():
            param.requires_grad = False

        # Predictor
        self.predictor = BYOL_Predictor(input_dim=repr_dim, output_dim=repr_dim)

    def forward(self, states, actions):
        # For simplicity, we don't even use actions in this minimal example
        s_enc = self.encoder(states)
        return s_enc

    def train_step(self, states, actions, optimizer, momentum=0.99):
        # Online encoding & prediction
        online_enc = self.encoder(states)
        online_pred = self.predictor(online_enc)

        with torch.no_grad():
            target_enc = self.target_encoder(states)

        loss = ((online_pred - target_enc)**2).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        optimizer.step()

        # Momentum update for target encoder
        with torch.no_grad():
            for param_q, param_k in zip(self.encoder.parameters(), self.target_encoder.parameters()):
                param_k.data = momentum * param_k.data + (1 - momentum) * param_q.data

        return loss.item()
