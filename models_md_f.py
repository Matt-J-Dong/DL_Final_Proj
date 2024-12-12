import torch
import torch.nn as nn

class SimpleCNN_Encoder(nn.Module):
    def __init__(self, output_dim=256, input_channels=2):
        super(SimpleCNN_Encoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1,1)),
        )
        self.fc = nn.Linear(128, output_dim)

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class LSTM_Predictor(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=512, dropout=0.1):
        super(LSTM_Predictor, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.residual = nn.Linear(input_dim, output_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, s_prev, u_prev):
        x = torch.cat([s_prev, u_prev], dim=-1).unsqueeze(1)
        res = self.residual(x.squeeze(1))
        out, (h, c) = self.lstm(x)
        out = self.dropout(out)
        out = self.fc(out.squeeze(1)) + res
        return out

class JEPA_Model(nn.Module):
    def __init__(self, device="cuda", repr_dim=256, action_dim=2, dropout=0.1):
        super(JEPA_Model, self).__init__()
        self.device = device
        self.repr_dim = repr_dim
        self.action_dim = action_dim
        self.encoder = SimpleCNN_Encoder(output_dim=repr_dim, input_channels=2).to(device)
        self.predictor = LSTM_Predictor(input_dim=repr_dim+action_dim, output_dim=repr_dim, hidden_dim=512, dropout=dropout).to(device)
        self.target_encoder = SimpleCNN_Encoder(output_dim=repr_dim, input_channels=2).to(device)
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

    def train_step(self, states, actions, optimizer, momentum=0.99, distance_function="l2", add_noise=False, lambda_cov=0.5):
        B, T, C, H, W = states.shape
        if add_noise:
            states = states + 0.01 * torch.randn_like(states)
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
        with torch.no_grad():
            for param_q, param_k in zip(self.encoder.parameters(), self.target_encoder.parameters()):
                param_k.data = momentum * param_k.data + (1 - momentum) * param_q.data
        return loss.item(), pred_encs

    def compute_energy(self, predicted_encs, target_encs, distance_function="l2"):
        if distance_function == "l2":
            energy = torch.sum((predicted_encs - target_encs)**2) / (predicted_encs.size(0)*predicted_encs.size(1))
        elif distance_function == "cosine":
            cos = nn.CosineSimilarity(dim=-1)
            energy = -torch.sum(cos(predicted_encs, target_encs)) / (predicted_encs.size(0)*predicted_encs.size(1))
        else:
            raise ValueError("Unknown distance function")
        return energy

    def variance_regularization(self, states, epsilon=1e-4):
        std = torch.sqrt(states.var(dim=0) + 1e-10)
        return torch.mean(torch.relu(epsilon - std))

    def covariance_regularization(self, states):
        if states.ndim == 3:
            states = states.view(-1, states.size(-1))
        batch_size, dim = states.size()
        norm_states = states - states.mean(dim=0, keepdim=True)
        cov_matrix = (norm_states.T @ norm_states) / (batch_size - 1)
        off_diag = cov_matrix - torch.diag(torch.diag(cov_matrix))
        return torch.sum(off_diag**2)
