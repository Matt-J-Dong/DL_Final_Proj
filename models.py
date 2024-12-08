# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import math

###############################################################################
# Vision Transformer Encoder
###############################################################################
class PatchEmbed(nn.Module):
    def __init__(self, img_size=64, patch_size=8, in_chans=2, embed_dim=256):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        num_patches = self.grid_size * self.grid_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches+1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=.02)

    def forward(self, x):
        # x: [B, 2, H, W]
        B, C, H, W = x.shape
        x = self.proj(x) # [B, embed_dim, H/patch, W/patch]
        x = x.flatten(2).transpose(1,2) # [B, num_patches, embed_dim]

        cls_token = self.cls_token.expand(B, -1, -1) # [B,1,embed_dim]
        x = torch.cat((cls_token, x), dim=1) # [B, num_patches+1, embed_dim]

        x = x + self.pos_embed
        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(self, dim=256, num_heads=4, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim*mlp_ratio)),
            nn.ReLU(True),
            nn.Linear(int(dim*mlp_ratio), dim),
            nn.Dropout(drop)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformerEncoder(nn.Module):
    def __init__(self, img_size=64, patch_size=8, in_chans=2, embed_dim=256, depth=4, num_heads=4, mlp_ratio=4.0):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.blocks = nn.ModuleList([TransformerEncoderBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: [B,2,H,W]
        x = self.patch_embed(x) # [B, num_patches+1, embed_dim]
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        # Take the cls_token output as the image embedding
        cls_emb = x[:,0] # [B,embed_dim]
        return cls_emb


###############################################################################
# Predictor (Decoder)
#
# Given the previous embedding s_{t-1} and action u_{t-1}, predict next embedding s_t.
# We'll use a GRU for simplicity.
###############################################################################
class Predictor(nn.Module):
    def __init__(self, state_dim=256, action_dim=2, hidden_dim=256):
        super().__init__()
        # Input to the GRU: previous state + action
        self.gru = nn.GRU(state_dim + action_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, state_dim)

    def forward(self, s_prev, u_prev):
        # s_prev: [B, D], u_prev: [B, A]
        # Combine them and add a seq dim for GRU
        x = torch.cat([s_prev, u_prev], dim=-1).unsqueeze(1) # [B,1,D+A]
        output, _ = self.gru(x) # [B,1,H]
        s_pred = self.fc(output.squeeze(1)) # [B,D]
        return s_pred


###############################################################################
# JEPA Model (ViT Encoder + GRU Predictor + Target Encoder)
###############################################################################
class JEPA_ViTModel(nn.Module):
    def __init__(self, device="cuda", repr_dim=256, action_dim=2, img_size=64):
        super(JEPA_ViTModel, self).__init__()
        self.device = device
        self.repr_dim = repr_dim
        self.action_dim = action_dim

        # Encoder and Target Encoder
        self.encoder = VisionTransformerEncoder(img_size=img_size, embed_dim=repr_dim).to(device)
        self.target_encoder = VisionTransformerEncoder(img_size=img_size, embed_dim=repr_dim).to(device)
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        # Predictor (Decoder)
        self.predictor = Predictor(state_dim=repr_dim, action_dim=action_dim, hidden_dim=repr_dim).to(device)

    def forward(self, states, actions):
        """
        states: [B,T,C,H,W]
        actions: [B,T-1,2]

        Output:
            predicted_embeddings: [B,T,D]
        """
        B, T, C, H, W = states.shape
        device = states.device

        # Get initial state embedding s_0
        o_0 = states[:,0] # [B,C,H,W]
        s_0 = self.encoder(o_0) # [B,D]

        pred_encs = [s_0]
        s_prev = s_0

        for t in range(1,T):
            u_prev = actions[:, t-1] # [B,2]
            s_pred = self.predictor(s_prev, u_prev) # [B,D]
            pred_encs.append(s_pred)
            s_prev = s_pred

        pred_encs = torch.stack(pred_encs, dim=1) # [B,T,D]
        return pred_encs

    def update_target_encoder(self, momentum=0.99):
        with torch.no_grad():
            for param_q, param_k in zip(self.encoder.parameters(), self.target_encoder.parameters()):
                param_k.data = momentum * param_k.data + (1 - momentum) * param_q.data


class Prober(nn.Module):
    def __init__(
        self,
        embedding: int,
        arch: str,
        output_shape: List[int],
    ):
        super().__init__()
        import numpy as np
        self.output_dim = np.prod(output_shape)
        self.output_shape = output_shape
        self.arch = arch

        arch_list = list(map(int, arch.split("-"))) if arch != "" else []
        f = [embedding] + arch_list + [self.output_dim]
        layers = []
        for i in range(len(f) - 2):
            layers.append(nn.Linear(f[i], f[i + 1]))
            layers.append(nn.ReLU(True))
        layers.append(nn.Linear(f[-2], f[-1]))
        self.prober = nn.Sequential(*layers)

    def forward(self, e):
        output = self.prober(e)
        return output
