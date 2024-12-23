# lora.py

import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=4, alpha=1.0):
        super(LoRALinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.scaling = alpha / r

        # The base weight (frozen during LoRA training)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.weight.requires_grad = False  # Freeze base weight

        # LoRA weights (trainable)
        self.W_down = nn.Parameter(torch.zeros(in_features, r))
        self.W_up = nn.Parameter(torch.zeros(r, out_features))
        nn.init.kaiming_uniform_(self.W_down, a=math.sqrt(5))
        nn.init.zeros_(self.W_up)

        # Bias term (optional)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        # Compute the LoRA update
        delta_weight = (self.W_down @ self.W_up) * self.scaling  # [in_features, out_features]
        delta_weight = delta_weight.T  # Transpose to [out_features, in_features]
        # Apply the combined weight
        return F.linear(x, self.weight + delta_weight, self.bias)
