import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

# 1. Define the Single Linear Layer Model
class SingleLinearModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(SingleLinearModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)
