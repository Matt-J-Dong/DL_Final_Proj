from typing import NamedTuple, List, Any, Dict
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from torch.optim import Adam
from tqdm.auto import tqdm
import numpy as np

from models import JEPA_Model  # used only for repr_dim
from dataset import WallDataset

class Prober(torch.nn.Module):
    def __init__(self, input_dim, arch="256", output_shape=(2,)):
        super().__init__()
        hidden_dim = int(arch)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.ReLU(True),
            torch.nn.Linear(hidden_dim, output_shape[0])
        )

    def forward(self, x):
        return self.net(x)

@dataclass
class ProbingConfig:
    probe_targets: str = "locations"
    lr: float = 0.0002
    epochs: int = 20
    schedule = None
    sample_timesteps: int = 30
    prober_arch: str = "256"

def location_losses(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    assert pred.shape == target.shape, f"Shape mismatch: pred={pred.shape}, target={target.shape}"
    mse = (pred - target).pow(2).mean(dim=0)
    return mse

class Normalizer:
    def normalize_location(self, locs):
        # Simple normalizer: assume locations in range [0,1]
        return locs

    def unnormalize_mse(self, mse):
        # No unnormalization if normalized is trivial
        return mse

class ProbingEvaluator:
    def __init__(
        self,
        device: "cuda",
        model: torch.nn.Module,
        probe_train_ds,
        probe_val_ds: dict,
        config: ProbingConfig = ProbingConfig(),
        quick_debug: bool = False,
    ):
        self.device = device
        self.config = config
        self.model = model
        self.model.eval()
        self.quick_debug = quick_debug
        self.ds = probe_train_ds
        self.val_ds = probe_val_ds
        self.normalizer = Normalizer()

    def train_pred_prober(self):
        repr_dim = self.model.repr_dim
        dataset = self.ds
        model = self.model

        config = self.config
        epochs = config.epochs
        if self.quick_debug:
            epochs = 1
        test_batch = next(iter(dataset))
        prober_output_shape = getattr(test_batch, "locations")[0, 0].shape
        prober = Prober(repr_dim, config.prober_arch, output_shape=prober_output_shape).to(self.device)

        optimizer_pred = Adam(prober.parameters(), lr=config.lr)
        step = 0
        for epoch in range(epochs):
            for batch in dataset:
                init_state = batch.states[:,0].to(self.device)
                actions = batch.actions.to(self.device)
                with torch.no_grad():
                    pred_encs = model(init_state, actions).transpose(0,1) # [T,B,D]

                target = batch.locations.to(self.device)
                target = self.normalizer.normalize_location(target)
                n_steps = pred_encs.shape[0]
                bs = pred_encs.shape[1]

                # Sample timesteps if needed
                if config.sample_timesteps is not None and config.sample_timesteps < n_steps:
                    sampled_pred_encs = []
                    sampled_targets = []
                    for i in range(bs):
                        indices = torch.randperm(n_steps)[:config.sample_timesteps]
                        sampled_pred_encs.append(pred_encs[indices,i,:])
                        sampled_targets.append(target[i, indices])
                    pred_encs = torch.stack(sampled_pred_encs, dim=1)
                    target = torch.stack(sampled_targets, dim=0)

                # pred_encs: [T,B,D] -> T steps for each batch
                # run prober per-step
                pred_locs = torch.stack([prober(x) for x in pred_encs], dim=1)
                # pred_locs: [T,B,2]
                losses = location_losses(pred_locs, target)
                loss = losses.mean()

                optimizer_pred.zero_grad()
                loss.backward()
                optimizer_pred.step()

                step += 1
                if self.quick_debug and step > 2:
                    break
        return prober

    @torch.no_grad()
    def evaluate_all(self, prober):
        avg_losses = {}
        for prefix, val_ds in self.val_ds.items():
            avg_losses[prefix] = self.evaluate_pred_prober(prober, val_ds, prefix)
        return avg_losses

    @torch.no_grad()
    def evaluate_pred_prober(self, prober, val_ds, prefix=""):
        probing_losses = []
        prober.eval()
        for batch in val_ds:
            init_state = batch.states[:,0].to(self.device)
            actions = batch.actions.to(self.device)
            pred_encs = self.model(init_state, actions).transpose(0,1)
            target = batch.locations.to(self.device)
            target = self.normalizer.normalize_location(target)

            pred_locs = torch.stack([prober(x) for x in pred_encs], dim=1)
            losses = location_losses(pred_locs, target)
            probing_losses.append(losses.cpu())
        losses_t = torch.stack(probing_losses, dim=0).mean(dim=0)
        losses_t = self.normalizer.unnormalize_mse(losses_t)
        average_eval_loss = losses_t.mean(dim=-1).mean().item()
        return average_eval_loss
