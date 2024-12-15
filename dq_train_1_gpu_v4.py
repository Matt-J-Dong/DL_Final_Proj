from dq_model_v2 import BYOL  # Import the BYOL class
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from torch import optim
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
import os
from torch.utils.data import random_split, DataLoader
from dataset import create_wall_dataloader
from models_jc import JEPA_Model
import torch.multiprocessing as mp
from torch.optim.lr_scheduler import CosineAnnealingLR, CyclicLR, StepLR, LambdaLR
#import wandb
import math
from main_jc import load_data as load_validation_data, evaluate_model
from evaluator_jc import ProbingEvaluator
from dq_model_v2 import BYOL

class Trainer:
    def __init__(self, config):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config
        print(f"Using device: {self.device}")
    
    def load_data(self):
        data_path = "data/DL24FA"

        full_loader = create_wall_dataloader(
            data_path=f"{data_path}/train",
            probing=False,
            device=self.device,
            train=True,
            batch_size=self.config["batch_size"],
        )

        full_dataset = full_loader.dataset
        train_size = int(self.config["split_ratio"] * len(full_dataset))

        train_loader = DataLoader(full_dataset, batch_size=self.config["batch_size"], shuffle=True)

        # validation datasets
        val_train_ds, val_val_ds = load_validation_data(self.device)

        return train_loader, val_train_ds, val_val_ds

    def train(self):
        train_loader, val_train_ds, val_val_ds = self.load_data()
        
        # Initialize BYOL model
        model = BYOL(
            encoder_fn=resnet50,
            projection_dim=self.config["repr_dim"],
            hidden_dim=4096,
            target_decay=self.config["momentum"]
        ).to(self.device)

        optimizer = optim.Adam(model.parameters(), lr=self.config["learning_rate"], weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.config["num_epochs"], eta_min=1e-6)

        for epoch in range(1, self.config["num_epochs"] + 1):
            print(f"Epoch {epoch}, Learning Rate: {optimizer.param_groups[0]['lr']}")
            epoch_loss = 0.0
            model.train()

            for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
                states, actions = batch.states.to(self.device), batch.actions.to(self.device)

                # Adjust forward pass for BYOL
                loss = model(states, actions)  # BYOL computes its own loss internally
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

                # if batch_idx % 10 == 0 or batch_idx == len(train_loader) - 1:
                #     wandb.log({"loss": loss.item(), 'lr': optimizer.param_groups[0]['lr']})

                if batch_idx % 100 == 0 and batch_idx != 0:
                    print(f"Epoch [{epoch}/{self.config['num_epochs']}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")
                    self.save_model(model, f"{epoch}_batch_{batch_idx}")

            avg_epoch_loss = epoch_loss / len(train_loader)
            print(f"Epoch [{epoch}/{self.config['num_epochs']}] Average Loss: {avg_epoch_loss:.4f}")
            scheduler.step()

            if epoch % self.config["save_every"] == 0:
                self.save_model(model, epoch)
                self.validate_model(model, val_train_ds, val_val_ds)
                model.train()

        print("Training completed.")
        self.save_model(model, "final")

def main():
    config = {
        "batch_size": 512,
        "num_epochs": 20,
        "learning_rate": 1e-5,
        # 'step_per_epoch': 1000,
        "momentum": 0.996,
        "split_ratio": 1.0,
        "lambda_energy": 1.0,
        "lambda_var": 25.0,
        "lambda_cov": 1.0,
        'lambda_contrastive': 1.0,
        "max_grad_norm": 1.0,
        "min_variance": 1.0,
        "save_every": 1,
        'margin': 0.5,
        'distance_function': 'l2',
    }


    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
