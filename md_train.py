# md_train.py

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
import os
from torch.utils.data import random_split, DataLoader
from dataset import create_wall_dataloader
from models_md import JEPA_Model
from torch.optim.lr_scheduler import CosineAnnealingLR, CyclicLR, StepLR
import wandb
from dotenv import load_dotenv
from evaluator_md import ProbingConfig, ProbingEvaluator #This is the correct import path

load_dotenv()
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
os.environ["WANDB_API_KEY"] = WANDB_API_KEY
wandb.login(key=WANDB_API_KEY)

if torch.cuda.is_available():
        device = torch.device('cuda')
else:
    device = torch.device('cpu')

class Trainer:
    def __init__(self, config):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config
        print(f"Using device: {self.device}")

    def load_data(self):
        data_path = "/scratch/DL24FA"

        # Create Training DataLoader with probing=False
        train_loader = create_wall_dataloader(
            data_path=f"{data_path}/train",
            probing=False,  # Training does not require 'locations'
            device=self.device,
            train=True,
            batch_size=self.config["batch_size"],
        )

        # Create Validation DataLoader with probing=True
        probe_val_normal_ds = create_wall_dataloader(
            data_path=f"{data_path}/probe_normal/val",
            probing=True,
            device=device,
            train=False,
        )

        probe_val_wall_ds = create_wall_dataloader(
            data_path=f"{data_path}/probe_wall/val",
            probing=True,
            device=device,
            train=False,
        )

        val_loader = {"normal": probe_val_normal_ds, "wall": probe_val_wall_ds}

        full_train_dataset = train_loader.dataset
        train_size = int(self.config["split_ratio"] * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size

        # Ensure consistent splits by using the same generator
        generator = torch.Generator().manual_seed(42)
        train_subset, _ = random_split(full_train_dataset, [train_size, val_size], generator=generator)

        # Update DataLoaders with subsets
        train_loader = DataLoader(train_subset, batch_size=self.config["batch_size"], shuffle=True)

        # Store datasets for probing evaluation
        self.train_dataset = train_subset

        return train_loader, val_loader

    def save_model(self, model, epoch):
        save_path = self.config.get("save_path", "checkpoints")
        os.makedirs(save_path, exist_ok=True)
        save_file = os.path.join(save_path, f"jepa_model_epoch_{epoch}.pth")
        torch.save(model.state_dict(), save_file)
        print(f"Model saved to {save_file}")

    def validate_model(self, model, val_loader):
        model.eval()
        val_loss, var_culm, cov_culm = 0.0, 0.0, 0.0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating", leave=False):
                states, actions = batch.states.to(self.device), batch.actions.to(self.device)
                init_state = states[:, 0]

                # Slice actions for T-1 timesteps
                pred_encs = model.forward(init_state, actions[:, :-1])

                # Compute target encodings for all T timesteps
                target_encs = torch.stack([model.target_encoder(states[:, t]) for t in range(actions.size(1) + 1)], dim=1)

                # Compute loss
                loss, _, var, cov, _ = model.compute_loss(
                    pred_encs, target_encs[:, :pred_encs.size(1)], debug=True, **self.config
                )
                val_loss += loss.item()
                var_culm += var
                cov_culm += cov

        avg_val_loss = val_loss / len(val_loader)
        avg_var = var_culm / len(val_loader)
        avg_cov = cov_culm / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}, Variance: {avg_var:.4f}, Covariance: {avg_cov:.4f}")

        # Begin Probing Evaluation
        probe_lr = self.config.get("probe_lr", 0.0002)

        # Prepare datasets for ProbingEvaluator
        probe_train_ds = self.train_dataset
        probe_val_ds = self.val_dataset  # Validation dataset with probing=True

        # Define ProbingConfig
        probing_config = ProbingConfig(
            probe_targets="locations",
            lr=probe_lr,
            epochs=20,
            schedule=None,
            sample_timesteps=30,
            prober_arch="256",
        )

        # Initialize ProbingEvaluator
        evaluator = ProbingEvaluator(
            device=self.device,
            model=model,
            probe_train_ds=probe_train_ds,
            probe_val_ds=probe_val_ds,  # Pass the validation dataset with probing=True
            config=probing_config,
            quick_debug=False
        )

        # Train prober
        prober = evaluator.train_pred_prober()

        # Evaluate prober
        avg_losses = evaluator.evaluate_all(prober=prober)
        val_loss_normal = avg_losses.get("normal", 0.0)
        val_loss_wall = avg_losses.get("wall", 0.0)

        current_probe_lr = probe_lr
        print(f"Probing Evaluation - Validation normal loss: {val_loss_normal:.4f}, Validation wall loss: {val_loss_wall:.4f}, Probing LR: {current_probe_lr}")

        # Log probing results to wandb
        wandb.log({
            "val_loss_normal": val_loss_normal,
            "val_loss_wall": val_loss_wall,
            "probing_lr": current_probe_lr
        })

        # Write probing results to losses_testing.txt
        with open("losses_testing.txt", "a") as f:  # Changed filename
            f.write(f"Validation: val_loss_normal={val_loss_normal}, val_loss_wall={val_loss_wall}, probing_lr={current_probe_lr}\n")

        return avg_val_loss

    def train(self):
        train_loader, val_loader = self.load_data()
        model = JEPA_Model(device=self.device, repr_dim=256, action_dim=2, dropout_prob=0).to(self.device)

        optimizer = optim.Adam(model.parameters(), lr=self.config["learning_rate"], weight_decay=1e-4)
        scheduler = CyclicLR(
            optimizer,
            base_lr=self.config["learning_rate"] / 10,
            max_lr=self.config["learning_rate"] * 0.5,
            step_size_up=2 * len(train_loader),
            mode='triangular2'
        )
        
        # optimizer = torch.optim.SGD(model.parameters(), lr=self.config['learning_rate'], momentum=self.config['momentum'], weight_decay=1e-4)
        # scheduler = StepLR(optimizer, step_size=50, gamma=0.4)  # Reduce LR by 50% every 5 epochs

        for epoch in range(1, self.config["num_epochs"] + 1):
            print(f"Epoch {epoch}, Learning Rate: {optimizer.param_groups[0]['lr']}")
            # wandb.log({"lr": optimizer.param_groups[0]['lr']})
            epoch_loss = 0.0
            model.train()

            for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
                states, actions = batch.states.to(self.device), batch.actions.to(self.device)

                loss, e_loss, var_loss, cov_loss, contra_loss = model.train_step(
                    states=states,
                    actions=actions,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    debug=True,
                    **self.config
                )
                epoch_loss += loss

                if batch_idx % 10 == 0 or batch_idx == len(train_loader) - 1:
                    wandb.log({
                        "loss": loss, 
                        "energy_loss": e_loss, 
                        "variance_loss": var_loss, 
                        "covariance_loss": cov_loss,
                        "contrastive_loss": contra_loss,
                        'lr': optimizer.param_groups[0]['lr']
                    })

                if batch_idx % 100 == 0:
                    print(f"Epoch [{epoch}/{self.config['num_epochs']}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss:.4f}")
                    self.save_model(model, f"{epoch}_batch_{batch_idx}")
                    val_loss = self.validate_model(model, val_loader)
                    wandb.log({"val_loss": val_loss})

            avg_epoch_loss = epoch_loss / len(train_loader)
            print(f"Epoch [{epoch}/{self.config['num_epochs']}] Average Loss: {avg_epoch_loss:.4f}")
            scheduler.step()

            if epoch % self.config["save_every"] == 0:
                self.save_model(model, epoch)
                val_loss = self.validate_model(model, val_loader)
                wandb.log({"val_loss": val_loss})

        print("Training completed.")
        self.save_model(model, "final")


def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    config = {
        "batch_size": 1024,
        "num_epochs": 10,
        "learning_rate": 5e-3,
        'step_per_epoch': 1000,
        "momentum": 0.996,
        "split_ratio": 0.9,
        "lambda_energy": 1.0,
        "lambda_var": 0.0,
        "lambda_cov": 0.0,
        "max_grad_norm": 1.0,
        "min_variance": 0.1,
        "save_every": 1,
        'margin': 0.5,
        'lambda_contrastive': 0.0,
        'distance_function': 'l2',
        "probe_lr": 0.0002,  # Added probe_lr to config
    }

    wandb.init(
        project="DL_Final_Project_2024",
        config=config
    )

    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
