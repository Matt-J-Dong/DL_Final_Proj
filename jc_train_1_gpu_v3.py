import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
import os
from torch.utils.data import random_split, DataLoader
from dataset import create_wall_dataloader
from models_jc import JEPA_Model
import torch.multiprocessing as mp
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb

class Trainer:
    def __init__(self, config):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config
        print(f"Using device: {self.device}")

    def load_data(self):
        data_path = "/scratch/DL24FA"

        full_loader = create_wall_dataloader(
            data_path=f"{data_path}/train",
            probing=False,
            device=self.device,
            train=True,
            batch_size=self.config["batch_size"],
        )

        full_dataset = full_loader.dataset
        train_size = int(self.config["split_ratio"] * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=self.config["batch_size"], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config["batch_size"], shuffle=False)

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

        print(f"Validation Variance: {var_culm / len(val_loader)}, Covariance: {cov_culm / len(val_loader)}")
        return val_loss / len(val_loader)


    def train(self):
        train_loader, val_loader = self.load_data()
        model = JEPA_Model(device=self.device, repr_dim=256, action_dim=2, dropout_prob=0).to(self.device)

        optimizer = optim.Adam(model.parameters(), lr=self.config["learning_rate"])
        scheduler = CosineAnnealingLR(optimizer, T_max=self.config["num_epochs"])

        for epoch in range(1, self.config["num_epochs"] + 1):
            print(f"Epoch {epoch}, Learning Rate: {optimizer.param_groups[0]['lr']}")
            epoch_loss = 0.0
            model.train()

            for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
                states, actions = batch.states.to(self.device), batch.actions.to(self.device)

                loss, e_loss, var_loss, cov_loss, contra_loss = model.train_step(
                    states=states,
                    actions=actions,
                    optimizer=optimizer,
                    debug=True,
                    **self.config
                )
                epoch_loss += loss

                if batch_idx % 10 == 0 or batch_idx == len(train_loader) - 1:
                    wandb.log({"loss": loss, 
                               "energy_loss": e_loss, 
                               "variance_loss": var_loss, 
                               "covariance_loss": cov_loss,
                               "contrastive_loss": contra_loss})

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
    config = {
        "batch_size": 512,
        "num_epochs": 20,
        "learning_rate": 2e-4,
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
    }

    wandb.init(
        project="DL_Final_Project_2024",
        config=config
    )

    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
