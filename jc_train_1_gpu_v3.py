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
import wandb
import math

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

        train_loader = DataLoader(full_dataset, batch_size=self.config["batch_size"], shuffle=True)

        return train_loader

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
        train_loader = self.load_data()
        model = JEPA_Model(device=self.device, repr_dim=256, action_dim=2, dropout_prob=0).to(self.device)

        optimizer = optim.Adam(model.parameters(), lr=self.config["learning_rate"], weight_decay=1e-4)
        scheduler = get_cosine_schedule_with_warmup(optimizer, 
                                                    num_warmup_steps=10, 
                                                    num_training_steps=self.config["num_epochs"] * len(train_loader), 
                                                    initial_lr=self.config["learning_rate"], 
                                                    final_lr=1e-8)
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
                    wandb.log({"loss": loss, 
                               "energy_loss": e_loss, 
                               "variance_loss": var_loss, 
                               "covariance_loss": cov_loss,
                               "contrastive_loss": contra_loss,
                               'lr': optimizer.param_groups[0]['lr']})

                if batch_idx % 100 == 0:
                    print(f"Epoch [{epoch}/{self.config['num_epochs']}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss:.4f}")
                    self.save_model(model, f"{epoch}_batch_{batch_idx}")
                    # val_loss = self.validate_model(model, val_loader)
                    # wandb.log({"val_loss": val_loss})

            avg_epoch_loss = epoch_loss / len(train_loader)
            print(f"Epoch [{epoch}/{self.config['num_epochs']}] Average Loss: {avg_epoch_loss:.4f}")
            scheduler.step()

            if epoch % self.config["save_every"] == 0:
                self.save_model(model, epoch)
                # val_loss = self.validate_model(model, val_loader)
                # wandb.log({"val_loss": val_loss})

        print("Training completed.")
        self.save_model(model, "final")



# Define the learning rate scheduler with warmup and cosine decay
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1, initial_lr=1e-3, final_lr=1e-5):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            lr = initial_lr * float(current_step) / float(max(1, num_warmup_steps))
        else:
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            lr = final_lr + 5 * (initial_lr - final_lr) * (1.0 + math.cos(math.pi * progress))
        return lr / initial_lr  # Return multiplicative factor
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def main():
    config = {
        "batch_size": 512,
        "num_epochs": 20,
        "learning_rate": 2e-4,
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
        'margin': 1.0,
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
