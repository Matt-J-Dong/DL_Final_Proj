# md_train_v5.py

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
import os
from torch.utils.data import random_split, DataLoader
from dataset_md import create_wall_dataloader
from models_md_v5 import JEPA_Model
from torch.optim.lr_scheduler import CosineAnnealingLR, CyclicLR, StepLR
import wandb
from dotenv import load_dotenv
from evaluator_md import ProbingEvaluator, ProbingConfig
import torch.multiprocessing as mp

model_version = "v5"
model_size = 128  # Updated to match repr_dim in the checkpoint

load_dotenv()
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
os.environ["WANDB_API_KEY"] = WANDB_API_KEY
wandb.login(key=WANDB_API_KEY)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

#mp.set_start_method('spawn', force=True)

class Trainer:
    def __init__(self, config):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config
        print(f"Using device: {self.device}")

    def load_data(self):
        data_path = "/scratch/DL24FA"

        # Create Training DataLoader with probing=False
        full_loader = create_wall_dataloader(
            data_path=f"{data_path}/train",
            probing=False,
            device=self.device,
            train=True,
            #No batch_size argument
        )

        # Create Probing Training DataLoader with probing=True and specified batch_size
        probe_train_ds = create_wall_dataloader(
            data_path=f"{data_path}/probe_normal/train",
            probing=True,
            device=self.device,
            train=True,
            #No batch_size argument
        )

        # Create Validation DataLoaders with probing=True and specified batch_size
        probe_val_normal_ds = create_wall_dataloader(
            data_path=f"{data_path}/probe_normal/val",
            probing=True,
            device=self.device,
            train=False,
            #No batch_size argument
        )

        probe_val_wall_ds = create_wall_dataloader(
            data_path=f"{data_path}/probe_wall/val",
            probing=True,
            device=self.device,
            train=False,
            #No batch_size argument
        )

        val_loader = {"normal": probe_val_normal_ds, "wall": probe_val_wall_ds}

        full_dataset = full_loader.dataset
        train_size = int(self.config["split_ratio"] * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, _ = random_split(full_dataset, [train_size, val_size])

        # Save datasets for probing
        self.train_dataset = train_dataset
        self.probe_train_dataset = probe_train_ds.dataset  # **Store probe_train_dataset**
        self.val_loader = val_loader

        train_loader = DataLoader(train_dataset, batch_size=self.config["batch_size"], shuffle=True)

        # Debugging: Inspect the shapes of 'locations' in each DataLoader
        print("\n--- Debugging DataLoader Shapes ---")

        # Function to print the shape of 'locations' from a DataLoader
        def inspect_dataloader(dataloader, name):
            try:
                batch = next(iter(dataloader))
                locations = getattr(batch, "locations", None)
                if locations is not None and locations.numel() > 0:
                    print(f"{name} 'locations' shape: {locations.shape}")
                else:
                    print(f"{name} does not have 'locations' attribute or it's empty.")
            except Exception as e:
                print(f"Error inspecting {name} DataLoader: {e}")

        # Inspect Training DataLoader
        inspect_dataloader(train_loader, "Training")

        # Inspect Probing Training DataLoader
        inspect_dataloader(probe_train_ds, "Probing Training")

        # Inspect Validation DataLoaders
        inspect_dataloader(probe_val_normal_ds, "Validation Normal")
        inspect_dataloader(probe_val_wall_ds, "Validation Wall")

        print("--- End of Debugging DataLoader Shapes ---\n")

        return train_loader, val_loader

    def save_model(self, model, epoch):
        save_path = self.config.get("save_path", "checkpoints")
        os.makedirs(save_path, exist_ok=True)
        save_file = os.path.join(save_path, f"jepa_model_{model_version}_epoch_{epoch}.pth")
        torch.save(model.state_dict(), save_file)
        print(f"Model saved to {save_file}")

    # Replaced the existing validate_model method with the new validation code from test.py
    def validate_model(self, model, epoch, avg_epoch_loss, optimizer):
        # Retrieve probe learning rate from config or use default
        probe_lr = self.config.get("probe_lr", 0.0002)

        model.eval()

        probing_config = ProbingConfig(
            probe_targets="locations",
            lr=probe_lr,  # Overridden below
            epochs=20,
            schedule=None,
            sample_timesteps=30,
            prober_arch="256",
        )

        # Initialize ProbingEvaluator with probing training and validation datasets
        evaluator = ProbingEvaluator(
            device=self.device,
            model=model,
            probe_train_ds=self.probe_train_dataset,  # **Use probe_train_dataset**
            probe_val_ds=self.val_loader,              # Using the validation datasets for probing
            config=probing_config,
            quick_debug=False
        )

        # Override probe_lr from config
        evaluator.config.lr = probe_lr

        # Debugging: Wrap the call to train_pred_prober in a try-except block
        try:
            # Train the probing model
            prober = evaluator.train_pred_prober()
        except IndexError as e:
            print(f"\nIndexError encountered during probing training: {e}")
            print("Inspecting 'locations' tensor in training dataset:")
            # Inspect a sample from the probing training dataset
            if hasattr(self.probe_train_dataset, '__getitem__'):
                sample = self.probe_train_dataset[0]
                locations = getattr(sample, "locations", None)
                if locations is not None and locations.numel() > 0:
                    print(f"Probing Training sample 'locations' shape: {locations.shape}")
                else:
                    print("Probing Training sample does not have 'locations' attribute or it's empty.")
            print("Inspecting 'locations' tensors in validation datasets:")
            for key, val_ds in self.val_loader.items():
                if hasattr(val_ds, '__iter__'):
                    try:
                        val_sample = next(iter(val_ds))
                        locations = getattr(val_sample, "locations", None)
                        if locations is not None and locations.numel() > 0:
                            print(f"Validation '{key}' sample 'locations' shape: {locations.shape}")
                        else:
                            print(f"Validation '{key}' sample does not have 'locations' attribute or it's empty.")
                    except Exception as ex:
                        print(f"Error accessing validation '{key}' dataset: {ex}")
            raise e  # Re-raise the exception after debugging

        # Evaluate the probing model
        avg_losses = evaluator.evaluate_all(prober=prober)
        val_loss_normal = avg_losses.get("normal", 0.0)
        val_loss_wall = avg_losses.get("wall", 0.0)

        current_probe_lr = probe_lr
        print(f"Validation normal loss: {val_loss_normal:.4f}, Validation wall loss: {val_loss_wall:.4f}, Probing LR: {current_probe_lr}")

        # Log metrics to Weights & Biases
        wandb.log({
            "epoch": epoch,
            "train_loss": avg_epoch_loss,
            "val_loss_normal": val_loss_normal,
            "val_loss_wall": val_loss_wall,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "probing_lr": current_probe_lr,
            "dropout": self.config.get("dropout_prob", 0.0),
            "lambda_cov": self.config.get("lambda_cov", 0.0),
            "batch_size": self.config.get("batch_size"),
            "momentum": self.config.get("momentum")
        })

        # Append losses to a text file for record-keeping
        with open(f"losses_model_{model_version}.txt", "a") as f:
            print(f"Writing file: {f}")
            f.write(f"Epoch {epoch}: train_loss={avg_epoch_loss}, val_loss_normal={val_loss_normal}, val_loss_wall={val_loss_wall}, probing_lr={current_probe_lr}\n")

    def train(self):
        train_loader, val_loader = self.load_data()
        model = JEPA_Model(
            device=self.device, 
            repr_dim=model_size,  # Set to 128 to match the updated repr_dim
            action_dim=2, 
            dropout_prob=self.config.get("dropout_prob", 0),
            margin=self.config.get("margin", 1.0)  # Pass margin to the model
        ).to(self.device)

        print(f"Initialized JEPA_Model with repr_dim={model.repr_dim}")

        optimizer = optim.Adam(model.parameters(), lr=self.config["learning_rate"], weight_decay=1e-4)
        scheduler = CyclicLR(
            optimizer,
            base_lr=self.config["learning_rate"]/10,
            max_lr=self.config["learning_rate"]*0.5,
            step_size_up=2 * len(train_loader),
            mode='triangular2'
        )
        
        # If you prefer SGD, uncomment the following lines
        # optimizer = torch.optim.SGD(model.parameters(), lr=self.config['learning_rate'], momentum=self.config['momentum'], weight_decay=1e-4)
        # scheduler = StepLR(optimizer, step_size=50, gamma=0.4)  # Reduce LR by 50% every 50 epochs

        for epoch in range(1, self.config["num_epochs"] + 1):
            print(f"Epoch {epoch}, Learning Rate: {optimizer.param_groups[0]['lr']}")
            # wandb.log({"lr": optimizer.param_groups[0]['lr']})  # Optional: Log learning rate separately
            epoch_loss = 0.0
            model.train()

            for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
                states, actions = batch.states.to(self.device), batch.actions.to(self.device)

                loss, e_loss, var_loss, cov_loss, contra_loss, neg_loss = model.train_step(
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
                        "negative_loss": neg_loss,
                        'lr': optimizer.param_groups[0]['lr']
                    })

                if batch_idx % 100 == 0:
                    print(f"Epoch [{epoch}/{self.config['num_epochs']}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss:.4f}")
                    self.save_model(model, f"{epoch}_batch_{batch_idx}")
                    # Perform validation
                    val_loss = self.validate_model(model, epoch, epoch_loss / (batch_idx + 1), optimizer)
                    wandb.log({"val_loss": val_loss})

            avg_epoch_loss = epoch_loss / len(train_loader)
            print(f"Epoch [{epoch}/{self.config['num_epochs']}] Average Loss: {avg_epoch_loss:.4f}")

            # Step the scheduler after the epoch
            scheduler.step()

            if epoch % self.config["save_every"] == 0:
                self.save_model(model, epoch)
                val_loss = self.validate_model(model, epoch, avg_epoch_loss, optimizer)
                wandb.log({"val_loss": val_loss})

        print("Training completed.")
        self.save_model(model, "final")


def main():
    config = {
        "batch_size": 256,
        "num_epochs": 7,
        "learning_rate": 5e-4,
        'step_per_epoch': 1000,
        "momentum": 0.996,
        "split_ratio": 1.0,
        "lambda_energy": 1.0,
        "lambda_var": 1.0,          # Increased from 0.0 to 1.0 for regularization
        "lambda_cov": 1.0,          # Increased from 0.0 to 1.0 for regularization
        "max_grad_norm": 1.0,
        "min_variance": 1.0,
        "save_every": 1,
        'margin': 1.0,              # Added margin for contrastive loss
        'lambda_contrastive': 0.1,  # Increased from 0.0 to 0.1 to enable contrastive loss
        'lambda_negative': 0.5,     # Added lambda_negative for negative sampling loss
        'distance_function': 'l2',
        "probe_lr": 0.0002,         # Added probe_lr with default value
        "dropout_prob": 0.0         # Added dropout_prob with default value
    }

    wandb.init(
        project=f"DL_Final_Project_2024_model_{model_version}",
        config=config
    )

    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
