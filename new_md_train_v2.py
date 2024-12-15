# byol_train_new_v2.py

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
from torchvision.transforms import Compose, RandomResizedCrop, RandomHorizontalFlip, GaussianBlur, Normalize
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm
import os
import torchvision.transforms.functional as F
import numpy as np

# Replace these imports with your actual modules
from dataset_md import create_wall_dataloader  # Custom data loader
from evaluator_md import ProbingEvaluator, ProbingConfig  # Custom evaluator

model_version = "v2"
model_size = 64
path_data = "./data/DL24FA"

load_dotenv()
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
os.environ["WANDB_API_KEY"] = WANDB_API_KEY
wandb.login(key=WANDB_API_KEY)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Define BYOL components: Online Network and Target Network
class BYOL(nn.Module):
    def __init__(self, base_encoder=resnet18, projection_dim=128, hidden_dim=4096):
        """
        BYOL Model consisting of an online network and a target network.
        
        Args:
            base_encoder (nn.Module): The base encoder architecture (e.g., ResNet-18).
            projection_dim (int): Dimension of the projection head output.
            hidden_dim (int): Dimension of the hidden layer in the projector and predictor.
        """
        super(BYOL, self).__init__()
        # Online network components
        self.online_encoder = base_encoder(pretrained=False)
        # Modify the first convolution layer to accept 2 channels instead of 3
        self.online_encoder.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.online_encoder.fc = nn.Identity()  # Remove the classification head
        
        self.online_projector = nn.Sequential(
            nn.Linear(512, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim),
        )
        
        self.online_predictor = nn.Sequential(
            nn.Linear(projection_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim),
        )
        
        # Target network components
        self.target_encoder = base_encoder(pretrained=False)
        self.target_encoder.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.target_encoder.fc = nn.Identity()  # Remove the classification head
        
        self.target_projector = nn.Sequential(
            nn.Linear(512, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim),
        )
        
        # Initialize target network parameters with online network parameters
        self.target_encoder.load_state_dict(self.online_encoder.state_dict())
        self.target_projector.load_state_dict(self.online_projector.state_dict())
        
        # Disable gradients for target network
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        for param in self.target_projector.parameters():
            param.requires_grad = False

    def forward_online(self, x):
        """
        Forward pass for the online network.
        
        Args:
            x (Tensor): Input tensor.
        
        Returns:
            projections (Tensor): Projected features.
            predictions (Tensor): Predicted projections.
        """
        features = self.online_encoder(x)
        projections = self.online_projector(features)
        predictions = self.online_predictor(projections)
        return projections, predictions

    @torch.no_grad()
    def forward_target(self, x):
        """
        Forward pass for the target network.
        
        Args:
            x (Tensor): Input tensor.
        
        Returns:
            projections (Tensor): Projected features.
        """
        features = self.target_encoder(x)
        projections = self.target_projector(features)
        return projections

# Define cosine similarity loss
def cosine_similarity_loss(z1, z2):
    """
    Computes the cosine similarity loss between two sets of projections.
    
    Args:
        z1 (Tensor): Predictions from the online network.
        z2 (Tensor): Projections from the target network.
    
    Returns:
        loss (Tensor): Cosine similarity loss.
    """
    z1 = nn.functional.normalize(z1, dim=1)
    z2 = nn.functional.normalize(z2, dim=1)
    return 2 - 2 * (z1 * z2).sum(dim=1).mean()

# Momentum update for the target network
def update_target_network(online_net, target_net, momentum):
    """
    Updates the target network parameters with the online network parameters using momentum.
    
    Args:
        online_net (BYOL): Online network.
        target_net (BYOL): Target network.
        momentum (float): Momentum coefficient.
    """
    for param_online, param_target in zip(online_net.parameters(), target_net.parameters()):
        param_target.data = momentum * param_target.data + (1 - momentum) * param_online.data

# Data augmentation for BYOL
class BYOLAugmentations:
    def __init__(self, image_size=36):
        """
        BYOL augmentations comprising two different augmented views of the same image.
        
        Args:
            image_size (int): Desired image size after cropping.
        """
        self.image_size = image_size

    def __call__(self, x):
        """
        Apply augmentations to the input image.
        
        Args:
            x (Tensor): Input tensor of shape [N, C, H, W].
        
        Returns:
            x_aug (Tensor): Augmented image tensor.
        """
        # x is a batch tensor: [N, C, H, W]
        # Apply augmentation per image
        augmented = []
        for idx, img in enumerate(x):
            # Debugging: Print image shape
            # print(f"Augmenting image {idx} with shape: {img.shape}")  # Uncomment for debugging

            # Ensure the tensor is writable by making a copy
            img = img.clone()

            # Apply RandomResizedCrop
            try:
                i, j, h, w = RandomResizedCrop.get_params(img, scale=(0.2, 1.0), ratio=(3./4., 4./3.))
                img = F.resized_crop(img, i, j, h, w, self.image_size, interpolation=F.InterpolationMode.BILINEAR)
            except Exception as e:
                print(f"Error in RandomResizedCrop for image {idx}: {e}")
                raise e

            # Apply RandomHorizontalFlip
            if torch.rand(1).item() < 0.5:
                img = F.hflip(img)

            # Apply GaussianBlur
            img = F.gaussian_blur(img, kernel_size=3, sigma=(0.1, 2.0))

            # Normalize
            if img.shape[0] == 2:
                mean = (0.5, 0.5)
                std = (0.5, 0.5)
            elif img.shape[0] == 1:
                mean = (0.5,)
                std = (0.5,)
            else:
                mean = (0.5, 0.5, 0.5)
                std = (0.5, 0.5, 0.5)
            try:
                img = F.normalize(img, mean=mean, std=std)
            except Exception as e:
                print(f"Error in normalization for image {idx}: {e}")
                print(f"Image shape: {img.shape}, mean: {mean}, std: {std}")
                raise e

            augmented.append(img)
        augmented = torch.stack(augmented).to(x.device)
        return augmented

# Trainer class
class Trainer:
    def __init__(self, config):
        """
        Initializes the Trainer with the given configuration.
        
        Args:
            config (dict): Configuration dictionary with hyperparameters.
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.load_data()
        
        # Initialize BYOL components
        self.online_network = BYOL(base_encoder=resnet18, projection_dim=self.config['projection_dim'], hidden_dim=self.config['hidden_dim']).to(self.device)
        self.target_network = BYOL(base_encoder=resnet18, projection_dim=self.config['projection_dim'], hidden_dim=self.config['hidden_dim']).to(self.device)
        
        # Initialize target network with online network parameters
        self.target_network.load_state_dict(self.online_network.state_dict())
        
        # Disable gradients for target network
        for param in self.target_network.parameters():
            param.requires_grad = False
        
        # Optimizer: Update only the online predictor parameters
        self.optimizer = optim.Adam(self.online_network.online_predictor.parameters(), lr=self.config['learning_rate'])
        
        # Scheduler (optional)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.config['epochs'])
        
        # Initialize augmentations
        self.augmentations = self.config['augmentations']
        
    def load_data(self):
        """
        Loads the training and validation data using the custom data loader.
        
        Returns:
            train_loader (DataLoader): DataLoader for training.
            val_loader (dict): Dictionary of DataLoaders for validation.
        """
        data_path = path_data

        # Create Training DataLoader with probing=False
        full_loader = create_wall_dataloader(
            data_path=f"{data_path}/train",
            probing=False,
            device=self.device,
            batch_size=self.config["batch_size"],
            train=True,
        )

        # Create Probing Training DataLoader with probing=True and specified batch_size
        probe_train_ds = create_wall_dataloader(
            data_path=f"{data_path}/probe_normal/train",
            probing=True,
            device=self.device,
            batch_size=self.config["batch_size"],
            train=True,
        )

        # Create Validation DataLoaders with probing=True and specified batch_size
        probe_val_normal_ds = create_wall_dataloader(
            data_path=f"{data_path}/probe_normal/val",
            probing=True,
            device=self.device,
            batch_size=self.config["batch_size"],
            train=False,
        )

        probe_val_wall_ds = create_wall_dataloader(
            data_path=f"{data_path}/probe_wall/val",
            probing=True,
            device=self.device,
            batch_size=self.config["batch_size"],
            train=False,
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

        train_loader = DataLoader(train_dataset, batch_size=self.config["batch_size"], shuffle=True, drop_last=True)

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

        self.train_loader = train_loader

    def validate_model(self, model, epoch, avg_epoch_loss, optimizer):
        """
        Validates the model using the ProbingEvaluator.
        
        Args:
            model (BYOL): The BYOL model being trained.
            epoch (int): Current epoch number.
            avg_epoch_loss (float): Average training loss for the epoch.
            optimizer (Optimizer): The optimizer used for training.
        """
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
        with open(f"losses_model_{self.config['model_version']}.txt", "a") as f:
            print(f"Writing file: {f}")
            f.write(f"Epoch {epoch}: train_loss={avg_epoch_loss}, val_loss_normal={val_loss_normal}, val_loss_wall={val_loss_wall}, probing_lr={current_probe_lr}\n")

    def train(self):
        """
        Performs the training loop for BYOL.
        """
        # Initialize BYOL networks
        online_net = self.online_network
        target_net = self.target_network

        # Training loop
        for epoch in range(1, self.config["epochs"] + 1):
            online_net.train()
            total_loss = 0.0
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config['epochs']}")

            for batch_idx, batch in enumerate(progress_bar):
                # Assuming batch has 'states' and 'actions' attributes
                # Use 'states' as images
                x = getattr(batch, "states", None)
                if x is None:
                    print("Batch does not have 'states' attribute. Skipping this batch.")
                    continue
                # Fix non-writable tensor warning by making a writable copy
                if not x.is_contiguous():
                    x = x.contiguous()
                # Ensure x is writable
                if not x.requires_grad:
                    x = x.clone()
                x = x.to(self.device)

                # Apply two separate augmentations
                try:
                    x1 = self.augmentations(x)
                    x2 = self.augmentations(x)
                except Exception as e:
                    print(f"Error during augmentation: {e}")
                    continue

                # Forward pass through online network
                z1_online, p1 = online_net.forward_online(x1)
                z2_online, p2 = online_net.forward_online(x2)

                # Forward pass through target network
                with torch.no_grad():
                    z1_target = target_net.forward_target(x1)
                    z2_target = target_net.forward_target(x2)

                # Compute loss
                loss = cosine_similarity_loss(p1, z2_target) + cosine_similarity_loss(p2, z1_target)

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Update target network with momentum
                update_target_network(online_net, target_net, self.config['momentum'])

                total_loss += loss.item()

                # Update progress bar
                progress_bar.set_postfix({"loss": loss.item()})

            avg_train_loss = total_loss / len(self.train_loader)

            # Validation
            self.validate_model(online_net, epoch, avg_train_loss, self.optimizer)

            # Step the scheduler
            self.scheduler.step()

            # Save model checkpoint after each epoch
            checkpoint_path = os.path.join(self.config["save_dir"], f"byol_new_v2_epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'online_encoder_state_dict': online_net.online_encoder.state_dict(),
                'online_projector_state_dict': online_net.online_projector.state_dict(),
                'online_predictor_state_dict': online_net.online_predictor.state_dict(),
                'target_encoder_state_dict': target_net.target_encoder.state_dict(),
                'target_projector_state_dict': target_net.target_projector.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'train_loss': avg_train_loss,
            }, checkpoint_path)
            print(f"Model saved to {checkpoint_path}\n")

        print("Training completed.")
        # Save final model
        final_path = os.path.join(self.config["save_dir"], "byol_new_v2_final.pth")
        torch.save({
            'online_encoder_state_dict': online_net.online_encoder.state_dict(),
            'online_projector_state_dict': online_net.online_projector.state_dict(),
            'online_predictor_state_dict': online_net.online_predictor.state_dict(),
            'target_encoder_state_dict': target_net.target_encoder.state_dict(),
            'target_projector_state_dict': target_net.target_projector.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, final_path)
        print(f"Final model saved to {final_path}")

# Main function
def main():
    import wandb
    from dotenv import load_dotenv

    load_dotenv()
    WANDB_API_KEY = os.getenv("WANDB_API_KEY")
    if WANDB_API_KEY:
        os.environ["WANDB_API_KEY"] = WANDB_API_KEY
        wandb.login(key=WANDB_API_KEY)
    else:
        print("WANDB_API_KEY not found. Proceeding without W&B logging.")

    # Configuration dictionary
    config = {
        "batch_size": 128,
        "epochs": 100,
        "learning_rate": 1e-3,
        "projection_dim": 128,
        "hidden_dim": 4096,
        "momentum": 0.996,
        "split_ratio": 0.9,  # Adjusted based on your dataset split needs
        "model_version": "new_v2",
        "save_dir": "./checkpoints_new_v2",
        "probe_lr": 0.0002,
        "dropout_prob": 0.0,
        "lambda_cov": 1.0,
    }

    # Initialize data augmentations
    config['augmentations'] = BYOLAugmentations(image_size=36)  # Adjust image_size based on your dataset

    # Initialize and start training
    wandb.init(
        project=f"BYOL_Project_{config['model_version']}",
        config=config
    )

    trainer = Trainer(config)
    trainer.train()

if __name__ == "__main__":
    main()
