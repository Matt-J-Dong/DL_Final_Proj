import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
import os
from torch.utils.data import Subset
import glob

from dataset import create_wall_dataloader
from models import JEPA_ViTModel  # Use our new JEPA ViT-based JEPA model

def load_data(device, batch_size=2048):
    data_path = "./data/DL24FA"
    train_loader = create_wall_dataloader(
        data_path=f"{data_path}/train",
        probing=False,
        device=device,  # device is passed but dataset won't use it for GPU ops
        train=True,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=True,
    )
    return train_loader, None

def save_model(model, optimizer, epoch, save_path="checkpoints"):
    """
    Saves the model state, optimizer state, and current epoch to a checkpoint file.
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_file = os.path.join(save_path, f"jepa_md_model_epoch_{epoch}.pth") 
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, save_file)
    print(f"Checkpoint saved to {save_file}")

def load_checkpoint(model, optimizer, save_path="checkpoints"):
    """
    Loads the latest checkpoint from the save_path directory.
    Returns the epoch to resume from.
    """
    checkpoint_files = glob.glob(os.path.join(save_path, "jepa_md_model_epoch_*.pth"))
    if not checkpoint_files:
        print("No checkpoint found. Starting training from scratch.")
        return 1  # Start from epoch 1

    # Extract epoch numbers and find the latest checkpoint
    def extract_epoch(file_path):
        basename = os.path.basename(file_path)
        epoch_str = basename.replace("jepa_md_model_epoch_", "").replace(".pth", "")
        try:
            return int(epoch_str)
        except ValueError:
            return -1  # Invalid epoch number

    latest_checkpoint = max(checkpoint_files, key=extract_epoch)
    latest_epoch = extract_epoch(latest_checkpoint)
    if latest_epoch == -1:
        print("No valid checkpoint found. Starting training from scratch.")
        return 1

    # Load the checkpoint
    checkpoint = torch.load(latest_checkpoint, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Loaded checkpoint from {latest_checkpoint} (Epoch {latest_epoch})")
    return latest_epoch + 1  # Resume from the next epoch

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    batch_size = 2048
    total_epochs = 10
    learning_rate = 1e-3
    save_every = 1
    momentum = 0.99  # For target encoder momentum update

    train_loader, train_sampler = load_data(device, batch_size=batch_size)

    # Initialize the JEPA ViT Model
    model = JEPA_ViTModel(device=device, repr_dim=256, action_dim=2, img_size=64)
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Attempt to load from the latest checkpoint
    starting_epoch = 1
    save_path = "checkpoints"
    starting_epoch = load_checkpoint(model, optimizer, save_path=save_path)

    model.train()

    for epoch in range(starting_epoch, total_epochs + 1):
        epoch_loss = 0.0

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{total_epochs}")):
            # Move data to GPU here
            states = batch.states.to(device, non_blocking=True)   # [B,T,C,H,W]
            actions = batch.actions.to(device, non_blocking=True) # [B,T-1,2]

            B, T, C, H, W = states.shape

            # Forward pass: predict embeddings for the entire sequence
            # pred_encs: [B,T,D]
            pred_encs = model(states, actions)

            # Compute target embeddings using target encoder for each timestep
            target_list = []
            for t in range(T):
                # states[:,t]: [B,C,H,W]
                s_target = model.target_encoder(states[:, t])  # [B,D]
                target_list.append(s_target)
            target_encs = torch.stack(target_list, dim=1)  # [B,T,D]

            # Compute JEPA loss (MSE between predicted and target embeddings)
            loss = criterion(pred_encs, target_encs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Momentum update the target encoder
            model.update_target_encoder(momentum=momentum)

            loss_val = loss.item()
            epoch_loss += loss_val

            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch}/{total_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss:.4f}")

        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch}/{total_epochs}] Average Loss: {avg_epoch_loss:.4f}")

        if epoch % save_every == 0:
            save_model(model, optimizer, epoch, save_path=save_path)

    print("Training completed.")
    save_model(model, optimizer, "final", save_path=save_path)

if __name__ == "__main__":
    main()
