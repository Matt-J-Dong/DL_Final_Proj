# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
import os

from dataset import create_wall_dataloader
from models_md_q import SingleLinearModel  # Updated import to match models_md_q.py
from torch.utils.data import Subset
import random

def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    return device

def compute_energy(predicted_encs, target_encs, distance_function="l2"):
    if distance_function == "l2":
        energy = torch.sum((predicted_encs - target_encs) ** 2) / (predicted_encs.size(0))
    elif distance_function == "cosine":
        cos = nn.CosineSimilarity(dim=-1)
        energy = -torch.sum(cos(predicted_encs, target_encs)) / (predicted_encs.size(0))
    else:
        raise ValueError(f"Unknown distance function: {distance_function}")
    return energy

def main():
    device = get_device()

    # Hyperparameters
    batch_size = 64
    learning_rate = 1e-3
    num_epochs = 5
    input_dim = 8450
    output_dim = 256
    momentum = 0.99  # Momentum for target encoder

    # Load full dataset first
    data_path = "/scratch/DL24FA"
    full_loader = create_wall_dataloader(
        data_path=f"{data_path}/train",
        probing=False,
        device=device,
        train=True,
        batch_size=1  # Set to 1 temporarily to access individual samples
    )

    # Extract 1% subset
    dataset_size = len(full_loader.dataset)  # Assuming the dataset is attached to the DataLoader
    subset_size = max(1, int(dataset_size * 0.01))  # 1% of the dataset
    subset_indices = random.sample(range(dataset_size), subset_size)  # Randomly sample indices
    subset = Subset(full_loader.dataset, subset_indices)  # Create a subset

    # Create a DataLoader for the subset
    train_loader = torch.utils.data.DataLoader(
        subset, batch_size=batch_size, shuffle=True
    )

    # Initialize model
    model = SingleLinearModel(input_dim=input_dim, output_dim=output_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize target encoder as a copy of the model
    target_encoder = SingleLinearModel(input_dim=input_dim, output_dim=output_dim).to(device)
    target_encoder.load_state_dict(model.state_dict())
    for param in target_encoder.parameters():
        param.requires_grad = False  # Freeze target encoder

    model.train()
    target_encoder.eval()

    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
        for batch_idx, batch in enumerate(progress_bar):
            states = batch.states.to(device)   # states: [B, T, C, H, W]
            actions = batch.actions.to(device) # actions: [B, T-1, action_dim]

            B, T, C, H, W = states.shape
            # Flatten the initial state (e.g., states[:,0]) into 8450-dim vector
            # Assuming states[:,0] has shape [B, C, H, W], we flatten it
            init_state = states[:, 0].view(B, -1)
            assert init_state.size(1) == input_dim, f"Expected init_state to have {input_dim} features, got {init_state.size(1)}"

            # Forward pass through the main model
            predicted_encs = model(init_state)  # [B, 256]

            with torch.no_grad():
                # Forward pass through the target encoder
                target_encs = target_encoder(init_state)  # [B, 256]

            # Compute energy-based loss
            loss = compute_energy(predicted_encs, target_encs, distance_function="l2")

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update target encoder with momentum
            with torch.no_grad():
                for param_q, param_k in zip(model.parameters(), target_encoder.parameters()):
                    param_k.data = momentum * param_k.data + (1.0 - momentum) * param_q.data

            epoch_loss += loss.item()
            progress_bar.set_postfix({"Loss": loss.item()})

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch}, Average Loss: {avg_loss:.6f}")

    print("Training completed.")

if __name__ == "__main__":
    main()
