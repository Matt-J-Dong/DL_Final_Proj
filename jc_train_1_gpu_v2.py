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
from normalizer import Normalizer

def get_device():
    """Set the device for single-GPU training."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    return device

def load_data(device, batch_size=64, split_ratio=0.8):
    data_path = "/scratch/DL24FA"

    # Create the full DataLoader
    full_loader = create_wall_dataloader(
        data_path=f"{data_path}/train",
        probing=False,
        device=device,
        train=True,
        batch_size=batch_size,
    )

    # Extract the dataset from the DataLoader
    full_dataset = full_loader.dataset

    # Split into train and validation datasets
    train_size = int(split_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Create DataLoaders for train and validation sets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

def save_model(model, epoch, save_path="checkpoints"):
    os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(save_path, f"jepa_model_epoch_{epoch}.pth")
    torch.save(model.state_dict(), save_file)
    print(f"Model saved to {save_file}")

def validate_model(model, val_loader, device, distance_function="l2", lambda_energy=1.0, lambda_var=1.0, lambda_cov=0.0):
    """
    Perform validation. Simplifies computation for speed by using a subset.
    """
    model.eval()
    val_loss = 0.0
    var_culm = 0.0
    cov_culm = 0.0
    normalizer = Normalizer()  # Initialize the normalizer

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating", leave=False):  # Minimize display overhead
            states = batch.states.to(device)
            actions = batch.actions.to(device)

            # Normalize the states
            locations = states[:, :, 0]  # Extract the 0th channel
            # print(f"Shape of locations before reshaping: {locations.shape}")  # Debugging
            locations = locations.reshape(-1, 2)  # Reshape to match normalization input
            states[:, :, 0] = normalizer.normalize_location(locations).reshape(states[:, :, 0].shape)


            # Simplify validation by reducing the temporal aspect of states/actions if applicable
            B, T, C, H, W = states.shape
            sampled_t = max(1, T // 2)  # Use one representative state-action pair to reduce computation
            sampled_states = states[:, :sampled_t]
            sampled_actions = actions[:, :sampled_t - 1]

            init_state = sampled_states[:, 0]
            predicted_encs = model.forward(init_state, sampled_actions)

            # Use only sampled target states for validation
            target_encs = []
            for t in range(sampled_t):
                o_t = sampled_states[:, t]
                s_target = model.target_encoder(o_t)
                target_encs.append(s_target)
            target_encs = torch.stack(target_encs, dim=1)

            # Compute loss for the sampled subset
            loss, _, var, cov = model.compute_loss(
                predicted_encs,
                target_encs,
                distance_function,
                debug=True,
                lambda_energy=lambda_energy,
                lambda_var=lambda_var,
                lambda_cov=lambda_cov,
            )
            val_loss += loss.item()
            var_culm += var
            cov_culm += cov

    print(f"Validation Variance: {var_culm / len(val_loader)}, (as percentage of loss: {var_culm / val_loss})")
    print(f"Validation Covariance: {cov_culm / len(val_loader)}, (as percentage of loss: {cov_culm / val_loss})")
    return val_loss / len(val_loader)

def train_model(
    device,
    model,
    train_loader,
    val_loader,
    num_epochs=1,
    learning_rate=1e-3,
    momentum=0.996,
    save_every=1,
    distance_function="l2",
    lambda_energy=1.0,
    lambda_var=1.0,
    lambda_cov=0.0,
):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    model.train()

    # Initialize normalizer
    normalizer = Normalizer()

    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}, Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        epoch_loss = 0.0

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            states = batch.states.to(device)
            actions = batch.actions.to(device)

            # Normalize location data (reshape to match Normalizer input)
            locations = states[:, :, 0]  # Extract the 0th channel
            # print(f"Shape of locations before reshaping: {locations.shape}")  # Debugging
            locations = locations.reshape(-1, 2)  # Reshape to match normalization input
            states[:, :, 0] = normalizer.normalize_location(locations).reshape(states[:, :, 0].shape)


            # Perform a training step
            loss = model.train_step(
                states=states,
                actions=actions,
                optimizer=optimizer,
                momentum=momentum,
                distance_function=distance_function,
                lambda_energy=lambda_energy,
                lambda_var=lambda_var,
                lambda_cov=lambda_cov,
            )
            epoch_loss += loss

            if batch_idx % 100 == 0:
                print(
                    f"Epoch [{epoch}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss:.4f}"
                )
                save_model(model, f"{epoch}_batch_{batch_idx}")

                # Perform validation
                val_loss = validate_model(model, val_loader, device, distance_function, lambda_energy, lambda_var, lambda_cov)
                print(f"Validation Loss (Batch {batch_idx}): {val_loss:.4f}")

        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch}/{num_epochs}] Average Loss: {avg_epoch_loss:.4f}")

        # Step the learning rate scheduler
        scheduler.step()

        if epoch % save_every == 0:
            save_model(model, epoch)

            # Perform validation at the end of each epoch
            val_loss = validate_model(model, val_loader, device, distance_function)
            print(f"Validation Loss (Epoch {epoch}): {val_loss:.4f}")

    print("Training completed.")
    return model

def main():
    device = get_device()
    batch_size = 512
    num_epochs = 10
    learning_rate = 2e-4
    momentum = 0.996
    split_ratio = 0.9
    lambda_energy, lambda_var, lambda_cov = 1.0, 0.0, 0.0  # Tunable hyperparameters

    mp.set_start_method('spawn')

    train_loader, val_loader = load_data(device, batch_size=batch_size, split_ratio=split_ratio)

    model = JEPA_Model(device=device, repr_dim=256, action_dim=2, dropout_prob=0)
    model.to(device)

    trained_model = train_model(
        device=device,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        momentum=momentum,
        save_every=1,
        lambda_energy=lambda_energy,
        lambda_var=lambda_var,
        lambda_cov=lambda_cov,
    )

    save_model(trained_model, "final")

if __name__ == "__main__":
    main()
