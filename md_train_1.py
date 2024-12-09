import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
import os
from torch.utils.data import Subset
from glob import glob
import matplotlib.pyplot as plt

from dataset import create_wall_dataloader
from models_md import JEPA_Model
import torch.multiprocessing as mp


def get_device():
    """Set the device for single-GPU training."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    return device


def load_data(device, batch_size=64, is_distributed=False, subset_size=1000):
    data_path = "/scratch/DL24FA"

    train_loader = create_wall_dataloader(
        data_path=f"{data_path}/train",
        probing=False,
        device=device,
        train=True,
        batch_size=batch_size,
    )

    return train_loader, None


def save_model(model, optimizer, epoch, batch, save_path="checkpoints"):
    """Save the model and optimizer state_dicts, along with the current epoch and batch."""
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_file = os.path.join(save_path, f"jepa_model_epoch_{epoch}_batch_{batch}.pth")
    checkpoint = {
        'epoch': epoch,
        'batch': batch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, save_file)
    print(f"Checkpoint saved to {save_file}")


def load_checkpoint(model, optimizer, save_path="checkpoints"):
    """Load the latest checkpoint from the save_path."""
    checkpoint_files = glob(os.path.join(save_path, "jepa_model_epoch_*_batch_*.pth"))
    if not checkpoint_files:
        print("No checkpoint found. Starting training from scratch.")
        return 1, 0  # Starting epoch and batch

    # Sort checkpoints by epoch and batch
    checkpoint_files.sort(key=lambda x: (
        int(x.split("_epoch_")[1].split("_")[0]),
        int(x.split("_batch_")[1].split(".pth")[0])
    ), reverse=True)

    latest_checkpoint = checkpoint_files[0]
    print(f"Loading checkpoint from {latest_checkpoint}")
    checkpoint = torch.load(latest_checkpoint, map_location=model.device)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    start_batch = checkpoint['batch'] + 1  # Start from the next batch
    print(f"Resuming from epoch {start_epoch}, batch {start_batch}")
    return start_epoch, start_batch


def plot_loss(losses, save_path="loss_graph.png"):
    """Plot and save the loss graph."""
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label="Training Loss", color='blue')
    plt.title("Training Loss Over Time")
    plt.xlabel("Batch Number")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    print(f"Loss graph saved to {save_path}")


def train_model(
    device,
    model,
    train_loader,
    num_epochs=1,
    learning_rate=1e-3,
    momentum=0.99,
    save_every=1,
    save_every_batch=50,
    train_sampler=None,
    distance_function="l2"  # Specify the distance function for the energy loss
):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Attempt to load from the latest checkpoint
    start_epoch, start_batch = load_checkpoint(model, optimizer, save_path="checkpoints")

    model.train()
    losses = []  # To store loss values for plotting

    for epoch in range(start_epoch, num_epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        epoch_loss = 0.0

        # If resuming from a checkpoint, adjust the iterator to start from the correct batch
        if epoch == start_epoch and start_batch > 0:
            data_iter = iter(train_loader)
            for _ in range(start_batch):
                try:
                    next(data_iter)
                except StopIteration:
                    break
        else:
            data_iter = iter(train_loader)
            start_batch = 0  # Reset start_batch for new epochs

        for batch_idx, batch in enumerate(tqdm(data_iter, desc=f"Epoch {epoch}")):
            global_batch_idx = batch_idx + 1 if epoch != start_epoch else batch_idx + start_batch + 1

            states = batch.states.to(device)  # [B, T, 2, 64, 64]
            actions = batch.actions.to(device)  # [B, T-1, 2]

            # Perform a training step using the energy function as loss
            loss = model.train_step(
                states=states,
                actions=actions,
                optimizer=optimizer,
                momentum=momentum,
                distance_function=distance_function,  # Use energy-based loss
            )

            epoch_loss += loss
            losses.append(loss.item())  # Save the loss for plotting

            if global_batch_idx % save_every_batch == 0:
                print(
                    f"Epoch [{epoch}/{num_epochs}], Batch [{global_batch_idx}/{len(train_loader)}], Loss: {loss:.4f}"
                )
                save_model(model, optimizer, epoch, global_batch_idx)

        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch}/{num_epochs}] Average Loss: {avg_epoch_loss:.4f}")

        # Save model checkpoint at the end of the epoch
        if epoch % save_every == 0:
            save_model(model, optimizer, epoch, global_batch_idx)

    # Plot and save the loss graph
    plot_loss(losses)

    print("Training completed.")
    return model


def main():
    device = get_device()

    batch_size = 512
    num_epochs = 10
    learning_rate = 1e-3
    momentum = 0.99

    # for multiprocessing
    mp.set_start_method('spawn')

    # Load data (not distributed)
    train_loader, train_sampler = load_data(device, batch_size=batch_size, is_distributed=False)

    # Initialize the JEPA model
    model = JEPA_Model(device=device, repr_dim=256, action_dim=2)
    model.to(device)

    # Train the model
    trained_model = train_model(
        device=device,
        model=model,
        train_loader=train_loader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        momentum=momentum,
        save_every=1,
        save_every_batch=50,  # Save every 50 batches
        train_sampler=train_sampler
    )

    # Save the final model
    save_model(trained_model, optimizer=None, epoch="final", batch="final")


if __name__ == "__main__":
    main()
