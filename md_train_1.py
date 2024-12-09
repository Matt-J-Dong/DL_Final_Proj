'''
singularity exec --nv --overlay /scratch/$USER/my_env/overlay-15GB-500K.ext3:ro /scratch/work/public/singularity/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif /bin/bash
source /ext3/env.sh
cd /scratch/dq2024/DL_Final_Proj/
'''

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
import os
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
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
    """
    Load the training data with or without DistributedSampler based on the distributed flag.

    Args:
        device (torch.device): The device to which data tensors are moved.
        batch_size (int): Number of samples per batch.
        is_distributed (bool): Whether to use DistributedSampler.
        subset_size (int): Number of samples to use from the dataset (optional).

    Returns:
        DataLoader: The training data loader.
        DistributedSampler or None: The sampler used.
    """
    data_path = "/scratch/DL24FA"

    # Initialize the dataset
    dataset = create_wall_dataloader(
        data_path=f"{data_path}/train",
        probing=False,
        device=device,
        train=True,
        batch_size=batch_size,
    ).dataset

    # Initialize the sampler if distributed
    train_sampler = DistributedSampler(dataset, shuffle=True) if is_distributed else None

    # Initialize the DataLoader
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, train_sampler


def save_model(model, optimizer, epoch, batch, save_path="checkpoints", is_main_process=True):
    """
    Save the model and optimizer state_dicts, along with the current epoch and batch.
    Only the main process should perform saving to avoid conflicts.

    Args:
        model (nn.Module): The model to save.
        optimizer (torch.optim.Optimizer or None): The optimizer to save.
        epoch (int or str): Current epoch number or a string identifier.
        batch (int or str): Current batch number or a string identifier.
        save_path (str): Directory path to save the checkpoint.
        is_main_process (bool): Flag indicating if the current process is the main process.
    """
    if not is_main_process:
        return

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_file = os.path.join(save_path, f"jepa_model_epoch_{epoch}_batch_{batch}.pth")
    checkpoint = {
        'epoch': epoch,
        'batch': batch,
        'model_state_dict': model.module.state_dict() if isinstance(model, nn.parallel.DistributedDataParallel) else model.state_dict(),
    }
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    torch.save(checkpoint, save_file)
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    print(f"[Rank {rank}] Checkpoint saved to {save_file}")


def load_checkpoint(model, optimizer, save_path="checkpoints", is_main_process=True, device=device):
    """
    Load the latest checkpoint from the save_path.
    Returns the starting epoch and batch. If no checkpoint is found, returns (1, 0).
    Only the main process performs loading. Other processes wait until loading is complete.

    Args:
        model (nn.Module): The model to load the state into.
        optimizer (torch.optim.Optimizer or None): The optimizer to load the state into.
        save_path (str): Directory path to load the checkpoint from.
        is_main_process (bool): Flag indicating if the current process is the main process.
        device (torch.device): The device to move tensors to.

    Returns:
        tuple: (start_epoch (int), start_batch (int))
    """
    if is_main_process:
        checkpoint_files = glob(os.path.join(save_path, "jepa_model_epoch_*_batch_*.pth"))
        if not checkpoint_files:
            print("[Main Process] No checkpoint found. Starting training from scratch.")
            return 1, 0  # Starting epoch and batch

        # Sort checkpoints by epoch and batch in descending order
        checkpoint_files.sort(key=lambda x: (
            int(x.split("_epoch_")[1].split("_")[0]),
            int(x.split("_batch_")[1].split(".pth")[0])
        ), reverse=True)

        latest_checkpoint = checkpoint_files[0]
        print(f"[Main Process] Loading checkpoint from {latest_checkpoint}")
        checkpoint = torch.load(latest_checkpoint, map_location=device)  # Load on the specified device

        model_state = checkpoint.get('model_state_dict', None)
        optimizer_state = checkpoint.get('optimizer_state_dict', None)
        epoch = checkpoint.get('epoch', 1)
        batch = checkpoint.get('batch', 0) + 1

        if model_state is None:
            print("[Main Process] 'model_state_dict' not found in checkpoint. Ignoring.")
            return 1, 0
        if optimizer_state is None and optimizer is not None:
            print("[Main Process] 'optimizer_state_dict' not found in checkpoint. Ignoring optimizer state.")

        # Load model state
        if isinstance(model, nn.parallel.DistributedDataParallel):
            model.module.load_state_dict(model_state)
        else:
            model.load_state_dict(model_state)

        if optimizer is not None and optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)

        # Broadcast the epoch and batch to all processes
        start_epoch = epoch
        start_batch = batch
    else:
        start_epoch = 1
        start_batch = 0

    # Broadcast the start_epoch and start_batch from main process to all other processes
    if torch.distributed.is_initialized():
        start_epoch_tensor = torch.tensor(start_epoch).to(device)
        start_batch_tensor = torch.tensor(start_batch).to(device)
        torch.distributed.broadcast(start_epoch_tensor, src=0)
        torch.distributed.broadcast(start_batch_tensor, src=0)
        start_epoch = start_epoch_tensor.item()
        start_batch = start_batch_tensor.item()

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
    start_epoch, start_batch = load_checkpoint(model, optimizer, save_path="checkpoints", is_main_process=True, device=device)

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
                save_model(model, optimizer, epoch, global_batch_idx, save_path="checkpoints", is_main_process=True)

        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch}/{num_epochs}] Average Loss: {avg_epoch_loss:.4f}")

        # Save model checkpoint at the end of the epoch
        if epoch % save_every == 0:
            save_model(model, optimizer, epoch, global_batch_idx, save_path="checkpoints", is_main_process=True)

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
    mp.set_start_method('spawn', force=True)

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
    if trained_model is not None:
        save_model(trained_model, optimizer=None, epoch="final", batch="final", save_path="checkpoints", is_main_process=True)


if __name__ == "__main__":
    main()
