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

from dataset import create_wall_dataloader
from models_md import JEPA_Model
from evaluator import ProbingEvaluator
import torch.multiprocessing as mp
import torch.distributed as dist


def get_device(local_rank):
    """Set the device for each process."""
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cpu')
    rank = dist.get_rank() if dist.is_initialized() else 0
    print(f"Process {rank} using device: {device}")
    return device


def load_data(batch_size, is_distributed, device):
    """
    Load the training data with or without DistributedSampler based on the distributed flag.

    Args:
        batch_size (int): Number of samples per batch.
        is_distributed (bool): Whether to use DistributedSampler.
        device (torch.device): The device to which data tensors are moved.

    Returns:
        DataLoader: The training data loader.
        DistributedSampler or None: The sampler used.
    """
    data_path = "/scratch/DL24FA"
    # data_path = "./data/DL24FA"

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


def save_model(model, optimizer, epoch, batch, save_path="checkpoints_2gpu", is_main_process=True):
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
    save_file = os.path.join(save_path, f"2gpu_jepa_model_epoch_{epoch}_batch_{batch}.pth")
    checkpoint = {
        'epoch': epoch,
        'batch': batch,
        'model_state_dict': model.module.state_dict() if isinstance(model, nn.parallel.DistributedDataParallel) else model.state_dict(),
    }
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    torch.save(checkpoint, save_file)
    rank = dist.get_rank() if dist.is_initialized() else 0
    print(f"[Rank {rank}] Checkpoint saved to {save_file}")


def load_checkpoint(model, optimizer, save_path="checkpoints_2gpu", is_main_process=True, device=torch.device('cuda')):
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
        checkpoint_files = glob(os.path.join(save_path, "2gpu_jepa_model_epoch_*_batch_*.pth"))
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
        checkpoint = torch.load(latest_checkpoint, map_location=device)  # Load on CPU first

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
    if dist.is_initialized():
        start_epoch_tensor = torch.tensor(start_epoch).to(device)
        start_batch_tensor = torch.tensor(start_batch).to(device)
        dist.broadcast(start_epoch_tensor, src=0)
        dist.broadcast(start_batch_tensor, src=0)
        start_epoch = start_epoch_tensor.item()
        start_batch = start_batch_tensor.item()

    return start_epoch, start_batch


def train_model(
    local_rank,
    world_size,
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
    """
    Train the model using DistributedDataParallel.

    Args:
        local_rank (int): The local rank of the process.
        world_size (int): Total number of processes.
        device (torch.device): The device to train on.
        model (nn.Module): The model to train.
        train_loader (DataLoader): The training data loader.
        num_epochs (int): Number of epochs to train.
        learning_rate (float): Learning rate for the optimizer.
        momentum (float): Momentum for the optimizer.
        save_every (int): Save checkpoint every 'save_every' epochs.
        save_every_batch (int): Save checkpoint every 'save_every_batch' batches.
        train_sampler (DistributedSampler or None): The sampler for the training data.
        distance_function (str): The distance function to use for loss computation.
    """
    # Wrap the model with DDP
    model.to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank] if torch.cuda.is_available() else None)

    # Initialize the optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Determine if the current process is the main process
    is_main_process = dist.get_rank() == 0

    # Attempt to load from the latest checkpoint
    start_epoch, start_batch = load_checkpoint(model, optimizer, save_path="checkpoints_2gpu", is_main_process=is_main_process, device=device)

    # Ensure all processes start from the same epoch and batch
    dist.barrier()

    model.train()

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

        # Use tqdm only on the main process for clean logging
        if is_main_process:
            pbar = tqdm(data_iter, desc=f"Epoch {epoch}")
        else:
            pbar = data_iter

        for batch_idx, batch in enumerate(pbar):
            global_batch_idx = batch_idx + 1 if epoch != start_epoch else batch_idx + start_batch + 1

            states = batch.states.to(device)  # [B, T, 2, 64, 64]
            actions = batch.actions.to(device)  # [B, T-1, 2]

            # Perform a training step using the energy function as loss
            loss = model.module.train_step(
                states=states,
                actions=actions,
                optimizer=optimizer,
                momentum=momentum,
                distance_function=distance_function,  # Use energy-based loss
            )

            epoch_loss += loss

            if is_main_process and global_batch_idx % save_every_batch == 0:
                print(
                    f"Epoch [{epoch}/{num_epochs}], Batch [{global_batch_idx}/{len(train_loader)}], Loss: {loss:.4f}"
                )
                save_model(model, optimizer, epoch, global_batch_idx, save_path="checkpoints_2gpu", is_main_process=True)

        # Average loss across all batches
        avg_epoch_loss = epoch_loss / len(train_loader)
        if is_main_process:
            print(f"Epoch [{epoch}/{num_epochs}] Average Loss: {avg_epoch_loss:.4f}")

        # Save model checkpoint at the end of the epoch
        if is_main_process and epoch % save_every == 0:
            save_model(model, optimizer, epoch, global_batch_idx, save_path="checkpoints_2gpu", is_main_process=True)

    if is_main_process:
        print("Training completed.")

    # Cleanup
    dist.destroy_process_group()

    return model


def main_worker(local_rank, world_size, args):
    """
    Main worker function for each process.

    Args:
        local_rank (int): The local rank of the process.
        world_size (int): Total number of processes.
        args (dict): Dictionary of training arguments.
    """
    # Initialize the process group if using distributed training
    if world_size > 1:
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=local_rank
        )

    # Set device for this process
    device = get_device(local_rank)

    # Load data with DistributedSampler if distributed
    train_loader, train_sampler = load_data(
        batch_size=args['batch_size'],
        is_distributed=(world_size > 1),
        device=device
    )

    # Initialize the JEPA model
    model = JEPA_Model(device=device, repr_dim=256, action_dim=2)

    # Train the model
    trained_model = train_model(
        local_rank=local_rank,
        world_size=world_size,
        device=device,
        model=model,
        train_loader=train_loader,
        num_epochs=args['num_epochs'],
        learning_rate=args['learning_rate'],
        momentum=args['momentum'],
        save_every=args['save_every'],
        save_every_batch=args['save_every_batch'],
        train_sampler=train_sampler,
        distance_function=args['distance_function']
    )

    # Save the final model only from the main process
    if dist.is_initialized():
        is_main_process = dist.get_rank() == 0
    else:
        is_main_process = True

    if is_main_process:
        # Detach the model from DDP before saving
        final_model = trained_model.module if isinstance(trained_model, nn.parallel.DistributedDataParallel) else trained_model
        save_model(final_model, optimizer=None, epoch="final", batch="final", save_path="checkpoints_2gpu", is_main_process=True)


def main():
    """
    Entry point for the training script.
    Determines whether to use distributed training based on available GPUs and launches training accordingly.
    """
    # Hyperparameters and configurations
    args = {
        'batch_size': 512,
        'num_epochs': 10,
        'learning_rate': 1e-3,
        'momentum': 0.99,
        'save_every': 1,
        'save_every_batch': 50,
        'distance_function': "l2"
    }

    # Determine the number of GPUs available
    if torch.cuda.is_available():
        world_size = torch.cuda.device_count()
    else:
        world_size = 1

    if world_size > 1:
        # Launch processes for DDP
        mp.spawn(
            main_worker,
            args=(world_size, args),
            nprocs=world_size,
            join=True
        )
    else:
        # Single process (no DDP)
        main_worker(0, world_size, args)


if __name__ == "__main__":
    main()
