import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
import os
from torch.utils.data import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from dataset import create_wall_dataloader
from models_md import JEPA_Model
from evaluator import ProbingEvaluator

def get_device(local_rank):
    """Set the device for multi-GPU distributed training."""
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    rank = dist.get_rank() if dist.is_initialized() else 0
    print(f"Process {rank} using device: {device}")
    return device

def load_data(device, batch_size=64, is_distributed=False):
    data_path = "/scratch/DL24FA"
    # Assuming create_wall_dataloader now returns the dataset if return_dataset_only is True
    ds = create_wall_dataloader(
        data_path=f"{data_path}/train",
        probing=False,
        device='cpu',  # load dataset on CPU
        batch_size=512,
        train=True
    )

    if is_distributed:
        sampler = DistributedSampler(ds, shuffle=True)
        train_loader = torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )
        return train_loader, sampler
    else:
        train_loader = torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )
        return train_loader, None

def save_model(model, epoch, save_path="checkpoints", step=None):
    """
    Save the model checkpoint.

    Args:
        model: The model to save.
        epoch (int): Current epoch number.
        save_path (str): Directory to save the checkpoint.
        step (int, optional): Current step within the epoch. Defaults to None.
    """
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:  # Only main rank saves
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if step is not None:
            save_file = os.path.join(save_path, f"jepa_model_epoch_{epoch}_step_{step}.pth")
        else:
            save_file = os.path.join(save_path, f"jepa_model_epoch_{epoch}.pth")
        torch.save({
            'model_state_dict': model.state_dict(),
            'epoch': epoch
        }, save_file)
        print(f"Model saved to {save_file}")

def load_checkpoint(model, checkpoint_path, is_distributed):
    """
    Load the model checkpoint.

    Args:
        model: The model to load the state_dict into.
        checkpoint_path (str): Path to the checkpoint file.
        is_distributed (bool): Whether the model is wrapped in DDP.

    Returns:
        int: The starting epoch.
    """
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if is_distributed:
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Loaded checkpoint from {checkpoint_path}, starting from epoch {start_epoch + 1}")
        return start_epoch
    else:
        print("No checkpoint found, starting from epoch 0")
        return 0

def train_model(
    device,
    model,
    train_loader,
    num_epochs=1,
    learning_rate=1e-3,
    momentum=0.99,
    save_every=1,
    train_sampler=None,
    distance_function="l2",
    start_epoch=0
):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()

    rank = dist.get_rank() if dist.is_initialized() else 0
    num_steps = len(train_loader)
    quarter_steps = num_steps // 4

    for epoch in range(start_epoch + 1, start_epoch + num_epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        epoch_loss = 0.0

        # Only rank 0 will have tqdm progress bar
        if rank == 0:
            iterator = tqdm(train_loader, desc=f"Epoch {epoch}", total=num_steps)
        else:
            iterator = train_loader

        for batch_idx, batch in enumerate(iterator):
            states = batch.states.to(device, non_blocking=True)  # [B, T, 2, 64, 64]
            actions = batch.actions.to(device, non_blocking=True) # [B, T-1, 2]

            # Perform a training step using the energy function as loss
            loss = model.train_step(
                states=states,
                actions=actions,
                optimizer=optimizer,
                momentum=momentum,
                distance_function=distance_function
            )

            epoch_loss += loss

            # Save checkpoints at quarter intervals
            if rank == 0:
                if (batch_idx + 1) in [quarter_steps, 2 * quarter_steps, 3 * quarter_steps]:
                    save_model(model, epoch, step=batch_idx + 1)

                # Also print intermediate loss every 100 steps
                if (batch_idx + 1) % 100 == 0:
                    print(f"Epoch [{epoch}], Batch [{batch_idx + 1}/{num_steps}], Loss: {loss:.4f}")

            # Update tqdm progress bar
            if rank == 0:
                iterator.set_postfix(loss=loss.item())

        avg_epoch_loss = epoch_loss / num_steps
        if rank == 0:
            print(f"Epoch [{epoch}] Average Loss: {avg_epoch_loss:.4f}")

        # Save model checkpoint at the end of the epoch
        if rank == 0 and epoch % save_every == 0:
            save_model(model, epoch)

    if rank == 0:
        print("Training completed.")
    return model

def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))

    is_distributed = (world_size > 1)
    if is_distributed:
        dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)

    device = get_device(local_rank)

    batch_size = 512
    num_epochs = 10
    learning_rate = 1e-3
    momentum = 0.99

    train_loader, train_sampler = load_data(device, batch_size=batch_size, is_distributed=is_distributed)

    model = JEPA_Model(device=device, repr_dim=256, action_dim=2).to(device)
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    # Load from checkpoint if exists
    checkpoint_path = "./checkpoints/jepa_model_epoch_10.pth"  # Example, adjust as needed
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        # Only load on rank 0 then broadcast
        if rank == 0:
            print(f"Rank 0 loading checkpoint from {checkpoint_path}")
            start_epoch = load_checkpoint(model, checkpoint_path, is_distributed)
        # Broadcast start_epoch to all ranks
        if is_distributed:
            start_epoch_tensor = torch.tensor(start_epoch, device=device)
            dist.broadcast(start_epoch_tensor, src=0)
            start_epoch = start_epoch_tensor.item()

    # Train the model
    trained_model = train_model(
        device=device,
        model=model,
        train_loader=train_loader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        momentum=momentum,
        save_every=1,
        train_sampler=train_sampler,
        distance_function="l2",
        start_epoch=start_epoch
    )

    # Save the final model (only rank 0)
    if rank == 0:
        save_model(trained_model.module if is_distributed else trained_model, epoch="final")

    # Cleanup DDP
    if is_distributed:
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    # Ensure that the start method is set before any multiprocessing starts
    mp.set_start_method('spawn', force=True)
    main()
