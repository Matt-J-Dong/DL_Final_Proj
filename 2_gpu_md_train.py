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
    print(f"Process {dist.get_rank()} using device: {device}")
    return device

def load_data(batch_size=64, is_distributed=False):
    data_path = "/scratch/DL24FA"
    # Load dataset using create_wall_dataloader which returns a DataLoader
    # To use a DistributedSampler, we need direct access to the dataset.
    # We will modify approach: we assume create_wall_dataloader returns a dataset if needed
    # If create_wall_dataloader doesn't allow that, we need a slight refactor.
    # For simplicity, let's assume create_wall_dataloader can return a dataset directly:
    # If not, you must implement a function that returns just the dataset.

    ds = create_wall_dataloader(
        data_path=f"{data_path}/train",
        probing=False,
        device='cpu',  # load on CPU only, no .to(device) in dataset
        return_dataset_only=True  # Assume we modified create_wall_dataloader to allow this
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
        # Non-distributed
        train_loader = torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )
        return train_loader, None

def save_model(model, epoch, save_path="checkpoints"):
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:  # Only main rank saves
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_file = os.path.join(save_path, f"jepa_model_epoch_{epoch}.pth")
        torch.save({
            'model_state_dict': model.state_dict(),
            'epoch': epoch
        }, save_file)
        print(f"Model saved to {save_file}")

def load_checkpoint(model, checkpoint_path):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Loaded checkpoint from {checkpoint_path}, starting from epoch {start_epoch+1}")
        return start_epoch
    else:
        print("No checkpoint found, starting from epoch 1")
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

    for epoch in range(start_epoch+1, start_epoch+num_epochs+1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        epoch_loss = 0.0

        iterator = tqdm(train_loader, desc=f"Epoch {epoch}", disable=(rank!=0))
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
                if (batch_idx == quarter_steps) or (batch_idx == 2*quarter_steps) or (batch_idx == 3*quarter_steps):
                    save_model(model, f"{epoch}_step_{batch_idx}")

                # Also print intermediate loss every 100 steps
                if batch_idx % 100 == 0:
                    print(f"Epoch [{epoch}], Batch [{batch_idx}/{num_steps}], Loss: {loss:.4f}")

        avg_epoch_loss = epoch_loss / num_steps
        if rank == 0:
            print(f"Epoch [{epoch}] Average Loss: {avg_epoch_loss:.4f}")

        # Save model checkpoint at the end of the epoch
        if rank == 0 and epoch % save_every == 0:
            save_model(model, f"{epoch}_final")

    if rank == 0:
        print("Training completed.")
    return model

def main():
    # Distributed init
    # If running single GPU without DDP, set WORLD_SIZE=1 and RANK=0 in env.
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.ge
