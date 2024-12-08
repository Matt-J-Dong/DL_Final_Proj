import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm.auto import tqdm
import os
import torch.multiprocessing as mp
import torch.distributed as dist

from dataset import create_wall_dataloader
from models_md import JEPA_Model


def get_device(local_rank):
    """Set the device for multi-GPU training."""
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def load_data(device, batch_size=64, is_distributed=False, subset_size=1000, rank=0, world_size=1):
    data_path = "/scratch/DL24FA"

    train_loader = create_wall_dataloader(
        data_path=f"{data_path}/train",
        probing=False,
        device=device,
        train=True,
        batch_size=batch_size,
    )

    if is_distributed:
        train_sampler = DistributedSampler(
            train_loader.dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
        train_loader = DataLoader(
            train_loader.dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True,
        )
    else:
        train_sampler = None

    return train_loader, train_sampler


def save_model(model, epoch, save_path="checkpoints"):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_file = os.path.join(save_path, f"jepa_model_epoch_{epoch}.pth")
    torch.save(model.state_dict(), save_file)
    print(f"Model saved to {save_file}")


def train_model(
    device,
    model,
    train_loader,
    num_epochs=1,
    learning_rate=1e-3,
    momentum=0.99,
    save_every=1,
    train_sampler=None,
    distance_function="l2",  # Specify the distance function for the energy loss
    rank=0,
):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()

    for epoch in range(1, num_epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        epoch_loss = 0.0

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}", disable=(rank != 0))):
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

            epoch_loss += loss.item()

            if batch_idx % 100 == 0 and rank == 0:
                print(
                    f"Epoch [{epoch}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss:.4f}"
                )
                save_model(model.module, f"{epoch}_batch_{batch_idx}")

        avg_epoch_loss = epoch_loss / len(train_loader)
        if rank == 0:
            print(f"Epoch [{epoch}/{num_epochs}] Average Loss: {avg_epoch_loss:.4f}")

        # Save model checkpoint
        if epoch % save_every == 0 and rank == 0:
            save_model(model.module, f"{epoch}_final")

    if rank == 0:
        print("Training completed.")
    return model


def setup_ddp(rank, world_size):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """Clean up the distributed environment."""
    dist.destroy_process_group()


def main_ddp(rank, world_size):
    setup_ddp(rank, world_size)

    device = get_device(rank)

    batch_size = 512 // world_size
    num_epochs = 10
    learning_rate = 1e-3
    momentum = 0.99

    # Load data in a distributed way
    train_loader, train_sampler = load_data(
        device, batch_size=batch_size, is_distributed=True, rank=rank, world_size=world_size
    )

    # Initialize the JEPA model and wrap with DDP
    model = JEPA_Model(device=device, repr_dim=256, action_dim=2).to(device)
    model = DDP(model, device_ids=[rank])

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
        rank=rank,
    )

    if rank == 0:
        save_model(trained_model.module, "final")

    cleanup_ddp()


def main():
    world_size = 2  # Number of GPUs
    mp.spawn(main_ddp, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
