# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
import torch.distributed as dist
import torch.multiprocessing as mp
import os  # Import os to access environment variables
from torch.utils.data import Subset


from dataset import create_wall_dataloader
from models import JEPA_Model
from evaluator import ProbingEvaluator
import os  # Already imported


def get_device(local_rank):
    """Set the device for distributed training."""
    print(f"Process {local_rank} is setting up its device.")
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    print(f"Process {local_rank} using device: {device}")
    return device

def load_data(device, batch_size=64, is_distributed=False, subset_size=1000):
    data_path = "./data/DL24FA"

    train_loader = create_wall_dataloader(
        data_path=f"{data_path}/train",
        probing=False,
        device=device,
        train=True,
        batch_size=batch_size,  # Ensure batch_size is set here
    )

    # Access the dataset from the DataLoader
    train_dataset = train_loader.dataset

    # TODO: Create a subset of the dataset for testing
    # indices = list(range(subset_size))
    # train_dataset = Subset(train_dataset, indices)

    if is_distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        # Re-create DataLoader with the distributed sampler
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
        )
    else:
        train_sampler = None
        # If not distributed, use the original DataLoader
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
        )

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
    train_sampler=None
):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()

    rank = dist.get_rank()

    for epoch in range(1, num_epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        epoch_loss = 0.0

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}", disable=(rank != 0))):
            states = batch.states.to(device)  # [B, T, 2, 64, 64]
            actions = batch.actions.to(device)  # [B, T-1, 2]

            #print(f"Initial state shape: {states[:, 0].shape}") --> debug

            # Perform a training step
            loss = model.module.train_step(
                states=states,
                actions=actions,
                criterion=criterion,
                optimizer=optimizer,
                momentum=momentum,
            )

            epoch_loss += loss

            # Debugging print statements
            if batch_idx % 100 == 0 and rank == 0:
                print(
                    f"Epoch [{epoch}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss:.4f}"
                )

        avg_epoch_loss = epoch_loss / len(train_loader)
        if rank == 0:
            print(f"Epoch [{epoch}/{num_epochs}] Average Loss: {avg_epoch_loss:.4f}")

        # Save model checkpoint
        if epoch % save_every == 0 and rank == 0:
            save_model(model.module, epoch)

    if rank == 0:
        print("Training completed.")

    return model


def main():
    # Get local_rank from environment variable
    local_rank = int(os.environ["LOCAL_RANK"])
    main_worker(local_rank)


def main_worker(local_rank):
    dist.init_process_group(backend='nccl')

    device = get_device(local_rank)

    batch_size = 1024
    num_epochs = 10
    learning_rate = 1e-3
    momentum = 0.99
    r = 4 #lora rank
    alpha = 1.0 #lora scaling factor

    # Load data
    train_loader, train_sampler = load_data(device, batch_size=batch_size, is_distributed=True)

    # Initialize the JEPA model
    #model = JEPA_Model(device=device, repr_dim=256, action_dim=2,r=r, alpha=alpha)
    model = JEPA_Model(device=device, repr_dim=256, action_dim=2)
    print(device)
    model.to(device)

    # Wrap the model with DistributedDataParallel
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], output_device=local_rank
    )

    # Train the model
    trained_model = train_model(
        device=device,
        model=model,
        train_loader=train_loader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        momentum=momentum,
        save_every=1,
        train_sampler=train_sampler
    )

    # Optionally, save the final model (only on rank 0)
    if dist.get_rank() == 0:
        save_model(trained_model.module, "final")


if __name__ == "__main__":
    main()
