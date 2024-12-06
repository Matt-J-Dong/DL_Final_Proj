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
from torch.utils.data import Subset

from dataset import create_wall_dataloader
from models import JEPA_Model
from evaluator import ProbingEvaluator
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
    data_path = "./data/DL24FA"

    train_loader = create_wall_dataloader(
        data_path=f"{data_path}/train",
        probing=False,
        device=device,
        train=True,
        batch_size=batch_size,
    )

    # Create a subset of the dataset for testing (if desired)
    # train_dataset = train_loader.dataset

    # if we want to test with smaller subset of data
    # indices = list(range(subset_size))
    # train_dataset = Subset(train_dataset, indices)

    # Without distributed, just use a standard DataLoader
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     batch_size=batch_size,
    #     shuffle=True,
    # )

    return train_loader, None

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

    for epoch in range(1, num_epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        epoch_loss = 0.0

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            states = batch.states # [B, T, 2, 64, 64]
            actions = batch.actions # [B, T-1, 2]

            # Perform a training step
            loss = model.train_step(
                states=states,
                actions=actions,
                criterion=criterion,
                optimizer=optimizer,
                momentum=momentum,
            )

            epoch_loss += loss

            if batch_idx % 100 == 0:
                print(
                    f"Epoch [{epoch}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss:.4f}"
                )

        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch}/{num_epochs}] Average Loss: {avg_epoch_loss:.4f}")

        # Save model checkpoint
        if epoch % save_every == 0:
            save_model(model, epoch)

    print("Training completed.")
    return model

def main():
    device = get_device()

    batch_size = 512
    num_epochs = 10
    learning_rate = 1e-3
    momentum = 0.99

    # for multiprocessing
    #mp.set_start_method('spawn', force=True)

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
        train_sampler=train_sampler
    )

    # Save the final model
    save_model(trained_model, "final")

if __name__ == "__main__":
    main()
