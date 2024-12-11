# main_md_3.py

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
import os
import glob
import torch.multiprocessing as mp

from dataset import create_wall_dataloader
from models_md_3 import JEPA_Model

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

def save_model(model, optimizer, epoch, batch_idx, save_path="checkpoints"):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_file = os.path.join(save_path, f"jepa_model_3_epoch_{epoch}_batch_{batch_idx}.pth")
    torch.save({
        'epoch': epoch,
        'batch_idx': batch_idx,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, save_file)
    print(f"Model saved to {save_file}")

def load_latest_checkpoint(model, optimizer, device, checkpoint_dir="checkpoints"):
    if not os.path.exists(checkpoint_dir):
        return 1, 0  # No checkpoint: start at epoch 1, batch 0
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "jepa_model_3_epoch_*_batch_*.pth"))
    if len(checkpoint_files) == 0:
        return 1, 0  # No checkpoint: start at epoch 1, batch 0

    # Sort by modification time and load the latest
    checkpoint_files.sort(key=os.path.getmtime)
    latest_checkpoint = checkpoint_files[-1]

    print(f"Loading from checkpoint: {latest_checkpoint}")
    checkpoint = torch.load(latest_checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Move model to the correct device after loading
    model.to(device)
    
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    start_epoch = checkpoint['epoch']
    start_batch_idx = checkpoint['batch_idx'] + 1  # resume from the next batch after the saved one
    return start_epoch, start_batch_idx

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
    dropout=0.1
):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    start_epoch, start_batch_idx = load_latest_checkpoint(model, optimizer, device, checkpoint_dir="checkpoints")
    model.to(device)
    model.train()

    for epoch in range(start_epoch, num_epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        epoch_loss = 0.0

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            if batch_idx < start_batch_idx and epoch == start_epoch:
                continue

            states = batch.states.to(device)
            actions = batch.actions.to(device)

            loss = model.train_step(
                states=states,
                actions=actions,
                optimizer=optimizer,
                momentum=momentum,
                distance_function=distance_function,
            )
            epoch_loss += loss

            # Save checkpoint every 50 batches
            if batch_idx % 50 == 0:
                print(
                    f"Epoch [{epoch}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss:.4f}"
                )
                save_model(model, optimizer, epoch, batch_idx)

        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch}/{num_epochs}] Average Loss: {avg_epoch_loss:.4f}")

        # After finishing an epoch, reset start_batch_idx
        start_batch_idx = 0

        # Save model checkpoint every epoch
        if epoch % save_every == 0:
            save_model(model, optimizer, epoch, -1)

        # Step the scheduler after each epoch
        scheduler.step()

    print("Training completed.")
    return model

def main():
    device = get_device()

    batch_size = 512
    num_epochs = 10
    learning_rate = 1e-4
    momentum = 0.99
    dropout = 0.1

    mp.set_start_method('spawn', force=True)

    train_loader, train_sampler = load_data(device, batch_size=batch_size, is_distributed=False)

    model = JEPA_Model(device=device, repr_dim=256, action_dim=2, dropout=dropout)
    model.to(device)

    trained_model = train_model(
        device=device,
        model=model,
        train_loader=train_loader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        momentum=momentum,
        save_every=1,
        train_sampler=train_sampler,
        dropout=dropout
    )

    # Save the final model
    optimizer = optim.Adam(trained_model.parameters(), lr=learning_rate)
    save_model(trained_model, optimizer, "final_epoch", -1)

if __name__ == "__main__":
    main()
