import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
import os
import glob
import torch.multiprocessing as mp

from dataset_md import create_wall_dataloader
from models_md_4 import JEPA_Model

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
    if batch_idx == -1:
        # Use a "final" suffix for end-of-epoch checkpoint
        save_file = os.path.join(save_path, f"jepa_model_4_epoch_{epoch}_final.pth")
    else:
        save_file = os.path.join(save_path, f"jepa_model_4_epoch_{epoch}_batch_{batch_idx}.pth")
    torch.save({
        'epoch': epoch,
        'batch_idx': batch_idx,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, save_file)
    print(f"Model saved to {save_file}")

def load_latest_checkpoint(model, optimizer, checkpoint_dir="checkpoints"):
    if not os.path.exists(checkpoint_dir):
        return 1, 0  # No checkpoint: start at epoch 1, batch 0
    # Include both final and batch checkpoints
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "jepa_model_epoch_*_batch_*.pth")) + \
                       glob.glob(os.path.join(checkpoint_dir, "jepa_model_epoch_*_final.pth"))
    if len(checkpoint_files) == 0:
        return 1, 0  # No checkpoint: start at epoch 1, batch 0

    # Sort by modification time and load the latest
    checkpoint_files.sort(key=os.path.getmtime)
    latest_checkpoint = checkpoint_files[-1]

    print(f"Loading from checkpoint: {latest_checkpoint}")
    checkpoint = torch.load(latest_checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    start_epoch = checkpoint['epoch']
    start_batch_idx = checkpoint['batch_idx']
    if start_batch_idx != -1:
        start_batch_idx += 1  # resume from next batch after the saved one
    else:
        # If it's a final checkpoint, start at next epoch from batch 0
        start_epoch += 1
        start_batch_idx = 0
    return start_epoch, start_batch_idx

def validate_model(model, val_loader, device, distance_function="l2"):
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for batch in val_loader:
            states = batch.states.to(device)
            actions = batch.actions.to(device)
            B, T, C, H, W = states.shape
            # Forward pass to get predictions
            init_state = states[:, 0]
            pred_encs = model.forward(init_state, actions)

            # Generate target embeddings
            target_encs = []
            for t in range(T):
                o_t = states[:, t]
                s_target = model.target_encoder(o_t)
                target_encs.append(s_target)
            target_encs = torch.stack(target_encs, dim=1)

            # Compute energy
            loss = model.compute_energy(pred_encs, target_encs, distance_function)
            total_loss += loss.item()
            count += 1
    model.train()
    return total_loss / count if count > 0 else float('inf')


def train_model(
    device,
    model,
    train_loader,
    probe_val_normal_ds,
    probe_val_wall_ds,
    num_epochs=1,
    learning_rate=1e-3,
    momentum=0.99,
    save_every=1,
    train_sampler=None,
    distance_function="l2",
    dropout=0.1
):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # Example scheduler: reduce LR by 0.1 every 5 epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    start_epoch, start_batch_idx = load_latest_checkpoint(model, optimizer, checkpoint_dir="checkpoints")
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

        # Validation step
        val_loss_normal = validate_model(model, probe_val_normal_ds, device, distance_function=distance_function)
        val_loss_wall = validate_model(model, probe_val_wall_ds, device, distance_function=distance_function)
        print(f"Validation Loss (Normal): {val_loss_normal:.4f}")
        print(f"Validation Loss (Wall): {val_loss_wall:.4f}")

        # Reset start_batch_idx for next epoch
        start_batch_idx = 0

        # Save model checkpoint at the end of the epoch
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

    # Load training data
    train_loader, train_sampler = load_data(device, batch_size=batch_size, is_distributed=False)

    # Load validation data
    data_path = "/scratch/DL24FA"
    probe_val_normal_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_normal/val",
        probing=True,
        device=device,
        train=False,
        batch_size=batch_size
    )

    probe_val_wall_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_wall/val",
        probing=True,
        device=device,
        train=False,
        batch_size=batch_size
    )

    model = JEPA_Model(device=device, repr_dim=256, action_dim=2, dropout=dropout)
    model.to(device)

    trained_model = train_model(
        device=device,
        model=model,
        train_loader=train_loader,
        probe_val_normal_ds=probe_val_normal_ds,
        probe_val_wall_ds=probe_val_wall_ds,
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