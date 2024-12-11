import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
import os
from torch.utils.data import random_split, DataLoader
from dataset import create_wall_dataloader
from models_jc import JEPA_Model
import torch.multiprocessing as mp

def get_device():
    """Set the device for single-GPU training."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    return device

def load_data(device, batch_size=64, split_ratio=0.8):
    data_path = "/scratch/DL24FA"
    full_dataset = create_wall_dataloader(
        data_path=f"{data_path}/train",
        probing=False,
        device=device,
        train=True,
        batch_size=batch_size,
        return_dataset=True  # Modify this function to return the full dataset
    )

    train_size = int(split_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

def save_model(model, epoch, save_path="checkpoints"):
    os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(save_path, f"jepa_model_epoch_{epoch}.pth")
    torch.save(model.state_dict(), save_file)
    print(f"Model saved to {save_file}")

def validate_model(model, val_loader, device, distance_function="l2"):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            states = batch.states.to(device)
            actions = batch.actions.to(device)
            loss = model.compute_loss(
                states=states,
                actions=actions,
                distance_function=distance_function,
            )
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)
    print(f"Validation Loss: {avg_val_loss:.4f}")
    return avg_val_loss

def train_model(
    device,
    model,
    train_loader,
    val_loader,
    num_epochs=1,
    learning_rate=1e-3,
    momentum=0.99,
    save_every=1,
    train_sampler=None,
    distance_function="l2"
):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    model.train()

    for epoch in range(1, num_epochs + 1):
        if train_sampler:
            train_sampler.set_epoch(epoch)
        epoch_loss = 0.0

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
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

            if batch_idx % 100 == 0:
                print(
                    f"Epoch [{epoch}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss:.4f}"
                )
                save_model(model, f"{epoch}_batch_{batch_idx}")
                validate_model(model, val_loader, device, distance_function)

        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch}/{num_epochs}] Average Loss: {avg_epoch_loss:.4f}")

        # Step the scheduler at the end of each epoch
        scheduler.step()

        if epoch % save_every == 0:
            save_model(model, epoch)
            validate_model(model, val_loader, device, distance_function)

    print("Training completed.")
    return model


def main():
    device = get_device()
    batch_size = 512
    num_epochs = 10
    learning_rate = 1e-4
    momentum = 0.99

    mp.set_start_method('spawn')

    train_loader, val_loader = load_data(device, batch_size=batch_size)

    model = JEPA_Model(device=device, repr_dim=256, action_dim=2)
    model.to(device)

    trained_model = train_model(
        device=device,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        momentum=momentum,
        save_every=1
    )

    save_model(trained_model, "final")

if __name__ == "__main__":
    main()
