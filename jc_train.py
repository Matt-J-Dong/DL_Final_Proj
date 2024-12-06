# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

from dataset import create_wall_dataloader
from models import JEPA_Model
from evaluator import ProbingEvaluator
import os


def get_device():
    """Check for GPU availability."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    return device


def load_data(device, batch_size=64):
    data_path = " /scratch/DL24FA/"

    train_ds = create_wall_dataloader(
        data_path=f"{data_path}/train",
        probing=False,
        device=device,
        batch_size=batch_size,
        train=True,
    )

    return train_ds


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
    num_epochs=10,
    learning_rate=1e-3,
    momentum=0.99,
    save_every=1,
):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model = model.to(device)
    model.train()

    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            states = batch.states.to(device)  # [B, T, 2, 64, 64]
            actions = batch.actions.to(device)  # [B, T-1, 2]

            # Perform a training step
            loss = model.train_step(
                states=states,
                actions=actions,
                criterion=criterion,
                optimizer=optimizer,
                momentum=momentum,
            )

            epoch_loss += loss

            # Debugging print statements
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


if __name__ == "__main__":
    device = get_device()
    batch_size = 512
    num_epochs = 10
    learning_rate = 1e-3
    momentum = 0.99

    # Load data
    train_loader = load_data(device, batch_size=batch_size)

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
    )

    # Optionally, save the final model
    save_model(trained_model, "final")