import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
import os
from torch.utils.data import Subset

from dataset import create_wall_dataloader
# Import our new SimpleResNetModel instead of JEPA_Model
from models import SimpleResNetModel

def load_data(device, batch_size=64):
    data_path = "./data/DL24FA"

    train_loader = create_wall_dataloader(
        data_path=f"{data_path}/train",
        probing=False,
        device=device,
        train=True,
        batch_size=batch_size,
    )

    return train_loader, None

def save_model(model, epoch, save_path="checkpoints"):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_file = os.path.join(save_path, f"jepa_model_epoch_{epoch}.pth")
    torch.save(model.state_dict(), save_file)
    print(f"Model saved to {save_file}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    batch_size = 512
    num_epochs = 10
    learning_rate = 1e-3
    save_every=1

    # Load data (not distributed)
    train_loader, train_sampler = load_data(device, batch_size=batch_size)

    # Initialize the SimpleResNetModel
    model = SimpleResNetModel(device=device, repr_dim=256, action_dim=2)
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()

    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            states = batch.states.to(device)  # [B, T, 2, 64, 64]
            actions = batch.actions.to(device)  # [B, T-1, 2]

            # Extract init_state and call model
            init_state = states[:,0]  # [B,2,64,64]
            pred_encs = model(init_state, actions)  # [B,T,D]

            # Dummy target: just zeros
            target_encs = torch.zeros_like(pred_encs)

            loss = criterion(pred_encs, target_encs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            epoch_loss += loss_val

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
    # Save the final model
    save_model(model, "final")


if __name__ == "__main__":
    main()
