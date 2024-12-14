import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from dataset import create_wall_dataloader
from model import JEPA_Model

def main():
    device = torch.device("cpu")  # CPU-only run

    # Hyperparameters
    input_dim = 128    # Adjust these based on your data
    action_dim = 16     # Adjust if your actions have a different dimension
    repr_dim = 32
    batch_size = 64
    num_epochs = 5
    learning_rate = 1e-3
    momentum = 0.99

    # Load original dataset from npy files
    data_path = "/scratch/DL24FA"
    train_loader = create_wall_dataloader(
        data_path=f"{data_path}/train",
        probing=False,
        device=device,
        train=True,
        batch_size=batch_size,
    )

    model = JEPA_Model(input_dim=input_dim, repr_dim=repr_dim, action_dim=action_dim, device=device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Simple training loop
    for epoch in range(1, num_epochs+1):
        epoch_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            states = batch.states.to(device)   # [B, T, C, H, W] originally, but assume flattened or adjust code
            actions = batch.actions.to(device) # [B, T-1, action_dim], adjust if needed

            # If your original data has shape [B,T,C,H,W], you may need to flatten or average them
            # For simplicity, assume states is already a suitable shape:
            # If not, do something like:
            # states = states.mean(dim=[2,3,4]) # average over C,H,W dims if needed

            # Similarly for actions, if you need a single vector:
            # actions = actions.mean(dim=1)

            # Ensure states and actions match input_dim and action_dim
            # If states originally is [B,T,C,H,W], you need to reduce to [B,input_dim].
            # For testing, let's just flatten:
            B,T,C,H,W = states.shape
            states = states.view(B, -1)  # flatten all except batch
            # If states dimension after flatten is large, adjust input_dim or slicing
            # For testing, let's assert states.size(1) == input_dim:
            # If not correct, you'll need to choose a suitable reshaping.
            assert states.size(1) == input_dim, f"Expected states to have {input_dim} features, got {states.size(1)}"

            # For actions, suppose actions is [B,T-1,action_dim], let's just pick first action:
            # or average over time steps:
            B,Tm1,A = actions.shape
            actions = actions.mean(dim=1)
            assert actions.size(1) == action_dim, f"Expected actions to have {action_dim} features, got {actions.size(1)}"

            loss_value = model.train_step(states, actions, optimizer, momentum=momentum)
            epoch_loss += loss_value

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")

    print("Training completed.")

if __name__ == "__main__":
    main()
