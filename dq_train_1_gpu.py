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
from models_dq import JEPA_Model
from evaluator import ProbingEvaluator
import torch.multiprocessing as mp


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
            states = batch.states.to(device)  # [B, T, 2, 64, 64]
            actions = batch.actions.to(device)  # [B, T-1, 2]

            B, T, C, H, W = states.shape

            # 1. Forward pass: predict embeddings
            init_state = states[:, 0]  # [B, C, H, W]
            pred_encs = model.forward(init_state, actions)  # [B, T, D]

            # 2. Compute target embeddings with the target encoder
            target_encs = []
            for t in range(T):
                o_t = states[:, t]  # [B, C, H, W]
                s_target = model.target_encoder(o_t)  # [B, D]
                target_encs.append(s_target)
            target_encs = torch.stack(target_encs, dim=1)  # [B, T, D]

            # 3. Compute loss
            loss = criterion(pred_encs, target_encs)

            # 4. Backpropagation and optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 5. Update target encoder parameters using momentum update
            with torch.no_grad():
                for param_q, param_k in zip(model.encoder.parameters(), model.target_encoder.parameters()):
                    param_k.data = momentum * param_k.data + (1 - momentum) * param_q.data

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
    return model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    batch_size = 512
    num_epochs = 10
    learning_rate = 1e-2
    momentum = 0.99

    # for multiprocessing
    #mp.set_start_method('spawn')

    # Load data (not distributed)
    train_loader, train_sampler = load_data(device, batch_size=batch_size)

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
