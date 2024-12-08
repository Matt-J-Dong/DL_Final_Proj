import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
import os
from torch.utils.data import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from dataset import WallDataset
from models import JEPA_ViTModel

def get_train_dataset():
    # Load dataset on CPU
    data_path = "./data/DL24FA/train"
    ds = WallDataset(data_path=data_path, probing=False, device='cpu')
    return ds

def save_model(model, epoch, save_path="checkpoints", is_main=True):
    if is_main:  # only the main (rank 0) process saves
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_file = os.path.join(save_path, f"jepa_model_epoch_{epoch}.pth")
        torch.save(model.state_dict(), save_file)
        print(f"Model saved to {save_file}")

def main_worker(local_rank, world_size, batch_size=2048, num_epochs=10, learning_rate=1e-3, save_every=1, momentum=0.99):
    # 1. Initialize distributed process group
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=int(os.environ["RANK"]))

    # 2. Set device for this process
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    rank = dist.get_rank()
    if rank == 0:
        print(f"Process {rank} using device {device}")

    # 3. Load dataset and create DistributedSampler and DataLoader
    train_ds = get_train_dataset()
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=False,
        drop_last=True  # optional: ensures each batch is the same size on all ranks
    )

    # 4. Initialize the JEPA model
    model = JEPA_ViTModel(device=device, repr_dim=256, action_dim=2, img_size=64)
    model.to(device)

    # Wrap model with DDP
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()

    for epoch in range(1, num_epochs + 1):
        train_sampler.set_epoch(epoch)  # shuffle dataset each epoch
        epoch_loss = 0.0

        # Only rank 0 prints progress
        iterator = tqdm(train_loader, desc=f"Epoch {epoch}", disable=(rank!=0))

        for batch_idx, batch in enumerate(iterator):
            states = batch.states.to(device, non_blocking=True)   # [B,T,C,H,W]
            actions = batch.actions.to(device, non_blocking=True) # [B,T-1,2]

            B, T, C, H, W = states.shape
            # Forward pass: predicted embeddings
            pred_encs = model(states, actions)  # [B,T,D]

            # Compute target embeddings via target_encoder (model.module gives original model)
            with torch.no_grad():
                target_list = []
                for t in range(T):
                    s_target = model.module.target_encoder(states[:, t])  # [B,D]
                    target_list.append(s_target)
                target_encs = torch.stack(target_list, dim=1)  # [B,T,D]

            loss = criterion(pred_encs, target_encs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update target encoder
            model.module.update_target_encoder(momentum=momentum)

            loss_val = loss.item()
            epoch_loss += loss_val

            # Save checkpoint at halfway through the epoch
            if rank == 0 and batch_idx == len(train_loader) // 2:
                # Save a checkpoint at the halfway point
                save_model(model.module, f"epoch_{epoch}_half")

            if rank == 0 and batch_idx % 100 == 0:
                print(f"Epoch [{epoch}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss:.4f}")

        # Average epoch loss across all ranks
        loss_tensor = torch.tensor(epoch_loss, device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        avg_epoch_loss = loss_tensor.item() / world_size / len(train_loader)

        if rank == 0:
            print(f"Epoch [{epoch}/{num_epochs}] Average Loss: {avg_epoch_loss:.4f}")

        # Save model at the end of the epoch (only on main rank)
        if rank == 0 and epoch % save_every == 0:
            save_model(model.module, epoch)

    if rank == 0:
        print("Training completed.")
        save_model(model.module, "final")

    dist.barrier()
    dist.destroy_process_group()

def main():
    # Assume the script is launched with torchrun and environment variables are set
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # Hyperparameters can be tuned as needed
    batch_size = 2048
    num_epochs = 10
    learning_rate = 1e-3
    save_every = 1
    momentum = 0.99

    main_worker(local_rank, world_size, batch_size, num_epochs, learning_rate, save_every, momentum)

if __name__ == "__main__":
    main()
