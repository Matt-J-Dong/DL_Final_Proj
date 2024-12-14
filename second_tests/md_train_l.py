# md_train_l.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm.auto import tqdm
import os
import glob
import torch.multiprocessing as mp
import torch.distributed as dist

from dataset import create_wall_dataloader
from models_md_l import JEPA_Model  # Updated import to models_md_l
from evaluator_md import ProbingEvaluator, ProbingConfig
from dotenv import load_dotenv
import wandb

load_dotenv()
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
os.environ["WANDB_API_KEY"] = WANDB_API_KEY
wandb.login(key=WANDB_API_KEY)

def get_device():
    """Set the device for the current process."""
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{dist.get_rank()}')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    return device

def load_data(batch_size=64, is_distributed=False):
    data_path = "/scratch/DL24FA"
    train_loader = create_wall_dataloader(
        data_path=f"{data_path}/train",
        probing=False,
        train=True,
        batch_size=batch_size,
        is_distributed=is_distributed
    )
    return train_loader

def save_model(model, optimizer, epoch, batch_idx, learning_rate, dropout, lambda_cov, probe_lr, save_path="checkpoints_wandb"):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Include probe_lr and version 'l' in filename
    if batch_idx == -1:
        save_file = os.path.join(
            save_path,
            f"jepa_model_l_epoch_{epoch}_final_lr_{learning_rate}_do_{dropout}_cov_{lambda_cov}_probe_{probe_lr}.pth"
        )
    else:
        save_file = os.path.join(
            save_path,
            f"jepa_model_l_epoch_{epoch}_batch_{batch_idx}_lr_{learning_rate}_do_{dropout}_cov_{lambda_cov}_probe_{probe_lr}.pth"
        )

    torch.save({
        'epoch': epoch,
        'batch_idx': batch_idx,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, save_file)
    print(f"Model saved to {save_file}")

def get_probe_lr():
    return wandb.config.get("probe_lr", 0.0002)

def load_latest_checkpoint(model, optimizer, learning_rate, dropout, lambda_cov, probe_lr, checkpoint_dir="checkpoints_wandb"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not os.path.exists(checkpoint_dir):
        return 1, 0

    pattern = f"jepa_model_l_epoch_*_lr_{learning_rate}_do_{dropout}_cov_{lambda_cov}_probe_{probe_lr}.pth"
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, pattern))
    if len(checkpoint_files) == 0:
        return 1, 0

    checkpoint_files.sort(key=os.path.getmtime)
    latest_checkpoint = checkpoint_files[-1]

    print(f"Loading from checkpoint: {latest_checkpoint}")
    checkpoint = torch.load(latest_checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    start_epoch = int(checkpoint['epoch'])
    start_batch_idx = int(checkpoint['batch_idx'])
    if start_batch_idx != -1:
        start_batch_idx += 1
    else:
        start_epoch += 1
        start_batch_idx = 0
    return start_epoch, start_batch_idx

def train_model(
    rank,
    world_size,
    config
):
    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.manual_seed(42)
    
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print(f"Process {rank} using device {device}")

    # Initialize WandB within each process
    wandb.init(
        project="my_jepa_project_sweep_l_energy_reg",  # Updated project name to reflect version 'l'
        config=config,
        sync_tensorboard=True,
        group="ddp_training",
        job_type="train",
        reinit=True
    )

    # Hyperparameters from wandb.config (sweep)
    dropout = wandb.config.get("dropout", 0.0)
    learning_rate = wandb.config.get("learning_rate", 1e-3)
    lambda_cov = wandb.config.get("lambda_cov", 0.1)
    momentum = wandb.config.get("momentum", 0.99)
    batch_size = wandb.config.get("batch_size", 64)
    probe_lr = wandb.config.get("probe_lr", 0.0002)
    num_epochs = wandb.config.get("epochs", 10)
    target_average = wandb.config.get("target_average", 1.0)

    # Create DistributedSampler
    train_loader = create_wall_dataloader(
        data_path="/scratch/DL24FA/train",
        probing=False,
        train=True,
        batch_size=batch_size,
        is_distributed=True,
        rank=rank,
        world_size=world_size
    )

    probe_train_ds = create_wall_dataloader(
        data_path="/scratch/DL24FA/probe_normal/train",
        probing=True,
        train=True,
        batch_size=batch_size,
        is_distributed=False
    )

    probe_val_normal_ds = create_wall_dataloader(
        data_path="/scratch/DL24FA/probe_normal/val",
        probing=True,
        train=False,
        batch_size=batch_size,
        is_distributed=False
    )

    probe_val_wall_ds = create_wall_dataloader(
        data_path="/scratch/DL24FA/probe_wall/val",
        probing=True,
        train=False,
        batch_size=batch_size,
        is_distributed=False
    )

    model = JEPA_Model(device=device, repr_dim=256, action_dim=2, dropout=dropout)
    model.to(device)

    # Wrap the model with DDP
    model = DDP(model, device_ids=[rank] if torch.cuda.is_available() else None)

    distance_function = "l2"

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = CyclicLR(
        optimizer,
        base_lr=learning_rate/10,
        max_lr=learning_rate*0.5,
        step_size_up=2 * len(train_loader),
        mode='triangular2'
    )

    start_epoch, start_batch_idx = load_latest_checkpoint(
        model.module,  # Access the original model inside DDP
        optimizer,
        learning_rate,
        dropout,
        lambda_cov,
        probe_lr,
        checkpoint_dir="checkpoints_wandb"
    )

    model.train()

    val_ds = {"normal": probe_val_normal_ds, "wall": probe_val_wall_ds}

    best_val_loss_normal = None
    worse_count = 0
    patience = 2
    last_pred_encs = None

    for epoch in range(start_epoch, num_epochs + 1):
        # Set epoch for DistributedSampler
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        epoch_energy_loss = 0.0
        epoch_energy_reg = 0.0
        epoch_total_loss = 0.0

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}", position=rank)):
            if batch_idx < start_batch_idx and epoch == start_epoch:
                continue

            states = batch.states.to(device)
            actions = batch.actions.to(device)

            # Generate binary labels: 0 for good inputs, 1 for bad inputs
            # Here, we assume half of the batch is good and half is bad
            # Adjust this as per your data handling strategy
            B = states.size(0)
            labels = torch.cat([
                torch.zeros(B // 2, device=device),
                torch.ones(B - B // 2, device=device)
            ])

            energy_loss, energy_reg, total_loss, pred_encs = model.module.train_step(
                states=states, 
                actions=actions,
                labels=labels,
                optimizer=optimizer,
                momentum=momentum,
                distance_function=distance_function,
                add_noise=True,
                lambda_cov=lambda_cov,
                target_average=target_average
            )
            epoch_energy_loss += energy_loss.item()
            epoch_energy_reg += energy_reg.item()
            epoch_total_loss += total_loss.item()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            last_pred_encs = pred_encs

            if batch_idx % 50 == 0:
                if rank == 0:
                    print(
                        f"Epoch [{epoch}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Energy Loss: {energy_loss:.4f}, Energy Reg Loss: {energy_reg:.4f}, Total Loss: {total_loss:.4f}"
                    )
                    save_model(model.module, optimizer, epoch, batch_idx, learning_rate, dropout, lambda_cov, probe_lr)

        avg_epoch_energy_loss = epoch_energy_loss / len(train_loader)
        avg_epoch_energy_reg = epoch_energy_reg / len(train_loader)
        avg_epoch_total_loss = epoch_total_loss / len(train_loader)
        if rank == 0:
            print(f"Epoch [{epoch}/{num_epochs}] Average Energy Loss: {avg_epoch_energy_loss:.4f}, Average Energy Reg Loss: {avg_epoch_energy_reg:.4f}, Average Total Loss: {avg_epoch_total_loss:.4f}")

        # Check for representation collapse
        if last_pred_encs is not None:
            avg_std = last_pred_encs.std(dim=0).mean().item()
            if avg_std < 1e-3 and rank == 0:
                print("Warning: Potential representation collapse detected! Avg embedding std is very low.")
                wandb.log({"collapse_warning": 1})

        # Evaluate model using ProbingEvaluator
        if rank == 0:
            model.module.eval()
            probing_config = ProbingConfig(
                probe_targets="locations",
                lr=probe_lr,  # Overridden below
                epochs=20,
                schedule=None,
                sample_timesteps=30,
                prober_arch="256",
            )

            evaluator = ProbingEvaluator(
                device=device,
                model=model.module,
                probe_train_ds=probe_train_ds,
                probe_val_ds=val_ds,
                config=probing_config,
                quick_debug=False
            )

            # Override probe_lr from wandb.config
            evaluator.config.lr = probe_lr

            prober = evaluator.train_pred_prober()
            avg_losses = evaluator.evaluate_all(prober=prober)
            val_loss_normal = avg_losses["normal"]
            val_loss_wall = avg_losses["wall"]

            current_probe_lr = probe_lr
            print(f"Validation normal loss: {val_loss_normal:.4f}, Validation wall loss: {val_loss_wall:.4f}, Probing LR: {current_probe_lr}")

            wandb.log({
                "epoch": epoch,
                "train_energy_loss": avg_epoch_energy_loss,
                "train_energy_reg_loss": avg_epoch_energy_reg,
                "train_total_loss": avg_epoch_total_loss,
                "val_loss_normal": val_loss_normal,
                "val_loss_wall": val_loss_wall,
                "learning_rate": optimizer.param_groups[0]['lr'],
                "probing_lr": current_probe_lr,
                "dropout": dropout,
                "lambda_cov": lambda_cov,
                "batch_size": batch_size,
                "momentum": momentum
            })

            with open("losses_l.txt", "a") as f:
                f.write(f"Epoch {epoch}: train_energy_loss={avg_epoch_energy_loss}, train_energy_reg_loss={avg_epoch_energy_reg}, train_total_loss={avg_epoch_total_loss}, val_loss_normal={val_loss_normal}, val_loss_wall={val_loss_wall}, probing_lr={current_probe_lr}\n")

            # Check for early stopping
            if best_val_loss_normal is None:
                best_val_loss_normal = val_loss_normal
                worse_count = 0
            else:
                if val_loss_normal > best_val_loss_normal:
                    worse_count += 1
                    if worse_count == patience:
                        print("Early stopping triggered: Validation loss increased 4 epochs in a row.")
                        break
                else:
                    best_val_loss_normal = val_loss_normal
                    worse_count = 0

            # Final save every epoch
            save_model(model.module, optimizer, epoch, -1, learning_rate, dropout, lambda_cov, probe_lr)

            model.module.train()

        # Synchronize all processes
        dist.barrier()

    # Clean up
    dist.destroy_process_group()

def main():
    sweep_config = {
        "method": "grid",
        "parameters": {
            "momentum": {"values": [0.9, 0.99]},
            "batch_size": {"values": [128, 512]},
            "probe_lr": {"values": [0.0005, 0.002, 0.008]},
            "lambda_cov": {"values": [0.4, 0.7]},
            "learning_rate": {"values": [5e-5, 1e-4, 5e-4]},
            "dropout": {"values": [0.0]},
            "target_average": {"values": [1.0, 1.5, 2.0]}  # Added target_average hyperparameter
        }
    }

    # Initialize WandB sweep (if using sweep)
    # sweep_id = wandb.sweep(sweep=sweep_config, project="my_jepa_project_sweep_l_energy_reg")

    # Number of GPUs
    world_size = 2

    # Launch DDP processes
    mp.spawn(
        train_model,
        args=(world_size, sweep_config),
        nprocs=world_size,
        join=True
    )

    # If not using sweep, you can manually set config and call mp.spawn accordingly

if __name__ == "__main__":
    main()
