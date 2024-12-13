import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CyclicLR
from tqdm.auto import tqdm
import os
import glob
import torch.multiprocessing as mp

from dataset import create_wall_dataloader
from models_md_h import JEPA_Model  # Updated import to models_md_h
from evaluator_md import ProbingEvaluator, ProbingConfig
from dotenv import load_dotenv
import wandb

load_dotenv()
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
os.environ["WANDB_API_KEY"] = WANDB_API_KEY
wandb.login(key=WANDB_API_KEY)

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

def save_model(model, optimizer, epoch, batch_idx, learning_rate, dropout, lambda_cov, probe_lr, save_path="checkpoints_wandb"):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Include probe_lr and version 'h' in filename
    if batch_idx == -1:
        save_file = os.path.join(
            save_path,
            f"jepa_model_h_epoch_{epoch}_final_lr_{learning_rate}_do_{dropout}_cov_{lambda_cov}_probe_{probe_lr}.pth"
        )
    else:
        save_file = os.path.join(
            save_path,
            f"jepa_model_h_epoch_{epoch}_batch_{batch_idx}_lr_{learning_rate}_do_{dropout}_cov_{lambda_cov}_probe_{probe_lr}.pth"
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
    device = get_device()
    if not os.path.exists(checkpoint_dir):
        return 1, 0

    pattern = f"jepa_model_h_epoch_*_lr_{learning_rate}_do_{dropout}_cov_{lambda_cov}_probe_{probe_lr}.pth"
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
    device,
    model,
    train_loader,
    probe_train_ds,
    probe_val_normal_ds,
    probe_val_wall_ds,
    num_epochs=1,
    learning_rate=1e-3,
    momentum=0.99,
    save_every=1,
    train_sampler=None,
    distance_function="l2",
    dropout=0.0,
    lambda_cov=0.1,
    target_average=1.0,  # Added target_average parameter
):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    steps_per_epoch = len(train_loader)
    scheduler = CyclicLR(
        optimizer,
        base_lr=learning_rate/10,
        max_lr=learning_rate*0.5,
        step_size_up=2 * steps_per_epoch,
        mode='triangular2'
    )
    
    probe_lr = get_probe_lr()

    start_epoch, start_batch_idx = load_latest_checkpoint(model, optimizer, learning_rate, dropout, lambda_cov, probe_lr, checkpoint_dir="checkpoints_wandb")
    model.to(device)
    model.train()

    val_ds = {"normal": probe_val_normal_ds, "wall": probe_val_wall_ds}

    best_val_loss_normal = None
    worse_count = 0
    patience = 4
    last_pred_encs = None

    for epoch in range(start_epoch, num_epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        epoch_energy_loss = 0.0
        epoch_energy_reg = 0.0
        epoch_total_loss = 0.0

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
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

            energy_loss, energy_reg, total_loss, pred_encs = model.train_step(
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
            epoch_energy_loss += energy_loss
            epoch_energy_reg += energy_reg
            epoch_total_loss += total_loss

            optimizer.step()
            scheduler.step()

            last_pred_encs = pred_encs

            if batch_idx % 50 == 0:
                print(
                    f"Epoch [{epoch}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Energy Loss: {energy_loss:.4f}, Energy Reg Loss: {energy_reg:.4f}, Total Loss: {total_loss:.4f}"
                )
                save_model(model, optimizer, epoch, batch_idx, learning_rate, dropout, lambda_cov, probe_lr)

        avg_epoch_energy_loss = epoch_energy_loss / len(train_loader)
        avg_epoch_energy_reg = epoch_energy_reg / len(train_loader)
        avg_epoch_total_loss = epoch_total_loss / len(train_loader)
        print(f"Epoch [{epoch}/{num_epochs}] Average Energy Loss: {avg_epoch_energy_loss:.4f}, Average Energy Reg Loss: {avg_epoch_energy_reg:.4f}, Average Total Loss: {avg_epoch_total_loss:.4f}")

        # Check for representation collapse
        if last_pred_encs is not None:
            avg_std = last_pred_encs.std(dim=0).mean().item()
            if avg_std < 1e-3:
                print("Warning: Potential representation collapse detected! Avg embedding std is very low.")
                wandb.log({"collapse_warning": 1})

        # Evaluate model using ProbingEvaluator
        model.eval()

        probing_config = ProbingConfig(
            probe_targets="locations",
            lr=0.0002,  # default, overridden below
            epochs=20,
            schedule=None,
            sample_timesteps=30,
            prober_arch="256",
        )

        evaluator = ProbingEvaluator(
            device=device,
            model=model,
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
            "batch_size": wandb.config.get("batch_size"),
            "momentum": momentum
        })

        with open("losses_h.txt", "a") as f:
            f.write(f"Epoch {epoch}: train_energy_loss={avg_epoch_energy_loss}, train_energy_reg_loss={avg_epoch_energy_reg}, train_total_loss={avg_epoch_total_loss}, val_loss_normal={val_loss_normal}, val_loss_wall={val_loss_wall}, probing_lr={current_probe_lr}\n")

        model.train()

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

        start_batch_idx = 0

        if epoch % save_every == 0:
            save_model(model, optimizer, epoch, -1, learning_rate, dropout, lambda_cov, probe_lr)

def main():
    wandb.init(project="YOUR_PROJECT_NAME", config={
        "method": "energy_regularization"
    })  # Replace "YOUR_PROJECT_NAME" with your actual project name

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

    sweep_id = wandb.sweep(sweep_config, project="YOUR_PROJECT_NAME")  # Replace "YOUR_PROJECT_NAME" accordingly
    wandb.agent(sweep_id, function=run_training)

def run_training():
    wandb.init()

    device = get_device()

    # Hyperparameters from wandb.config (sweep)
    dropout = wandb.config.get("dropout", 0.0)
    learning_rate = wandb.config.get("learning_rate", 1e-3)
    lambda_cov = wandb.config.get("lambda_cov", 0.1)
    momentum = wandb.config.get("momentum", 0.99)
    batch_size = wandb.config.get("batch_size", 64)
    probe_lr = wandb.config.get("probe_lr", 0.0002)
    num_epochs = wandb.config.get("epochs", 10)
    target_average = wandb.config.get("target_average", 1.0)

    train_loader, train_sampler = load_data(device, batch_size=batch_size, is_distributed=False)

    data_path = "/scratch/DL24FA"
    probe_train_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_normal/train",
        probing=True,
        device=device,
        train=True,
        batch_size=batch_size
    )

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

    train_model(
        device=device,
        model=model,
        train_loader=train_loader,
        probe_train_ds=probe_train_ds,
        probe_val_normal_ds=probe_val_normal_ds,
        probe_val_wall_ds=probe_val_wall_ds,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        momentum=momentum,
        save_every=1,
        train_sampler=train_sampler,
        distance_function="l2",
        dropout=dropout,
        lambda_cov=lambda_cov,
        target_average=target_average
    )

    # Final save
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    save_model(model, optimizer, num_epochs, -1, learning_rate, dropout, lambda_cov, probe_lr)

    wandb.finish()

if __name__ == "__main__":
    main()
