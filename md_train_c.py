import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
import os
import glob
import torch.multiprocessing as mp

from dataset import create_wall_dataloader
from models_md_c import JEPA_Model  # Changed from _a to _c
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

def load_data(device, batch_size=64, is_distributed=False):
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

    # Include probe_lr in filename
    if batch_idx == -1:
        save_file = os.path.join(
            save_path,
            f"jepa_model_c_epoch_{epoch}_final_lr_{learning_rate}_do_{dropout}_cov_{lambda_cov}_probe_{probe_lr}.pth"
        )
    else:
        save_file = os.path.join(
            save_path,
            f"jepa_model_c_epoch_{epoch}_batch_{batch_idx}_lr_{learning_rate}_do_{dropout}_cov_{lambda_cov}_probe_{probe_lr}.pth"
        )

    torch.save({
        'epoch': epoch,
        'batch_idx': batch_idx,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, save_file)
    print(f"Model saved to {save_file}")

def load_latest_checkpoint(model, optimizer, learning_rate, dropout, lambda_cov, probe_lr, checkpoint_dir="checkpoints_wandb"):
    device = get_device()
    if not os.path.exists(checkpoint_dir):
        return 1, 0

    # Include probe_lr in pattern
    pattern = f"jepa_model_c_epoch_*_lr_{learning_rate}_do_{dropout}_cov_{lambda_cov}_probe_{probe_lr}.pth"
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
    dropout=0.1,
    lambda_cov=0.1,
):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer,
        base_lr=learning_rate/10,
        max_lr=learning_rate*0.5,
        step_size_up=2 * steps_per_epoch,
        mode='triangular2'
    )
    
    # Get probe_lr from wandb config, default to 0.0002 if not specified
    probe_lr = wandb.config.get("probe_lr", 0.0002)

    start_epoch, start_batch_idx = load_latest_checkpoint(model, optimizer, learning_rate, dropout, lambda_cov, probe_lr, checkpoint_dir="checkpoints_wandb")
    model.to(device)
    model.train()

    val_ds = {"normal": probe_val_normal_ds, "wall": probe_val_wall_ds}

    best_val_loss_normal = None
    worse_count = 0
    patience = 3

    for epoch in range(start_epoch, num_epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        epoch_loss = 0.0

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            if batch_idx < start_batch_idx and epoch == start_epoch:
                continue

            states = batch.states.to(device)
            actions = batch.actions.to(device)

            loss, _ = model.train_step(
                states=states,
                actions=actions,
                optimizer=optimizer,
                momentum=momentum,
                distance_function=distance_function,
                add_noise=True,
                lambda_cov=lambda_cov
            )
            epoch_loss += loss

            optimizer.step()
            scheduler.step()

            if batch_idx % 50 == 0:
                print(
                    f"Epoch [{epoch}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss:.4f}"
                )
                save_model(model, optimizer, epoch, batch_idx, learning_rate, dropout, lambda_cov, probe_lr)

        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch}/{num_epochs}] Average Loss: {avg_epoch_loss:.4f}")

        # Log to WandB
        wandb.log({
            "epoch": epoch,
            "train_loss": avg_epoch_loss,
            "learning_rate": optimizer.param_groups[0]['lr'],
        })

        model.train()

        if best_val_loss_normal is None:
            best_val_loss_normal = avg_epoch_loss
            worse_count = 0
        else:
            if avg_epoch_loss > best_val_loss_normal:
                worse_count += 1
                if worse_count == patience:
                    print("Early stopping triggered: Validation loss increased 3 epochs in a row.")
                    break
            else:
                best_val_loss_normal = avg_epoch_loss
                worse_count = 0

        start_batch_idx = 0

        if epoch % save_every == 0:
            save_model(model, optimizer, epoch, -1, learning_rate, dropout, lambda_cov, probe_lr)

    print("Training completed.")
    return model

def main():
    device = get_device()

    dropout_values = [0.05, 0.1]
    learning_rates = [1e-3, 5e-4, 1e-4]
    lambda_cov_values = [0.1, 0.5]

    batch_size = 512
    num_epochs = 10
    momentum = 0.99

    train_loader, train_sampler = load_data(device, batch_size=batch_size, is_distributed=False)

    for d in dropout_values:
        for lr in learning_rates:
            for cov in lambda_cov_values:
                wandb.init(project="my_jepa_project", config={
                    "dropout": d,
                    "learning_rate": lr,
                    "batch_size": batch_size,
                    "epochs": num_epochs,
                    "momentum": momentum,
                    "lambda_cov": cov,
                    "probe_lr": wandb.config.get("probe_lr", 0.0002),
                }, reinit=True)

                model = JEPA_Model(device=device, repr_dim=256, action_dim=2, dropout=d)
                model.to(device)

                trained_model = train_model(
                    device=device,
                    model=model,
                    train_loader=train_loader,
                    probe_train_ds=None,
                    probe_val_normal_ds=None,
                    probe_val_wall_ds=None,
                    num_epochs=num_epochs,
                    learning_rate=lr,
                    momentum=momentum,
                    save_every=1,
                    train_sampler=train_sampler,
                    dropout=d,
                    lambda_cov=cov,
                )

                wandb.finish()

if __name__ == "__main__":
    main()
