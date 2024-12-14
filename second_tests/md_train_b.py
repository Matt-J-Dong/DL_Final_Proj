import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
import os
import glob
import torch.multiprocessing as mp

from dataset import create_wall_dataloader
from models_md_b import JEPA_Model
from evaluator import ProbingEvaluator
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

def save_model(model, optimizer, epoch, batch_idx, learning_rate, dropout, lambda_cov, save_path="checkpoints_wandb"):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if batch_idx == -1:
        save_file = os.path.join(
            save_path,
            f"jepa_model_b_epoch_{epoch}_final_lr_{learning_rate}_do_{dropout}_cov_{lambda_cov}.pth"
        )
    else:
        save_file = os.path.join(
            save_path,
            f"jepa_model_b_epoch_{epoch}_batch_{batch_idx}_lr_{learning_rate}_do_{dropout}_cov_{lambda_cov}.pth"
        )

    torch.save({
        'epoch': epoch,
        'batch_idx': batch_idx,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, save_file)
    print(f"Model saved to {save_file}")

def load_latest_checkpoint(model, optimizer, learning_rate, dropout, lambda_cov, checkpoint_dir="checkpoints_wandb"):
    device = get_device()
    if not os.path.exists(checkpoint_dir):
        return 1, 0  # No checkpoint: start at epoch 1, batch 0

    pattern = f"jepa_model_b_epoch_*_lr_{learning_rate}_do_{dropout}_cov_{lambda_cov}.pth"
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, pattern))
    if len(checkpoint_files) == 0:
        return 1, 0  # No checkpoint

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
    dropout=0.3,     # (Method 1) Increased dropout already set in model.
    lambda_cov=0.1,
):
    # (Method 2) Increase weight decay for stronger regularization
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)

    steps_per_epoch = len(train_loader)
    # (Method 5) Lower max_lr in CyclicLR to 0.5 * learning_rate
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer,
        base_lr=learning_rate/10,
        max_lr=learning_rate*0.5,
        step_size_up=2 * steps_per_epoch,
        mode='triangular2'
    )
    
    start_epoch, start_batch_idx = load_latest_checkpoint(model, optimizer, learning_rate, dropout, lambda_cov, checkpoint_dir="checkpoints_wandb")
    model.to(device)
    model.train()

    val_ds = {"normal": probe_val_normal_ds, "wall": probe_val_wall_ds}

    best_val_loss_normal = None
    worse_count = 0

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
                save_model(model, optimizer, epoch, batch_idx, learning_rate, dropout, lambda_cov)

        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch}/{num_epochs}] Average Loss: {avg_epoch_loss:.4f}")

        # Evaluation
        model.eval()
        evaluator = ProbingEvaluator(
            device=device,
            model=model,
            probe_train_ds=probe_train_ds,
            probe_val_ds=val_ds,
            quick_debug=False
        )

        prober = evaluator.train_pred_prober()
        avg_losses = evaluator.evaluate_all(prober=prober)
        val_loss_normal = avg_losses["normal"]
        val_loss_wall = avg_losses["wall"]
        print(f"Validation normal loss: {val_loss_normal:.4f}")
        print(f"Validation wall loss: {val_loss_wall:.4f}")

        wandb.log({
            "epoch": epoch,
            "train_loss": avg_epoch_loss,
            "val_loss_normal": val_loss_normal,
            "val_loss_wall": val_loss_wall,
            "learning_rate": optimizer.param_groups[0]['lr']
        })

        # Write losses to file
        with open("losses_b.txt", "a") as f:
            f.write(f"Epoch {epoch}: train_loss={avg_epoch_loss}, val_loss_normal={val_loss_normal}, val_loss_wall={val_loss_wall}\n")

        model.train()

        # (Method 3) Stricter early stopping: reduce patience from 4 to 2
        if best_val_loss_normal is None:
            best_val_loss_normal = val_loss_normal
            worse_count = 0
        else:
            if val_loss_normal > best_val_loss_normal:
                worse_count += 1
                if worse_count == 2:  # reduced from 4 to 2
                    print("Early stopping triggered: Validation loss increased 2 epochs in a row.")
                    break
            else:
                best_val_loss_normal = val_loss_normal
                worse_count = 0

        start_batch_idx = 0

        if epoch % save_every == 0:
            save_model(model, optimizer, epoch, -1, learning_rate, dropout, lambda_cov)

    print("Training completed.")
    return model

def main():
    device = get_device()

    # We will try multiple runs with different dropouts, LRs, and lambda_cov
    dropout_values = [0.3]  # Using only the increased dropout value
    learning_rates = [1e-3, 5e-4, 1e-4]
    lambda_cov_values = [0.1,0.5]

    batch_size = 512
    num_epochs = 10
    momentum = 0.99

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

    for d in dropout_values:
        for lr in learning_rates:
            for cov in lambda_cov_values:
                wandb.init(project="my_jepa_project", config={
                    "dropout": d,
                    "learning_rate": lr,
                    "batch_size": batch_size,
                    "epochs": num_epochs,
                    "momentum": momentum,
                    "distance_function": "l2",
                    "lambda_cov": cov
                }, reinit=True)

                model = JEPA_Model(device=device, repr_dim=256, action_dim=2, dropout=d)
                model.to(device)

                trained_model = train_model(
                    device=device,
                    model=model,
                    train_loader=train_loader,
                    probe_train_ds=probe_train_ds,
                    probe_val_normal_ds=probe_val_normal_ds,
                    probe_val_wall_ds=probe_val_wall_ds,
                    num_epochs=num_epochs,
                    learning_rate=lr,
                    momentum=momentum,
                    save_every=1,
                    train_sampler=train_sampler,
                    dropout=d,
                    lambda_cov=cov,
                )

                optimizer = optim.Adam(trained_model.parameters(), lr=lr, weight_decay=1e-3)
                save_model(trained_model, optimizer, num_epochs, -1, lr, d, cov)

                wandb.finish()

if __name__ == "__main__":
    main()
