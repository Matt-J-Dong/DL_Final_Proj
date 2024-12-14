import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CyclicLR
from tqdm.auto import tqdm
import os

from dataset import create_wall_dataloader
from models_md_d import JEPA_Model

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

def main():
    device = get_device()

    # Hyperparameters
    batch_size = 64
    num_epochs = 10
    learning_rate = 1e-3
    momentum = 0.99
    dropout = 0.1
    lambda_cov = 0.1
    patience = 3

    data_path = "/scratch/DL24FA"
    train_loader, _ = create_wall_dataloader(
        data_path=f"{data_path}/train",
        probing=False,
        device=device,
        train=True,
        batch_size=batch_size,
    ), None

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

    val_ds = {"normal": probe_val_normal_ds, "wall": probe_val_wall_ds}

    model = JEPA_Model(device=device, repr_dim=256, action_dim=2, dropout=dropout)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    steps_per_epoch = len(train_loader)
    scheduler = CyclicLR(
        optimizer,
        base_lr=learning_rate/10,
        max_lr=learning_rate*0.5,
        step_size_up=2 * steps_per_epoch,
        mode='triangular2'
    )

    best_val_loss_normal = None
    worse_count = 0
    last_pred_encs = None

    for epoch in range(1, num_epochs+1):
        model.train()
        epoch_loss = 0.0
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            states = batch.states
            actions = batch.actions

            loss_value, pred_encs = model.train_step(
                states=states,
                actions=actions,
                optimizer=optimizer,
                momentum=momentum,
                distance_function="l2",
                add_noise=True,
                lambda_cov=lambda_cov
            )
            epoch_loss += loss_value
            optimizer.step()
            scheduler.step()
            last_pred_encs = pred_encs

            if batch_idx % 50 == 0:
                print(f"Epoch [{epoch}/{num_epochs}], Batch {batch_idx}, Loss: {loss_value:.4f}")
                if not os.path.exists("checkpoints"):
                    os.makedirs("checkpoints")
                torch.save({
                    'epoch': epoch,
                    'batch_idx': batch_idx,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, f"checkpoints/jepa_model_d_epoch_{epoch}_batch_{batch_idx}.pth")

        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch} Average Loss: {avg_epoch_loss:.4f}")

        # Check representation collapse
        if last_pred_encs is not None:
            avg_std = last_pred_encs.std(dim=0).mean().item()
            if avg_std < 1e-3:
                print("Warning: Potential representation collapse detected! Avg embedding std very low.")

        # Evaluate after each epoch
        model.eval()
        from evaluator import ProbingEvaluator, ProbingConfig
        evaluator = ProbingEvaluator(
            device=device,
            model=model,
            probe_train_ds=probe_train_ds,
            probe_val_ds=val_ds,
            config=ProbingConfig(),
            quick_debug=False
        )
        prober = evaluator.train_pred_prober()
        avg_losses = evaluator.evaluate_all(prober=prober)
        val_loss_normal = avg_losses["normal"]
        val_loss_wall = avg_losses["wall"]
        print(f"Validation normal loss: {val_loss_normal:.4f}, wall loss: {val_loss_wall:.4f}")

        with open("losses_c.txt", "a") as f:
            f.write(f"Epoch {epoch}: train_loss={avg_epoch_loss}, val_loss_normal={val_loss_normal}, val_loss_wall={val_loss_wall}\n")

        # Early stopping
        if best_val_loss_normal is None:
            best_val_loss_normal = val_loss_normal
            worse_count = 0
        else:
            if val_loss_normal > best_val_loss_normal:
                worse_count += 1
                if worse_count == patience:
                    print("Early stopping triggered.")
                    break
            else:
                best_val_loss_normal = val_loss_normal
                worse_count = 0

        if epoch % 1 == 0:
            if not os.path.exists("checkpoints"):
                os.makedirs("checkpoints")
            torch.save({
                'epoch': epoch,
                'batch_idx': -1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, f"checkpoints/jepa_model_d_epoch_{epoch}_final.pth")

    print("Training completed.")

if __name__ == "__main__":
    main()
