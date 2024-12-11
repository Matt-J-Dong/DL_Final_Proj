from dataset import create_wall_dataloader
from evaluator import ProbingEvaluator
import torch
from models_md_9 import JEPA_Model
import glob
import os
from collections import OrderedDict

def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    return device

def load_data(device):
    data_path = "/scratch/DL24FA"

    probe_train_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_normal/train",
        probing=True,
        device=device,
        train=True,
    )

    probe_val_normal_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_normal/val",
        probing=True,
        device=device,
        train=False,
    )

    probe_val_wall_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_wall/val",
        probing=True,
        device=device,
        train=False,
    )

    probe_val_ds = {"normal": probe_val_normal_ds, "wall": probe_val_wall_ds}

    return probe_train_ds, probe_val_ds

def load_model(checkpoint_path="./checkpoints_wandb"):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError("No checkpoint directory found.")
    checkpoint_files = glob.glob(os.path.join(checkpoint_path, "jepa_model_9_epoch_*_batch_*.pth"))
    if len(checkpoint_files) == 0:
        raise FileNotFoundError("No checkpoint found in the directory.")
    checkpoint_files.sort(key=os.path.getmtime)
    latest_checkpoint = checkpoint_files[-1]
    print(f"Loading model from {latest_checkpoint}")
    checkpoint = torch.load(latest_checkpoint, map_location=torch.device(device))
    model = JEPA_Model(device=device, repr_dim=266, action_dim=2)
    new_state_dict = OrderedDict()
    for k, v in checkpoint['model_state_dict'].items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    return model

def evaluate_model(device, model, probe_train_ds, probe_val_ds):
    evaluator = ProbingEvaluator(
        device=device,
        model=model,
        probe_train_ds=probe_train_ds,
        probe_val_ds=probe_val_ds,
        quick_debug=False,
    )

    prober = evaluator.train_pred_prober()
    avg_losses = evaluator.evaluate_all(prober=prober)
    for probe_attr, loss in avg_losses.items():
        print(f"{probe_attr} loss: {loss}")

if __name__ == "__main__":
    device = get_device()
    probe_train_ds, probe_val_ds = load_data(device)
    model = load_model("./checkpoints_wandb")
    evaluate_model(device, model, probe_train_ds, probe_val_ds)
