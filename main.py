# main.py
from dataset import create_wall_dataloader
from evaluator import ProbingEvaluator
import torch
from models import JEPA_ViTModel
import glob
from collections import OrderedDict

def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    return device

def load_data(device):
    data_path = "/scratch/dq2024/DL_Final_Proj/data/DL24FA"

    probe_train_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_normal/train",
        probing=True,
        device=device, # still passed but dataset won't use it for GPU ops
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

def load_model(device):
    # Initialize the JEPA ViT model
    model = JEPA_ViTModel(device=device, repr_dim=256, action_dim=2, img_size=64)
    # Attempt to load saved weights (optional)
    # If no checkpoint or mismatch, skip or use strict=False
    checkpoint_path = './checkpoints/jepa_model_epoch_final.pth'
    if glob.glob(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=False)
        print(f"Loaded model weights from {checkpoint_path}")
    else:
        print("No final checkpoint found, using initialized model.")
    model.eval()
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
    model = load_model(device)
    evaluate_model(device, model, probe_train_ds, probe_val_ds)
