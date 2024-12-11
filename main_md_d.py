import torch
import glob
import os
from collections import OrderedDict

from dataset import create_wall_dataloader
from models_md_d import JEPA_Model
from evaluator_lstm import ProbingEvaluator, ProbingConfig

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

def load_data(device):
    data_path = "/scratch/DL24FA"
    probe_train_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_normal/train",
        probing=True,
        device=device,
        train=True
    )

    probe_val_normal_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_normal/val",
        probing=True,
        device=device,
        train=False
    )

    probe_val_wall_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_wall/val",
        probing=True,
        device=device,
        train=False
    )

    probe_val_ds = {"normal": probe_val_normal_ds, "wall": probe_val_wall_ds}
    return probe_train_ds, probe_val_ds

def load_model(device):
    if not os.path.exists("checkpoints"):
        raise FileNotFoundError("No checkpoint directory found.")
    checkpoint_files = glob.glob("checkpoints/jepa_model_c_epoch_*_final.pth")
    if len(checkpoint_files) == 0:
        raise FileNotFoundError("No final checkpoint found.")
    checkpoint_files.sort(key=os.path.getmtime)
    latest_checkpoint = checkpoint_files[-1]
    print("Loading model from", latest_checkpoint)
    checkpoint = torch.load(latest_checkpoint, map_location=device)
    model = JEPA_Model(device=device, repr_dim=256, action_dim=2, dropout=0.1)
    new_state_dict = OrderedDict()
    for k,v in checkpoint['model_state_dict'].items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    return model

def main():
    device = get_device()
    probe_train_ds, probe_val_ds = load_data(device)
    model = load_model(device)
    model.eval()

    evaluator = ProbingEvaluator(
        device=device,
        model=model,
        probe_train_ds=probe_train_ds,
        probe_val_ds=probe_val_ds,
        config=ProbingConfig(),
        quick_debug=False
    )

    prober = evaluator.train_pred_prober()
    avg_losses = evaluator.evaluate_all(prober=prober)
    for prefix, loss in avg_losses.items():
        print(f"{prefix} loss: {loss}")

if __name__ == "__main__":
    main()
