from typing import NamedTuple, Optional
import torch
import numpy as np

class WallSample(NamedTuple):
    states: torch.Tensor
    locations: torch.Tensor
    actions: torch.Tensor

class WallDataset:
    def __init__(
        self,
        data_path,
        probing=False,
        device="cpu",  # We'll keep these on CPU by default
    ):
        self.device = device
        # Load the entire arrays into memory (no mmap_mode)
        states_arr = np.load(f"{data_path}/states.npy")  # Fully loaded into CPU memory
        actions_arr = np.load(f"{data_path}/actions.npy")

        # Convert to torch tensors on CPU
        self.states = torch.from_numpy(states_arr).float()  # On CPU
        self.actions = torch.from_numpy(actions_arr).float()

        if probing:
            locations_arr = np.load(f"{data_path}/locations.npy")
            self.locations = torch.from_numpy(locations_arr).float()
        else:
            self.locations = None

    def __len__(self):
        return len(self.states)

    def __getitem__(self, i):
        # Now, we just index the already loaded CPU tensors
        states = self.states[i]
        actions = self.actions[i]

        if self.locations is not None:
            locations = self.locations[i]
        else:
            locations = torch.empty(0)

        # Return the sample (still on CPU)
        return WallSample(states=states, locations=locations, actions=actions)

def create_wall_dataloader(
    data_path,
    probing=False,
    device="cpu",
    batch_size=64,
    train=True,
):
    ds = WallDataset(
        data_path=data_path,
        probing=probing,
        device=device,
    )

    # With everything in memory, you can try num_workers=0 for simplicity.
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=train,
        drop_last=True,
        pin_memory=False,  # No need for pin_memory since data is already in CPU memory
        num_workers=0      # Start with 0 and increase if needed
    )

    return loader
