Model version 6 validation loss increased 2 epochs in a row. RIP model.
Model version 7 likely also bad.

c12m85-a100-1
srun --account=csci_ga_2572-2024fa --partition=c12m85-a100-1 --gres=gpu:a100:1 --cpus-per-task=4 --time=08:00:00 --pty /bin/bash