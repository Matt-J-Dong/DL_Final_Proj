Model version 6 validation loss increased 2 epochs in a row. RIP model.
Model version 7 likely also bad.

Model version d is a new architecture. Version d is withou wandb and is basic.
Version e adds wandb.
Version c is the most recent version of ResNet50.
Version f builds on it with new hyperparameters.


c12m85-a100-1
srun --account=csci_ga_2572-2024fa --partition=c12m85-a100-1 --gres=gpu:a100:1 --cpus-per-task=4 --time=08:00:00 --pty /bin/bash\

Version v2 is an LSTM encoder.
Version v3 is a 1/2 size version of v2
Version v4 is some regularization testing.
Version v5 is adding back in validation.

again: normal loss: 259.8260803222656
wall loss: 187.7458038330078

v1: normal loss: 260.63848876953125
wall loss: 186.7103271484375

v2: normal loss: 249.39834594726562
wall loss: 207.0797576904297

v3: normal loss: 253.91708374023438
wall loss: 205.20118713378906

jc_v4:
normal loss: 255.3264617919922
wall loss: 197.2579345703125
normal loss: 259.0669860839844
wall loss: 186.3773651123047
normal loss: 258.6222229003906
wall loss: 187.432373046875