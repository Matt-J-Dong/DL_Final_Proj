#!/bin/bash
#SBATCH --nodes=1                        # requests 1 compute server
#SBATCH --ntasks-per-node=1              # runs 1 tasks on each server
#SBATCH --cpus-per-task=1                # uses 4 compute cores per task
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=64GB
#SBATCH --job-name=DeepLearning_model_m
#SBATCH --output=DeepLearning_model_m.out
#SBATCH --error=DeepLearning_model_m-error.txt
#SBATCH --mail-user=matthew.dong@nyu.edu
#SBATCH --mail-type=ALL
#SBATCH --account=csci_ga_2572-2024fa
#SBATCH --partition=c12m85-a100-1

singularity exec --nv --overlay /scratch/mjd9571/overlay-50G-10M.ext3:ro /scratch/mjd9571/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif /bin/bash -c "
cd /scratch/mjd9571/DL_Final_Proj
source /ext3/miniconda3/etc/profile.d/conda.sh
conda activate my_env

# Verify environment
pwd
env_list=\$(pip list)
echo "\$env_list"

# Check GPU status
nvidia-smi

# Source .env if it exists
if [ -f .env ]; then
    source .env
fi

# Set and export WANDB API key
# export WANDB_API_KEY=\$WANDB_API_KEY
# wandb login --relogin \$WANDB_API_KEY

# Run your script
python md_train_m.py
"