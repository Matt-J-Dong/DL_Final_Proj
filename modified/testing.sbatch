#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --mem=64GB
#SBATCH --job-name=autotrain
#SBATCH --output=testing.out
#SBATCH --error=testing-error.txt
#SBATCH --mail-user=matthew.dong@nyu.edu
#SBATCH --mail-type=ALL

singularity exec --nv --overlay $SCRATCH/new_env/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif /bin/bash -c "
source myenv/bin/activate

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
./testing.sh
"