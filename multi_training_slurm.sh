#!/bin/bash
#SBATCH -N1 -c1 -n8
#SBATCH --gpus-per-node=a100:4
###SBATCH --gpus-per-node=1
#SBATCH --partition=braintv
#SBATCH --mem=100G
#SBATCH -t10:00:00
#SBATCH --qos=braintv
###SBATCH --output=gpu_run.out
###SBATCH --error=gpu_run.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=shinya.ito@alleninstitute.org


module load cuda/11.1
# python custom_run_script.py
python -u multi_training.py --steps_per_epoch 100 --seq_len 600 --neurons 5000