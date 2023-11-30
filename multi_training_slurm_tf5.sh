#!/bin/bash
#SBATCH -N1 -c1 -n8
#SBATCH --gpus-per-node=a100:1
###SBATCH --gpus-per-node=1
#SBATCH --partition=braintv
#SBATCH --mem=100G
#SBATCH -t36:00:00
#SBATCH --qos=braintv
###SBATCH --output=gpu_run.out
###SBATCH --error=gpu_run.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=shinya.ito@alleninstitute.org


# module load cuda/11.1
# python custom_run_script.py
XLA_FLAGS=--xla_gpu_cuda_data_dir=/home/shinya.ito/realistic-model/miniconda3/envs/tf5 \
LD_LIBRARY_PATH=/home/shinya.ito/realistic-model/miniconda3/envs/tf5/lib \
python -u multi_training.py --n_epochs 20 --steps_per_epoch 100 --val_steps 2 --neurons 5000 --noaverage_grad_for_cell_type

