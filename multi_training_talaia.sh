#!/usr/bin/env bash
# Training launcher for the TalaIA cluster (BSAI). Run it on the TalaIA login
# node from the repository root:
#   bash multi_training_talaia.sh
# parallel_training_testing.py --cluster talaia submits sbatch jobs that inherit
# this shell's environment, so the conda env is activated here.
# No `set -u`: conda's compiler activation scripts read unset variables.
set -eo pipefail

cd "$(dirname "$0")"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate neuro_tf221
mkdir -p Out Error

# GPU classes: b200 (8/node, default), h200 (2/node), l40s (8/node).
gpu_type='b200'
num_replicas=1
# TalaIA enforces --mem: 80 GB was OOM-killed while setting up the 200k model.
# A B200 node has ~2 TB for 8 GPUs, so 256 GB is one GPU's share.
memory_gb=256
# History offload 'raw' runs 13.9 s/step here at batch 256 (packed histories
# cross HostMemory kernels without the uncommitted recorded_adjoint.py fix of
# Experiments/history_offload_20261023); resident histories take 7.6 s and fit
# a B200 easily (34 GiB TF peak). Without any offload: 7.5 s, 125 GiB.
history_offload='none'

################################## 200k network ##################################

for osi_cost in 2; do # default is 2
    python parallel_training_testing.py --cluster talaia --gpu_type $gpu_type --qos 'normal' --walltime '48:00' --memory_gb $memory_gb --osi_loss_method 'crowd_osi' --voltage_penalty_mode 'range' --neuropixels_df 'Neuropixels_data/OSI_DSI_neuropixels_v4.csv' --data_dir 'GLIF_network_nll_full' --seq_len 500 --optimizer 'exp_adam' --learning_rate 0.01 --lr_schedule 'warmup_cosine' --backward_mode 'recorded' --voltage_offload 'compressed' --history_offload $history_offload --lr_warmup_start_lr 0.003 --lr_warmup_target_lr 0.025 --lr_warmup_steps 50 --lr_cosine_min_lr 0.001 --lr_cosine_steps 950 --n_gpus $num_replicas --batch_size 256 --grating_batch_size 128 --gray_batch_size 128 --dtype 'float16' --neurons 203816 --loss_core_radius 400 --plot_core_radius 400 --train_recurrent --train_noise --osi_cost $osi_cost --rate_cost 10000 --voltage_cost 1.5 --sync_cost 1 --recurrent_weight_regularization 1 --n_runs 1 --n_epochs 10 --steps_per_epoch 100
done
