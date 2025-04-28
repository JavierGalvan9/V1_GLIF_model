#! /bi	/bash

# total_neurons = 296991
# core_neurons = 65871

# run -c 1 -m 24 -t 0:30 -o Out/tf212.out -e Error/tf212.err -j tf212 "python multi_training.py --batch_size 1 --neurons 5000 --seq_len 600 --n_epochs 1"
# run -g 1 -m 24 -t 0:30 -o Out/training_core.out -e Error/training_core.err -j drif_train "python multi_training.py --batch_size 1 --neurons 40000 --seq_len 600 --n_epochs 3"
# run -g 1 -m 24 -t 2:30 -o Out/gpu_1000_$orientation.out -e Error/gpu_1000_$orientation.err -j drif_train "python drifting_gratings_training.py --batch_size 1 --neurons 1000 --seq_len 3000 --gratings_orientation $orientation --gratings_frequency $frequency --n_simulations 1"
# run -g 1 -m 24 -t 2:30 -o Out/gpu_full_$orientation.out -e Error/gpu_full_$orientation.err -j drif_train "python drifting_gratings_training.py --batch_size 1 --neurons 296991 --seq_len 3000 --gratings_orientation $orientation --gratings_frequency $frequency --n_simulations 1"
# run -g 1 -m 24 -t 2:30 -o Out/training.out -e Error/training.err -j drif_train "kernprof -l -v multi_training.py --batch_size 1 --neurons 50000 --seq_len 600 --n_epochs 1"
# run -g 1 -m 24 -t 2:30 -o Out/training_core.out -e Error/training_core.err -j drif_train "scalene --reduced-profile multi_training.py --batch_size 1 --neurons 1000 --seq_len 600 --n_epochs 1"
# run -g 1 -m 48 -t 0:30 -o Out/scalene_gpu.out -e Error/scalene_gpu.err -j scalene "scalene --outfile v1_scalene.html --html multi_training.py --batch_size 1 --neurons 5000 --seq_len 600 --n_epochs 1"


# run -g 1 -m 24 -t 0:30 -o Out/training_core.out -e Error/training_core.err -j drif_train "python multi_training.py --batch_size 1 --neurons 40000 --seq_len 600 --n_epochs 3"
# run -g 1 -m 60 -t 1:45 -o Out/training_core.out -e Error/training_core.err -j drif_train "python multi_training.py --batch_size 1 --neurons 65871 --train_recurrent --osi_loss_method 'crowd_osi' --osi_cost 2 --rate_cost 100 --voltage_cost 1 --learning_rate 0.1 --seq_len 600 --n_epochs 10 --steps_per_epoch 20"

# # python parallel_training_testing.py --neurons 65871 --seq_len 500 --loss_core_radius 200 --plot_core_radius 200 --delays 50,0 --train_recurrent --train_noise --data_dir 'GLIF_network' --osi_loss_method 'crowd_osi' --osi_cost 20 --rate_cost 10000 --voltage_cost 1 --sync_cost 1 --recurrent_weight_regularization 1 --learning_rate 0.001 --n_runs 2 --n_epochs 50 --steps_per_epoch 25
num_replicas=1
# python parallel_training_testing.py --neuropixels_df 'Neuropixels_data/OSI_DSI_neuropixels_v4.csv' --data_dir 'GLIF_network_800microns' --seq_len 500 --gradient_checkpointing --optimizer 'exp_adam' --learning_rate 0.005 --n_gpus $num_replicas --batch_size 1 --dtype 'float16' --neurons 296991 --loss_core_radius 400 --plot_core_radius 400 --train_recurrent --train_noise --osi_cost 20 --rate_cost 10000 --voltage_cost 1 --sync_cost 1 --recurrent_weight_regularization 1 --n_runs 1 --n_epochs 30 --steps_per_epoch 100
python parallel_training_testing.py --neuropixels_df 'Neuropixels_data/OSI_DSI_neuropixels_v4.csv' --data_dir 'GLIF_network_nll_full' --seq_len 500 --gradient_checkpointing --optimizer 'exp_adam' --learning_rate 0.005 --n_gpus $num_replicas --batch_size 1 --dtype 'float16' --neurons 203816 --loss_core_radius 400 --plot_core_radius 400 --train_recurrent --train_noise --osi_cost 20 --rate_cost 10000 --voltage_cost 1 --sync_cost 1 --recurrent_weight_regularization 1 --n_runs 1 --n_epochs 30 --steps_per_epoch 100
# python parallel_training_testing.py --neuropixels_df 'Neuropixels_data/OSI_DSI_neuropixels_v4.csv' --data_dir 'GLIF_network_nll' --seq_len 500 --gradient_checkpointing --optimizer 'exp_adam' --learning_rate 0.005 --n_gpus $num_replicas --batch_size 1 --dtype 'float16' --neurons 65871 --loss_core_radius 200 --plot_core_radius 200 --train_recurrent --train_noise --osi_cost 20 --rate_cost 10000 --voltage_cost 1 --sync_cost 1 --recurrent_weight_regularization 1 --n_runs 1 --n_epochs 30 --steps_per_epoch 100
# # # python parallel_training_testing.py --seq_len 2000 --nogradient_checkpointing --optimizer 'exp_adam' --learning_rate 0.005 --data_dir 'GLIF_network' --n_gpus $num_replicas --batch_size 1 --dtype 'float16' --neurons 65871 --loss_core_radius 200 --plot_core_radius 200 --train_recurrent --train_noise --osi_cost 20 --rate_cost 10000 --voltage_cost 1 --sync_cost 1 --recurrent_weight_regularization 1 --n_runs 1 --n_epochs 10 --steps_per_epoch 100

# python multi_training_single_gpu_split.py --neurons 50000 --seq_len 500 --loss_core_radius 200 --plot_core_radius 200 --reset_every_step --delays 0,0 --learning_rate 0.001 --rate_cost 10000.0 --voltage_cost 1.0 --osi_cost 10.0 --recurrent_weight_regularization 1 --n_runs 1 --n_epochs 10 --steps_per_epoch 256 --train_recurrent --notrain_input --notrain_noise
# run -g 1 -m 24 -t 0:30 -o Out/tf212.out -e Error/tf212.err -j tf212 "python multi_training_single_gpu_split.py --neurons 65871 --seq_len 500 --loss_core_radius 200 --plot_core_radius 200 --delays 0,0 --learning_rate 0.001 --rate_cost 10000.0 --voltage_cost 1.0 --osi_cost 10.0 --recurrent_weight_regularization 1 --n_runs 1 --n_epochs 2 --steps_per_epoch 1 --train_recurrent --notrain_input --notrain_noise"



# python osi_dsi_estimator.py --random_weights --n_trials_per_angle 10 --dtype 'float16' --neurons 65871 --seq_len 200 --loss_core_radius 200 --plot_core_radius 200 --reset_every_step --delays 0,0 --restore_from 'Best_model' --ckpt_dir '/home/jgalvan/Desktop/Neurocoding/V1_GLIF_model/Simulation_results/v1_65871_random_weights_True/b_py7q' --run_session 1000 --train_noise --notrain_input --train_recurrent	

# python osi_dsi_estimator.py --hard_reset --random_weights --n_trials_per_angle 10 --dtype 'float16' --neurons 65871 --seq_len 200 --loss_core_radius 200 --plot_core_radius 200 --reset_every_step --delays 0,0 --restore_from '/home/jgalvan/Desktop/Neurocoding/V1_GLIF_model/Simulation_results/v1_65871_random_weights_True/b_py7q/Best_model' --ckpt_dir '/home/jgalvan/Desktop/Neurocoding/V1_GLIF_model/Simulation_results/v1_65871_random_weights_True/b_test' --run_session 1000 --train_noise --notrain_input --train_recurrent
# run -g 1 -m 80 -c 4 -t 1:30 -o Out/tf212.out -e Error/tf212.err -j tf212 "python osi_dsi_estimator.py --hard_reset --random_weights --n_trials_per_angle 10 --dtype 'float16' --neurons 65871 --seq_len 200 --loss_core_radius 200 --plot_core_radius 200 --reset_every_step --delays 0,0 --restore_from '/home/jgalvan/Desktop/Neurocoding/V1_GLIF_model/Simulation_results/v1_65871_random_weights_True/b_py7q/Best_model' --ckpt_dir '/home/jgalvan/Desktop/Neurocoding/V1_GLIF_model/Simulation_results/v1_65871_random_weights_True/b_test' --run_session 1000 --train_noise --notrain_input --train_recurrent" 

# export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH


# ### This script is used to run multiple training sessions with different parameters for a neural network model.
# num_replicas=1

# # Run with different batch sizes
# common_params="--neuropixels_df 'Neuropixels_data/OSI_DSI_neuropixels_v4.csv' --gradient_checkpointing --optimizer 'exp_adam' --learning_rate 0.005 --data_dir 'GLIF_network_nll' --n_gpus $num_replicas --seq_len 500 --dtype 'float16' --loss_core_radius 200 --plot_core_radius 200 --train_recurrent --train_noise --osi_cost 20 --rate_cost 10000 --voltage_cost 1 --sync_cost 1 --recurrent_weight_regularization 1 --n_runs 1 --n_epochs 30 --steps_per_epoch 100"

# for batch_size in 1 2 4 6; do
# for neurons in 1000 5000 10000 20000; do
#     echo "Starting training with batch size: $batch_size"
#     python parallel_training_testing.py --batch_size $batch_size --neurons $neurons --low_memory_gpu $common_params
#     echo "Completed training with batch size: $batch_size"
#     echo "-----------------------------------------"
# done
# done

# for batch_size in 1 2 4 6; do
# for neurons in 40000 65871; do
#     echo "Starting training with batch size: $batch_size"
#     python parallel_training_testing.py --batch_size $batch_size --neurons $neurons $common_params
#     echo "Completed training with batch size: $batch_size"
#     echo "-----------------------------------------"
# done
# done

# # Define common parameters
# common_params="--neuropixels_df 'Neuropixels_data/OSI_DSI_neuropixels_v4.csv' --gradient_checkpointing --optimizer 'exp_adam' --learning_rate 0.005 --data_dir 'GLIF_network_nll' --n_gpus $num_replicas --batch_size 1 --dtype 'float16' --loss_core_radius 200 --plot_core_radius 200 --train_recurrent --train_noise --osi_cost 20 --rate_cost 10000 --voltage_cost 1 --sync_cost 1 --recurrent_weight_regularization 1 --n_runs 1 --n_epochs 30 --steps_per_epoch 100"

# # Run with different sequence lengths
# for seq_len in 200 500 1000 2000; do
# for neurons in 1000 5000 10000 20000; do
#     echo "Starting training with sequence length: $seq_len"
#     python parallel_training_testing.py --seq_len $seq_len --neurons $neurons --low_memory_gpu $common_params
#     echo "Completed training with sequence length: $seq_len"
#     echo "-----------------------------------------"
# done
# done

# # Run with different sequence lengths
# for seq_len in 200 500 1000 2000; do
# for neurons in 40000 65871; do
#     echo "Starting training with sequence length: $seq_len"
#     python parallel_training_testing.py --seq_len $seq_len --neurons $neurons $common_params
#     echo "Completed training with sequence length: $seq_len"
#     echo "-----------------------------------------"
# done
# done
