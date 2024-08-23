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

python parallel_training_testing.py --neurons 65871 --seq_len 500 --loss_core_radius 200 --plot_core_radius 200 --delays 0,0 --reset_every_step --train_recurrent --osi_loss_method 'crowd_osi' --osi_cost 10 --rate_cost 10000 --voltage_cost 1 --recurrent_weight_regularization 1 --learning_rate 0.001 --n_runs 5 --n_epochs 10 --steps_per_epoch 20

# python multi_training_single_gpu_split.py --neurons 50000 --seq_len 500 --loss_core_radius 200 --plot_core_radius 200 --reset_every_step --delays 0,0 --learning_rate 0.001 --rate_cost 10000.0 --voltage_cost 1.0 --osi_cost 10.0 --recurrent_weight_regularization 1 --n_runs 1 --n_epochs 10 --steps_per_epoch 256 --train_recurrent --notrain_input --notrain_noise
# run -g 1 -m 24 -t 0:30 -o Out/tf212.out -e Error/tf212.err -j tf212 "python multi_training_single_gpu_split.py --neurons 65871 --seq_len 500 --loss_core_radius 200 --plot_core_radius 200 --delays 0,0 --learning_rate 0.001 --rate_cost 10000.0 --voltage_cost 1.0 --osi_cost 10.0 --recurrent_weight_regularization 1 --n_runs 1 --n_epochs 2 --steps_per_epoch 1 --train_recurrent --notrain_input --notrain_noise"

# python osi_dsi_estimator.py --n_trials_per_angle 3 --neurons 30000 --seq_len 500 --loss_core_radius 200 --plot_core_radius 200 --reset_every_step --delays 0,0 --ckpt_dir '/home/jgalvan/Desktop/Neurocoding/V1_GLIF_model/Simulation_results/v1_30000/b_f832' --restore_from 'Best_model' --run_session 1000 --notrain_noise --notrain_input --train_recurrent
