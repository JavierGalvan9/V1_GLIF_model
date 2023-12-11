#! /bi	/bash

# total_neurons = 296991

# run -c 1 -m 24 -t 0:30 -o Out/tf212.out -e Error/tf212.err -j tf212 "python multi_training.py --batch_size 1 --neurons 5000 --seq_len 600 --n_epochs 1"
run -g 1 -m 24 -t 0:30 -o Out/training_core.out -e Error/training_core.err -j drif_train "python multi_training.py --batch_size 1 --neurons 75000 --seq_len 600 --n_epochs 3"
# run -g 1 -m 24 -t 2:30 -o Out/gpu_1000_$orientation.out -e Error/gpu_1000_$orientation.err -j drif_train "python drifting_gratings_training.py --batch_size 1 --neurons 1000 --seq_len 3000 --gratings_orientation $orientation --gratings_frequency $frequency --n_simulations 1"
# run -g 1 -m 24 -t 2:30 -o Out/gpu_full_$orientation.out -e Error/gpu_full_$orientation.err -j drif_train "python drifting_gratings_training.py --batch_size 1 --neurons 296991 --seq_len 3000 --gratings_orientation $orientation --gratings_frequency $frequency --n_simulations 1"
# run -g 1 -m 24 -t 2:30 -o Out/training.out -e Error/training.err -j drif_train "kernprof -l -v multi_training.py --batch_size 1 --neurons 50000 --seq_len 600 --n_epochs 1"
# run -g 1 -m 24 -t 2:30 -o Out/training_core.out -e Error/training_core.err -j drif_train "scalene --reduced-profile multi_training.py --batch_size 1 --neurons 1000 --seq_len 600 --n_epochs 1"
# run -g 1 -m 48 -t 0:30 -o Out/scalene_gpu.out -e Error/scalene_gpu.err -j scalene "scalene --outfile v1_scalene.html --html multi_training.py --batch_size 1 --neurons 5000 --seq_len 600 --n_epochs 1"

