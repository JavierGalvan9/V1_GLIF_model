#! /bi	/bash

# total_neurons = 296991

for orientation in 45 #45 90 135 180 225 270 315
do for frequency in 2 #0 1 3 5 7 9 4 6 8
# do run -g 1 -m 24 -t 2:30 -o Out/gpu_1000_$orientation.out -e Error/gpu_1000_$orientation.err -j drif_train "python drifting_gratings_training.py --batch_size 1 --neurons 1000 --seq_len 3000 --gratings_orientation $orientation --gratings_frequency $frequency --n_simulations 1"
do run -g 1 -m 24 -t 0:30 -o Out/training_core.out -e Error/training_core.err -j drif_train "python multi_training.py --batch_size 1 --neurons 50000 --seq_len 600 --n_epochs 1"
# do run -g 1 -m 24 -t 2:30 -o Out/gpu_full_$orientation.out -e Error/gpu_full_$orientation.err -j drif_train "python drifting_gratings_training.py --batch_size 1 --neurons 296991 --seq_len 3000 --gratings_orientation $orientation --gratings_frequency $frequency --n_simulations 1"
# do run -g 1 -m 24 -t 2:30 -o Out/training.out -e Error/training.err -j drif_train "kernprof -l -v multi_training.py --batch_size 1 --neurons 1000 --seq_len 600 --n_epochs 1"
# do run -g 1 -m 24 -t 2:30 -o Out/training_core.out -e Error/training_core.err -j drif_train "scalene --reduced-profile multi_training.py --batch_size 1 --neurons 1000 --seq_len 600 --n_epochs 1"
# do run -g 1 -m 48 -t 0:30 -o Out/scalene_gpu.out -e Error/scalene_gpu.err -j scalene "scalene --reduced-profile --outfile v1_scalene.html --html multi_training.py --batch_size 1 --neurons 15000 --seq_len 600 --n_epochs 2"
done
done


