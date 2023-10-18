#! /bi	/bash

# for orientation in 45 #45 90 135 180 225 270 315
# do for frequency in 2 #0 1 3 5 7 9 4 6 8
# # do run -c 1 -m 20 -t 2:00 -o Out/core_cpu_full_help_$orientation.out -e Error/core_cpu_full_help_$orientation.err -j drif_gratings "python drifting_gratings.py --batch_size 1 --neurons 66634 --seq_len 3000 --gratings_orientation $orientation --gratings_frequency $frequency --n_simulations 1"
# # do run -c 1 -m 20 -t 2:00 -o Out/1000_cpu_full_help_$orientation.out -e Error/1000_cpu_full_help_$orientation.err -j drif_gratings "python drifting_gratings.py --batch_size 1 --neurons 1000 --seq_len 3000 --gratings_orientation $orientation --gratings_frequency $frequency --n_simulations 1"
# do run -c 1 -m 200 -t 2:00 -o Out/1000_cpu_full_help_$orientation.out -e Error/full_cpu_full_help_$orientation.err -j drif_gratings "python drifting_gratings.py --batch_size 1 --neurons 296991 --seq_len 3000 --gratings_orientation $orientation --gratings_frequency $frequency --n_simulations 1"
# # do run -c 1 -m 100 -t 1:00 -o Out/cpu_full_help_$orientation.out -e Error/cpu_full_help_$orientation.err -j drif_gratings "python drifting_gratings.py --batch_size 1 --neurons 66634 --core_only --seq_len 2500 --gratings_orientation $orientation --gratings_frequency $frequency --n_simulations 1"
# # do run -c 1 -m 100 -t 10:00 -o Out/cpu_full_help_$orientation.out -e Error/cpu_full_help_$orientation.err -j drif_gratings "kernprof -l -v drifting_gratings.py --batch_size 1 --neurons 100000 --seq_len 2500 --gratings_orientation $orientation --gratings_frequency $frequency --n_simulations 1 --noreverse --nooutput_currents --nofloat16"
# done
# done

# total_neurons = 296991

for orientation in 45 #45 90 135 180 225 270 315
do for frequency in 2 #0 1 3 5 7 9 4 6 8
# do run -g 1 -m 24 -t 2:30 -o Out/1full_help_$orientation.out -e Error/1full_help_$orientation.err -j drif_gratings "python drifting_gratings.py --batch_size 1 --neurons 1000 --seq_len 3000 --gratings_orientation $orientation --gratings_frequency $frequency --n_simulations 1"
do run -g 1 -m 24 -t 2:30 -o Out/gpu_core_$orientation.out -e Error/gpu_core_$orientation.err -j drif_gratings "python drifting_gratings.py --batch_size 1 --neurons 66634 --seq_len 3000 --gratings_orientation $orientation --gratings_frequency $frequency --n_simulations 1"
# do run -g 1 -m 200 -t 2:30 -o Out/gpu_core_$orientation.out -e Error/gpu_full_$orientation.err -j drif_gratings "python drifting_gratings.py --batch_size 1 --neurons 296991 --seq_len 3000 --gratings_orientation $orientation --gratings_frequency $frequency --n_simulations 1"
done
done
