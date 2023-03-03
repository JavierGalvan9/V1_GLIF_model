#! /bi	/bash

for orientation in 0 #45 90 135 180 225 270 315
do for frequency in 2 #0 1 3 5 7 9 4 6 8
do run -c 1 -m 100 -t 10:00 -o Out/5000_help_$orientation.out -e Error/5000_help_$orientation.err -j drif_gratings "python drifting_gratings.py --batch_size 1 --neurons 1574 --seq_len 2500 --gratings_orientation $orientation --gratings_frequency $frequency --n_simulations 1 --noreverse --nooutput_currents"
done
done
