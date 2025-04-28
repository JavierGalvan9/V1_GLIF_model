import subprocess
import json
import os
import re
import argparse
# import numpy as np
# import tensorflow as tf
# import tensorflow as tf
from v1_model_utils import toolkit
# # script_path = "bash d

# Create argument parser
parser = argparse.ArgumentParser()

# Define arguments
parser.add_argument('--task_name', default='drifting_gratings_firing_rates_distr', type=str)
parser.add_argument('--data_dir', default='GLIF_network', type=str)
parser.add_argument('--results_dir', default='Simulation_results', type=str)
parser.add_argument('--restore_from', default='', type=str)
parser.add_argument('--comment', default='', type=str)
parser.add_argument('--delays', default='100,0', type=str)
parser.add_argument('--scale', default='2,2', type=str)
parser.add_argument('--dtype', default='float32', type=str)

parser.add_argument('--optimizer', default='adam', type=str)
parser.add_argument('--learning_rate', default=0.001, type=float)
parser.add_argument('--rate_cost', default=100., type=float) #100
parser.add_argument('--voltage_cost', default=1., type=float)
parser.add_argument('--sync_cost', default=1., type=float)
parser.add_argument('--osi_cost', default=1., type=float)
parser.add_argument('--osi_loss_subtraction_ratio', default=0., type=float)
parser.add_argument('--osi_loss_method', default='crowd_osi', type=str)

parser.add_argument('--dampening_factor', default=0.1, type=float)
parser.add_argument('--recurrent_dampening_factor', default=0.1, type=float)
# parser.add_argument('--dampening_factor', default=0.5, type=float)
# parser.add_argument('--recurrent_dampening_factor', default=0.5, type=float)
parser.add_argument('--input_weight_scale', default=1.0, type=float)
parser.add_argument('--gauss_std', default=0.3, type=float)
parser.add_argument('--recurrent_weight_regularization', default=0.0, type=float)
parser.add_argument('--recurrent_weight_regularizer_type', default="mean", type=str)
parser.add_argument('--lr_scale', default=1.0, type=float)
# parser.add_argument('--input_f0', default=0.2, type=float)
parser.add_argument('--temporal_f', default=2.0, type=float)
parser.add_argument('--max_time', default=-1, type=float)
parser.add_argument('--loss_core_radius', default=400.0, type=float)
parser.add_argument('--plot_core_radius', default=400.0, type=float)

parser.add_argument('--n_runs', default=1, type=int) # number of runs with n_epochs each, with an osi/dsi evaluation after each
parser.add_argument('--n_epochs', default=50, type=int)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--neurons', default=0, type=int)
parser.add_argument('--steps_per_epoch', default=20, type=int)
parser.add_argument('--val_steps', default=1, type=int)

parser.add_argument('--n_input', default=17400, type=int)
parser.add_argument('--seq_len', default=600, type=int)
# parser.add_argument('--n_cues', default=3, type=int)
# parser.add_argument('--recall_duration', default=40, type=int)
parser.add_argument('--cue_duration', default=40, type=int)
# parser.add_argument('--interval_duration', default=40, type=int)
# parser.add_argument('--examples_in_epoch', default=32, type=int)
# parser.add_argument('--validation_examples', default=16, type=int)
parser.add_argument('--seed', default=3000, type=int)
parser.add_argument('--neurons_per_output', default=16, type=int)
parser.add_argument('--fano_samples', default=500, type=int)

# parser.add_argument('--float16', default=False, action='store_true')
# parser.add_argument('--caching', default=True, action='store_true')
parser.add_argument('--caching', dest='caching', action='store_true')
parser.add_argument('--nocaching', dest='caching', action='store_false')
parser.set_defaults(caching=True)

parser.add_argument('--core_only', default=False, action='store_true')
parser.add_argument('--core_loss', default=False, action='store_true')
parser.add_argument('--hard_reset', default=False, action='store_true')

parser.add_argument('--train_recurrent', default=False, action='store_true')
parser.add_argument('--train_recurrent_per_type', default=False, action='store_true')
parser.add_argument('--train_input', default=False, action='store_true')
parser.add_argument('--train_noise', default=False, action='store_true')

parser.add_argument('--connected_selection', default=True, action='store_true')
parser.add_argument('--neuron_output', default=False, action='store_true')

# parser.add_argument('--visualize_test', default=False, action='store_true')
parser.add_argument('--pseudo_gauss', default=False, action='store_true')
parser.add_argument('--bmtk_compat_lgn', default=True, action='store_true')
parser.add_argument('--reset_every_step', default=False, action='store_true')
parser.add_argument('--spontaneous_training', default=False, action='store_true')
parser.add_argument('--random_weights', default=False, action='store_true')
parser.add_argument('--uniform_weights', default=False, action='store_true')
parser.add_argument('--gradient_checkpointing', default=False, action='store_true')
parser.add_argument('--rotation', default='ccw', type=str)
parser.add_argument('--print_only', default=False, action='store_true', help='Only print the commands without submitting them')
parser.add_argument('--neuropixels_df', default='Neuropixels_data/v1_OSI_DSI_DF.csv', type=str, help='File name of the Neuropixels DataFrame for OSI/DSI analysis')


def submit_job(command, print_only=False):
    """ 
    Submit a job to the cluster using the command provided.
    If print_only is True, just print the command without submitting.
    """
    if print_only:
        print("\n=== COMMAND ===")
        print(" ".join(command))
        # For the wrapped python command, extract it for easier viewing
        if "--wrap" in command:
            wrap_index = command.index("--wrap")
            if wrap_index < len(command) - 1:
                print("\n=== PYTHON COMMAND ===")
                print(command[wrap_index + 1])
        # Return a dummy job ID for print-only mode
        return "DUMMY_JOB_ID"
    
    # Actually submit the job
    result = subprocess.run(command, capture_output=True, text=True)
    job_id = re.search(r'\d+', result.stdout.strip())
    job_id = job_id.group()
    return job_id

def main():                
    # Initialize the flags and customize the simulation main characteristics
    flags = parser.parse_args()
        
    # Get the neurons of each column of the network
    v1_neurons = flags.neurons

    # Save the configuration of the model based on the main features
    flag_str = f'v1_{v1_neurons}'
    for name, value in vars(flags).items():
        if value != parser.get_default(name) and name in ['n_input', 'core_only', 'connected_selection', 'random_weights', 'uniform_weights']:
            flag_str += f'_{name}_{value}'

    # Define flag string as the second part of results_path
    results_dir = f'{flags.results_dir}/{flag_str}'
    os.makedirs(results_dir, exist_ok=True)
    print('Simulation results path: ', results_dir)
    # Save the flags configuration as a dictionary in a JSON file
    with open(os.path.join(results_dir, 'flags_config.json'), 'w') as fp:
        json.dump(vars(flags), fp)

    # Generate a ticker for the current simulation
    sim_name = toolkit.get_random_identifier('b_')
    logdir = os.path.join(results_dir, sim_name)
    print(f'> Results for {flags.task_name} will be stored in:\n {logdir} \n')

    # Define the job submission commands for the training and evaluation scripts
    # training_commands = ["run", "-g", "1", "-G", "L40S", "-m", "48", "-t", "1:30"] # choose the L40S GPU with 48GB of memory 
    # evaluation_commands = ["run", "-g", "1", "-m", "65", "-t", "2:00"]
    training_commands = [
        "sbatch",
        "-N1", "-c1", "-n4",
        "--gpus=a100:1",
        "--partition=d3",
        "--mem=60G",
        "--time=12:00:00",
        "--qos=d3",
    ]
    evaluation_commands = [
        "sbatch",
        "-N1", "-c1", "-n4",
        "--gpus=a100:1",
        "--partition=d3",
        "--mem=200G",
        "--time=4:00:00",
        "--qos=d3",
    ]


    # Define the training and evaluation script calls
    # training_script = "python multi_training.py " 
    training_script = "python -u multi_training_single_gpu_split.py " 
    evaluation_script = "python -u osi_dsi_estimator.py " 

    # initial_benchmark_model = '/home/jgalvan/Desktop/Neurocoding/V1_GLIF_model/Simulation_results/v1_30000/b_3jo7/Best_model'
    initial_benchmark_model = ''

    # Append each flag to the string
    for name, value in vars(flags).items():
        # if value != parser.get_default(name) and name in ['learning_rate', 'rate_cost', 'voltage_cost', 'osi_cost', 'temporal_f', 'n_input', 'seq_len']:
        # if value != parser.get_default(name):
        if name not in ['seed', 'print_only']: 
            if type(value) == bool and value == False:
                training_script += f"--no{name} "
                evaluation_script += f"--no{name} "
            elif type(value) == bool and value == True:
                training_script += f"--{name} "
                evaluation_script += f"--{name} "
            else:
                training_script += f"--{name} {value} "
                evaluation_script += f"--{name} {value} "

    job_ids = []
    eval_job_ids = []

    # # Initial OSI/DSI test
    initial_evaluation_command = evaluation_commands + ["-o", f"Out/{sim_name}_{v1_neurons}_initial_test.out", "-e", f"Error/{sim_name}_{v1_neurons}_initial_test.err", "-J", f"{sim_name}_initial_test"]
    if initial_benchmark_model:
        initial_evaluation_script = evaluation_script + f"--seed {flags.seed} --ckpt_dir {logdir}  --run_session {-1} --restore_from {initial_benchmark_model}"
    else:
        initial_evaluation_script = evaluation_script + f"--seed {flags.seed} --ckpt_dir {logdir}  --run_session {-1}"

    initial_evaluation_command = initial_evaluation_command + ["--wrap"] + [initial_evaluation_script]
    # initial_evaluation_command = [initial_evaluation_script]
    eval_job_id = submit_job(initial_evaluation_command, flags.print_only)
    eval_job_ids.append(eval_job_id)

    for i in range(flags.n_runs):
        # Submit the training and evaluation jobs with dependencies: train0 - train1 & eval0 - rtrain2 & eval1 - ...
        if i == 0:
            new_training_command = training_commands + ["-o", f"Out/{sim_name}_{v1_neurons}_train_{i}.out", "-e", f"Error/{sim_name}_{v1_neurons}_train_{i}.err", "-J", f"{sim_name}_train_{i}"]
            if initial_benchmark_model:
                new_training_script = training_script + f"--seed {flags.seed + i} --ckpt_dir {logdir} --run_session {i} --restore_from {initial_benchmark_model} "
            else:
                new_training_script = training_script + f" --seed {flags.seed + i} --ckpt_dir {logdir} --run_session {i}"
            new_training_command = new_training_command + ["--wrap"] + [new_training_script]
            job_id = submit_job(new_training_command, flags.print_only)
        else:
            new_training_command = training_commands + ['-d', job_ids[i-1], "-o", f"Out/{sim_name}_{v1_neurons}_train_{i}.out", "-e", f"Error/{sim_name}_{v1_neurons}_train_{i}.err", "-J", f"{sim_name}_train_{i}"]
            # new_training_script = training_script + f"--osi_cost 0.1 --rate_cost 10 --seed {flags.seed + i} --ckpt_dir {logdir} --run_session {i}"
            new_training_script = training_script + f"--seed {flags.seed + i} --ckpt_dir {logdir} --run_session {i}"
            new_training_command = new_training_command + ["--wrap"] + [new_training_script]
            job_id = submit_job(new_training_command, flags.print_only)
        job_ids.append(job_id)

        if flags.n_runs == 1: # the run is a single run, no need to submit evaluation jobs. osi_dsi will be evaluated at the end of training run
            continue
        else:
            new_evaluation_command = evaluation_commands + ['-d', job_id, "-o", f"Out/{sim_name}_{v1_neurons}_test_{i}.out", "-e", f"Error/{sim_name}_{v1_neurons}_test_{i}.err", "-J", f"{sim_name}_test_{i}"]
            new_evaluation_script = evaluation_script + f"--seed {flags.seed + i} --ckpt_dir {logdir} --restore_from 'Intermediate_checkpoints' --run_session {i}"
            new_evaluation_command = new_evaluation_command + ["--wrap"] + [new_evaluation_script]
            eval_job_id = submit_job(new_evaluation_command, flags.print_only)
            eval_job_ids.append(eval_job_id)

    # Final evaluation with the best model
    final_evaluation_command = evaluation_commands + ['-d', job_id, "-o", f"Out/{sim_name}_{v1_neurons}_test_final.out", "-e", f"Error/{sim_name}_{v1_neurons}_test_final.err", "-J", f"{sim_name}_test_final"]
    final_evaluation_script = evaluation_script + f"--seed {flags.seed + i} --ckpt_dir {logdir} --restore_from 'Best_model' --run_session {i}"
    final_evaluation_command = final_evaluation_command + ["--wrap"] + [final_evaluation_script]
    eval_job_id = submit_job(final_evaluation_command, flags.print_only)
    eval_job_ids.append(eval_job_id)

    if flags.print_only:
        print("\n=== COMMANDS WERE ONLY PRINTED, NOT SUBMITTED ===\n")
    else:
        print("Submitted training jobs with the following JOBIDs:", job_ids)
        print("Submitted evaluation jobs with the following JOBIDs:", eval_job_ids)


if __name__ == '__main__':
    main()
