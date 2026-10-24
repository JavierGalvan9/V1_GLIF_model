import subprocess
import json
import os
import re
import argparse
import shlex
from v1_model_utils import toolkit
from v1_model_utils import spatial_layout
from v1_model_utils.offload_options import HISTORY_OFFLOAD_MODES, VOLTAGE_OFFLOAD_MODES
from v1_model_utils import training_utils
# # script_path = "bash d

# GPU classes each cluster's scheduler accepts; the first one is the default.
CLUSTER_GPU_TYPES = {
    'nuredduna': ('rtxpro6000', 'rtx3090', 'L40S'),
    'talaia': ('b200', 'h200', 'l40s'),
}
# TalaIA node sizes cap the GPUs one training job can request.
TALAIA_GPUS_PER_NODE = {'b200': 8, 'h200': 2, 'l40s': 8}
THREAD_VARIABLES = (
    'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
    'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS',
)

# Create argument parser
parser = argparse.ArgumentParser()

# Define arguments
parser.add_argument('--task_name', default='drifting_gratings_firing_rates_distr', type=str)
parser.add_argument('--data_dir', default='GLIF_network', type=str)
parser.add_argument('--results_dir', default='Simulation_results', type=str)
parser.add_argument('--restore_from', default='', type=str)
parser.add_argument('--comment', default='', type=str)
parser.add_argument('--delays', default='0,0', type=str)
parser.add_argument('--scale', default='2,2', type=str)
parser.add_argument('--dtype', default='float32', type=str, choices=['float16', 'float32'])
parser.add_argument(
    '--neuron_layout', default='morton', type=str,
    choices=list(spatial_layout.LAYOUTS),
    help='Runtime neuron numbering passed to both scripts.',
)
parser.add_argument(
    '--lgn_row_order', default='original', type=str,
    choices=list(spatial_layout.LGN_ROW_ORDERS),
    help='Runtime LGN row numbering passed to both scripts.',
)

parser.add_argument('--optimizer', default='exp_adam', type=str, choices=['adam', 'sgd', 'exp_adam'])
parser.add_argument('--learning_rate', default=0.005, type=float)
parser.add_argument('--lr_schedule', default='none', type=str, choices=['none', 'warmup_cosine'])
parser.add_argument('--lr_warmup_start_lr', default=0.08, type=float)
parser.add_argument('--lr_warmup_target_lr', default=0.04, type=float)
parser.add_argument('--lr_warmup_steps', default=120, type=int)
parser.add_argument('--lr_cosine_min_lr', default=0.001, type=float)
parser.add_argument('--lr_cosine_steps', default=880, type=int)
parser.add_argument('--rate_cost', default=10000., type=float)
parser.add_argument('--voltage_cost', default=1., type=float)
parser.add_argument('--sync_cost', default=1.5, type=float)
parser.add_argument('--osi_cost', default=20., type=float)
parser.add_argument('--annulus_loss_weight', default=0.1, type=float)
parser.add_argument('--osi_loss_subtraction_ratio', default=0., type=float)
parser.add_argument(
    '--osi_loss_method',
    default='crowd_osi',
    type=str,
    choices=[
        'crowd_osi',
        'adaptative_crowd_osi',
        'rolling_osi_emd',
        'batch_osi_emd',
        'crowd_spikes',
        'neuropixels_fr',
    ],
)
parser.add_argument(
    '--rolling_decay',
    default=-1.0,
    type=float,
    help='EMA decay for rolling_osi_emd. Set < 0 to auto-compute from batch_size and rolling_target_sample_ess.',
)
parser.add_argument(
    '--rolling_target_sample_ess',
    default=80.0,
    type=float,
    help='Target effective sample size in samples used when rolling_decay < 0 (auto mode).',
)
parser.add_argument(
    '--rolling_gradient_correction',
    dest='rolling_gradient_correction',
    action='store_true',
    help='Scale current-batch gradients through the rolling OSI/DSI EMA without changing forward values.',
)
parser.add_argument(
    '--norolling_gradient_correction',
    dest='rolling_gradient_correction',
    action='store_false',
)
parser.set_defaults(rolling_gradient_correction=False)
parser.add_argument(
    '--rolling_max_gradient_scale',
    default=20.0,
    type=float,
    help='Maximum gradient scale used by rolling_gradient_correction.',
)
parser.add_argument(
    '--rolling_warmup',
    dest='rolling_warmup',
    action='store_true',
    help='Ramp rolling OSI/DSI loss by current EMA effective sample size during cold start.',
)
parser.add_argument(
    '--norolling_warmup',
    dest='rolling_warmup',
    action='store_false',
)
parser.set_defaults(rolling_warmup=True)
parser.add_argument('--batch_emd_alignment_weight', default=1.0, type=float)

parser.add_argument('--dampening_factor', default=0.05, type=float)
parser.add_argument('--recurrent_dampening_factor', default=1.0, type=float)
parser.add_argument('--global_clipnorm', default=0.0, type=float, help='Clip the global norm of the summed (all-replica), unscaled gradient; <= 0 disables.')
parser.add_argument('--detach_reset', dest='detach_reset', action='store_true')
parser.add_argument('--nodetach_reset', dest='detach_reset', action='store_false')
parser.set_defaults(detach_reset=True)
parser.add_argument('--integration_scheme', type=str, default='exact', choices=['exact', 'euler'])
parser.add_argument('--detach_asc_reset', dest='detach_asc_reset', action='store_true')
parser.add_argument('--nodetach_asc_reset', dest='detach_asc_reset', action='store_false')
parser.set_defaults(detach_asc_reset=False)
parser.add_argument('--input_weight_scale', default=1.0, type=float)
parser.add_argument('--gauss_std', default=0.28, type=float)
parser.add_argument('--recurrent_weight_regularization', default=0.0, type=float)
parser.add_argument('--recurrent_weight_regularizer_type', default="emd", type=str, choices=['mean', 'emd'])
parser.add_argument('--voltage_penalty_mode', default='range', type=str, choices=['range', 'threshold'])
parser.add_argument('--lr_scale', default=1.0, type=float)
# parser.add_argument('--input_f0', default=0.2, type=float)
parser.add_argument('--temporal_f', default=2.0, type=float)
parser.add_argument('--max_time', default=-1, type=float)
parser.add_argument('--loss_core_radius', default=400.0, type=float)
parser.add_argument('--plot_core_radius', default=400.0, type=float)

parser.add_argument('--n_gpus', default=1, type=int)
parser.add_argument(
    '--cluster', default='nuredduna', choices=list(CLUSTER_GPU_TYPES),
    help='Cluster whose scheduler receives the jobs.',
)
parser.add_argument(
    '--gpu_type', default=None,
    help='Scheduler GPU class for training jobs (default: the first one listed for --cluster).',
)
parser.add_argument('--qos', default='normal', help='Slurm QoS for TalaIA jobs.')
parser.add_argument('--walltime', default='48:00', help='Scheduler wall time as hours:minutes.')
parser.add_argument('--memory_gb', default=80, type=int, help='Host memory (GB) requested for training jobs.')
parser.add_argument('--n_runs', default=1, type=int) # number of runs with n_epochs each, with an osi/dsi evaluation after each
parser.add_argument('--n_epochs', default=75, type=int)
parser.add_argument('--batch_size', default=2, type=int)
parser.add_argument('--grating_batch_size', default=1, type=int)
parser.add_argument('--gray_batch_size', default=1, type=int)
parser.add_argument('--neurons', default=0, type=int)
parser.add_argument('--steps_per_epoch', default=25, type=int)
parser.add_argument('--val_steps', default=1, type=int)

parser.add_argument(
    '--n_input', default=17400, type=training_utils.validate_n_input,
    help='Number of LGN inputs (1..17400).',
)
parser.add_argument('--seq_len', default=500, type=int)
parser.add_argument('--n_trials_per_angle', default=10, type=int)
# parser.add_argument('--n_cues', default=3, type=int)
# parser.add_argument('--recall_duration', default=40, type=int)
parser.add_argument('--cue_duration', default=40, type=int)
# parser.add_argument('--interval_duration', default=40, type=int)
# parser.add_argument('--examples_in_epoch', default=32, type=int)
# parser.add_argument('--validation_examples', default=16, type=int)
parser.add_argument('--seed', default=3000, type=int)
parser.add_argument('--neurons_per_output', default=16, type=int)
parser.add_argument('--fano_samples', default=512, type=int)

# parser.add_argument('--float16', default=False, action='store_true')
parser.add_argument('--caching', default=True, action='store_true')
parser.add_argument('--core_only', default=False, action='store_true')
parser.add_argument('--core_loss', default=False, action='store_true')
parser.add_argument('--hard_reset', default=False, action='store_true')
parser.add_argument('--low_memory_gpu', default=False, action='store_true')

parser.add_argument('--train_recurrent', dest='train_recurrent', action='store_true')
parser.add_argument('--notrain_recurrent', dest='train_recurrent', action='store_false')
parser.set_defaults(train_recurrent=True)
parser.add_argument('--train_recurrent_per_type', default=False, action='store_true')
parser.add_argument('--train_input', default=False, action='store_true')
parser.add_argument('--train_noise', default=False, action='store_true')
parser.add_argument('--compute_lgn_activity_gradient', default=False, action='store_true')
parser.add_argument('--compute_bkg_activity_gradient', default=False, action='store_true')
parser.add_argument('--sequential_stimuli', default=False, action='store_true')
parser.add_argument('--debug_gradients', dest='debug_gradients', action='store_true')
parser.add_argument('--nodebug_gradients', dest='debug_gradients', action='store_false')
parser.set_defaults(debug_gradients=False)

parser.add_argument('--connected_selection', default=True, action='store_true')
parser.add_argument('--neuron_output', default=False, action='store_true')

# parser.add_argument('--visualize_test', default=False, action='store_true')
parser.add_argument('--pseudo_gauss', default=False, action='store_true')
parser.add_argument(
    '--surrogate_gradient',
    default='gaussian',
    choices=['triangular', 'gaussian', 'slayer'],
)
parser.add_argument('--current_input', default=False, action='store_true')
parser.add_argument('--bmtk_compat_lgn', default=True, action='store_true')
parser.add_argument('--lgn_backend', default='grating', choices=['grating', 'movie', 'tensorflow'])
parser.add_argument('--reset_every_step', default=False, action='store_true')
parser.add_argument('--spontaneous_training', default=False, action='store_true')
parser.add_argument('--random_weights', default=False, action='store_true')
parser.add_argument('--uniform_weights', default=False, action='store_true')
parser.add_argument('--backward_mode', default='recorded', choices=['recorded', 'checkpointing', 'autodiff'],
                    help='Training backward mode (see multi_training.py)')
parser.add_argument(
    '--voltage_offload', nargs='?', const='adaptive', default='none',
    choices=VOLTAGE_OFFLOAD_MODES,
    help='Recorded membrane offload: bare flag enables adaptive; compressed is its legacy alias.',
)
parser.add_argument(
    '--history_offload', nargs='?', const='adaptive', default='none',
    choices=HISTORY_OFFLOAD_MODES,
    help='Recorded packed-history offload: bare flag enables adaptive; raw is its legacy alias.',
)
parser.add_argument('--segmented_rollout', dest='segmented_rollout', action='store_true',
                    help='Evaluation rollout in int32-safe chunks (see osi_dsi_estimator.py)')
parser.add_argument('--nosegmented_rollout', dest='segmented_rollout', action='store_false')
parser.set_defaults(segmented_rollout=True)
parser.add_argument('--pack_spike_checkpoints', dest='pack_spike_checkpoints', action='store_true')
parser.add_argument('--nopack_spike_checkpoints', dest='pack_spike_checkpoints', action='store_false')
parser.set_defaults(pack_spike_checkpoints=True)
parser.add_argument('--rotation', default='ccw', type=str)
parser.add_argument('--print_only', default=False, action='store_true', help='Only print the commands without submitting them')
parser.add_argument('--neuropixels_df', default='Neuropixels_data/OSI_DSI_neuropixels_v4.csv', type=str, help='File name of the Neuropixels DataFrame for OSI/DSI analysis')


def format_command(command):
    return shlex.join([str(part) for part in command])


def scheduler_command(flags, script, job_name, log_name, gpus, cpus, memory_gb, walltime,
                      gpu_type=None, dependency=None):
    """
    Build the submission command of one job for the scheduler of flags.cluster.
    Nuredduna jobs go through its `run` helper; TalaIA jobs call sbatch directly
    with the same resources, inheriting the submitting shell's environment.
    """
    logs = [f"Out/{log_name}.out", f"Error/{log_name}.err"]
    if flags.cluster == 'nuredduna':
        command = ["run", "-g", str(gpus)] + (["-G", gpu_type] if gpu_type else [])
        command += ["-c", str(cpus), "-m", str(memory_gb), "-t", walltime]
        if dependency:
            command += ["-d", dependency]
        return command + ["-o", logs[0], "-e", logs[1], "-j", job_name, script]

    gres = f"gpu:{gpu_type}:{gpus}" if gpu_type else f"gpu:{gpus}"
    threads = ",".join(f"{name}={cpus}" for name in THREAD_VARIABLES)
    command = [
        # TalaIA rejects --partition: it routes jobs by their GPU request.
        "sbatch", "-A", "uib_focosineu_acc", "--qos", flags.qos,
        f"--gres={gres}", "-c", str(cpus), f"--mem={memory_gb}G",
        # Like `run`, the wall time is given as hours:minutes.
        # Unbuffered output keeps the logs live, as `run`'s `srun -u` does.
        "-t", f"{walltime}:00", f"--export=ALL,PYTHONUNBUFFERED=1,{threads}",
    ]
    if dependency:
        command.append(f"--dependency=afterok:{dependency}")
    return command + ["-o", logs[0], "-e", logs[1], "-J", job_name, "--wrap", script]


def submit_job(command, print_only=False):
    """
    Submit a job to the cluster using the command provided.
    """
    if print_only:
        print(format_command(command))
        return None

    result = subprocess.run(command, capture_output=True, text=True)
    job_id = re.search(r'\d+', result.stdout.strip())
    if job_id is None:
        raise RuntimeError(
            "Could not parse job id from run output.\n"
            f"Command: {format_command(command)}\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )
    job_id = job_id.group()

    return job_id

def main():
    # Initialize the flags and customize the simulation main characteristics
    flags = parser.parse_args()
    gpu_types = CLUSTER_GPU_TYPES[flags.cluster]
    flags.gpu_type = flags.gpu_type or gpu_types[0]
    if flags.gpu_type not in gpu_types:
        parser.error(f"--gpu_type {flags.gpu_type} is not available on {flags.cluster}: {gpu_types}")
    if flags.cluster == 'talaia' and flags.n_gpus > TALAIA_GPUS_PER_NODE[flags.gpu_type]:
        parser.error(
            f"--n_gpus {flags.n_gpus} exceeds the {TALAIA_GPUS_PER_NODE[flags.gpu_type]} "
            f"{flags.gpu_type} GPUs of one TalaIA node"
        )

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
    if flags.restore_from == '':
        sim_name = toolkit.get_random_identifier('b_')
        logdir = os.path.join(results_dir, sim_name)
        initial_benchmark_model = ''
    else:
        sim_name = os.path.basename(os.path.dirname(flags.restore_from))
        logdir = os.path.dirname(flags.restore_from)
        initial_benchmark_model = flags.restore_from

    # logdir = '/home/jgalvan/Desktop/Neurocoding/V1_GLIF_model/Simulation_results/v1_65871/b_xwue'
    # sim_name = 'b_xwue'
    print(f'> Results for {flags.task_name} will be stored in:\n {logdir} \n')

    # Define the job resources for the training and evaluation scripts
    cpu_count = 4 if flags.low_memory_gpu else 8 * flags.n_gpus # even when using just 2 cpus the performance with respect to 8 degrades in 2-3% speed
    training_resources = dict(gpus=flags.n_gpus, gpu_type=flags.gpu_type, cpus=cpu_count, memory_gb=flags.memory_gb, walltime=flags.walltime)
    # Evaluation jobs take any available GPU class
    evaluation_resources = dict(gpus=1, cpus=8, memory_gb=200, walltime="3:00")

    # Define the training and evaluation script calls
    # training_script = "python multi_training.py "
    training_script = "python multi_training.py "
    evaluation_script = "python osi_dsi_estimator.py "

    # Append each flag to the string
    for name, value in vars(flags).items():
        if name not in ['seed', 'low_memory_gpu', 'print_only', 'cluster', 'gpu_type', 'qos', 'walltime', 'memory_gb']:
            # multi_training.py does not define the evaluation rollout flag.
            if name == 'segmented_rollout':
                pass
            elif isinstance(value, bool) and not value:
                training_script += f"--no{name} "
            elif isinstance(value, bool) and value:
                training_script += f"--{name} "
            else:
                training_script += f"--{name} {shlex.quote(str(value))} "

            # osi_dsi_estimator.py does not define training-only batch splits,
            # rolling-loss flags, training debug flags, or the backward options.
            if name in {
                'n_gpus', 'grating_batch_size', 'gray_batch_size',
                'debug_gradients', 'global_clipnorm',
                'detach_reset', 'detach_asc_reset',
                'pack_spike_checkpoints', 'backward_mode', 'voltage_offload', 'history_offload',
            } or name.startswith(('rolling_', 'batch_emd_')):
                continue

            if isinstance(value, bool) and not value:
                evaluation_script += f"--no{name} "
            elif isinstance(value, bool) and value:
                evaluation_script += f"--{name} "
            else:
                evaluation_script += f"--{name} {shlex.quote(str(value))} "

    job_ids = []
    eval_job_ids = []

    # initial_evaluation_command = evaluation_commands + ["-o", f"Out/{sim_name}_{v1_neurons}_initial_test.out", "-e", f"Error/{sim_name}_{v1_neurons}_initial_test.err", "-j", f"{sim_name}_initial_test"]
    # if initial_benchmark_model:
    #     initial_evaluation_script = evaluation_script + f"--dtype 'float32' --track_core_only --seq_len 500 --seed {flags.seed} --ckpt_dir {logdir} --run_session {-1} --restore_from {initial_benchmark_model}"
    # else:
    #     initial_evaluation_script = evaluation_script + f"--dtype 'float32' --track_core_only --seq_len 500 --seed {flags.seed} --ckpt_dir {logdir} --run_session {-1} --restore_from '' --evaluate_untrained"
    # eval_job_id = submit_job(initial_evaluation_command + [initial_evaluation_script], print_only=flags.print_only)
    # eval_job_ids.append(eval_job_id)

    for i in range(flags.n_runs):
        # Submit the training and evaluation jobs with dependencies: train0 - train1 & eval0 - rtrain2 & eval1 - ...
        if i == 0:
            dependency = None
            if initial_benchmark_model:
                new_training_script = training_script + f"--seed {flags.seed} --ckpt_dir {logdir} --run_session {i} --restore_from {initial_benchmark_model} "
            else:
                new_training_script = training_script + f"--seed {flags.seed} --ckpt_dir {logdir} --run_session {i}"
        else:
            dependency = job_ids[i-1]
            if flags.print_only:
                dependency = f"<{sim_name}_train_{i-1}_JOBID>"
            new_training_script = training_script + f"--seed {flags.seed} --ckpt_dir {logdir} --run_session {i}"
        new_training_command = scheduler_command(
            flags, new_training_script, f"{sim_name}_train_{i}", f"{sim_name}_{v1_neurons}_train_{i}",
            dependency=dependency, **training_resources,
        )
        job_id = submit_job(new_training_command, print_only=flags.print_only)
        job_ids.append(job_id)

        if flags.n_runs == 1: # the run is a single run, no need to submit evaluation jobs. osi_dsi will be evaluated at the end of training run
            continue
        else:
            dependency = job_id
            if flags.print_only:
                dependency = f"<{sim_name}_train_{i}_JOBID>"
            new_evaluation_script = evaluation_script + f"--dtype 'float32' --track_core_only --seq_len 200 --seed {flags.seed} --ckpt_dir {logdir} --restore_from 'Intermediate_checkpoints' --run_session {i}"
            new_evaluation_command = scheduler_command(
                flags, new_evaluation_script, f"{sim_name}_test_{i}", f"{sim_name}_{v1_neurons}_test_{i}",
                dependency=dependency, **evaluation_resources,
            )
            eval_job_id = submit_job(new_evaluation_command, print_only=flags.print_only)
            eval_job_ids.append(eval_job_id)

    # # Final evaluation with the best model
    # final_evaluation_command = evaluation_commands + ['-d', job_id, "-o", f"Out/{sim_name}_{v1_neurons}_test_final.out", "-e", f"Error/{sim_name}_{v1_neurons}_test_final.err", "-j", f"{sim_name}_test_final"]
    # final_evaluation_script = evaluation_script + f"--dtype 'float32' --track_core_only --seq_len 200 --seed {flags.seed} --ckpt_dir {logdir} --restore_from 'Best_model' --run_session {i}"
    # final_evaluation_command = final_evaluation_command + [final_evaluation_script]
    # eval_job_id = submit_job(final_evaluation_command)
    # eval_job_ids.append(eval_job_id)

    if flags.print_only:
        print("Print-only mode: no jobs submitted.")
    else:
        print("Submitted training jobs with the following JOBIDs:", job_ids)
        print("Submitted evaluation jobs with the following JOBIDs:", eval_job_ids)


if __name__ == '__main__':
    main()
