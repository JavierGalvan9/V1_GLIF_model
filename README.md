# V1_GLIF_model

A TensorFlow implementation of a biologically realistic mouse primary visual cortex (V1) model with Lateral Geniculate Nucleus (LGN) input, based on Allen Institute models. It simulates V1 neurons' responses to visual stimuli using Generalized Leaky Integrate-and-Fire (GLIF; specifically the GLIF_3 variant) neurons.

This repository requires SONATA-format network files, which can be generated or downloaded. Please refer to the main repository of the project, [biorealistic-v1-model](https://github.com/AllenInstitute/biorealistic-v1-model), for network-building instructions and download links.

## Model Overview

This model simulates a cortical column in mouse V1, processing LGN input together with background activity. Key features include:

- **LGN input processing**: Visual stimuli converted into neural activity patterns in LGN
- **Poisson background activity**: 100 Poisson nodes provide background noise
- **Different cell types**: Excitatory and inhibitory neurons across cortical layers
- **Synaptic double-alpha implementation**: Realistic synaptic dynamics modeling
- **Orientation and direction selectivity**: Analysis tools for measuring tuning properties
- **Neural population analysis**: Tools for visualizing and measuring neural activity

## Project Structure

- `lgn_model/`: LGN model for converting visual stimuli to neural activity
- `v1_model_utils/`: Utility modules for V1 model implementation
  - `load_sparse.py`: Functions for loading network connectivity
  - `models.py`: Core model implementation with GLIF neurons
  - `loss_functions.py`: Custom loss functions for biorealistic training
  - `callbacks.py`: Custom callbacks for TensorFlow training
  - `optimizers.py`: Custom optimizers for training
  - `plotting_utils.py`: Visualization tools
  - `model_metrics_analysis.py`: Analysis functions for model output
  - `other_v1_utils.py`: Various utility functions
- `stim_dataset.py`: Functions for generating visual stimuli, including drifting and static gratings
- `multi_training.py`: Distributed-GPU training implementation
- `osi_dsi_estimator.py`: Tool for evaluating orientation and direction selectivity after training or as a smoke test
- `osi_dsi_analysis.ipynb`: Notebook for exploring OSI/DSI analysis and tuning-angle statistics
- `Neuropixels_data/`: Reference data from Neuropixels recordings for comparison

## Getting Started

### System requirements

- **Operating system:** Linux. macOS and Windows are not supported because this repository requires an NVIDIA GPU with CUDA.
- **Hardware:** NVIDIA GPU with CUDA support is required. Training the core model at batch size 5 needs ~24 GB of VRAM; smaller batch sizes work on smaller GPUs.

### Prerequisites

- Conda and an NVIDIA driver compatible with CUDA 12.9
- The environment supplies Python 3.12, CUDA 12.9, and the matching compiler
  and runtime libraries; no system CUDA toolkit is required
- LaTeX (optional, for high-quality plot rendering)

### Installation

1. Clone this repository:
```bash
git clone https://github.com/JavierGalvan9/V1_GLIF_model.git
cd V1_GLIF_model
```

2. Create the conda environment from `environment.yml`:
```bash
conda env create -f environment.yml
conda activate neuro_tf221
```

`environment.yml` pins all direct dependencies. After a validated environment
update, regenerate the complete platform-specific lock with
`conda env export -n neuro_tf221 | sed '/^prefix: /d' > environment.lock.yml`.

Record the active driver, GPUs, TensorFlow CUDA/cuDNN build, compiler, and core
Python package versions with:

```bash
python runtime_diagnostics.py
```
The lock records all
resolved Conda and pip dependencies for exact Linux reproduction; use the
portable `environment.yml` when resolving for a different platform.

Confirm the active TensorFlow and CUDA toolchain before submitting GPU jobs:

```bash
python -c "import ctypes.util, keras, tensorflow as tf; print('TensorFlow', tf.__version__); print('Keras', keras.__version__); print(tf.sysconfig.get_build_info()); print('cudart', ctypes.util.find_library('cudart')); print(tf.config.list_physical_devices('GPU'))"
nvcc --version
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv
```

TensorFlow's pip CUDA packages provide the runtime libraries while Conda's
`cuda-nvcc` package provides the custom-operation compiler. The host driver
must support the environment's CUDA 12.9 runtime. RTX 3090 (`sm86`), L40S
(`sm89`), and RTX PRO 6000 Blackwell (`sm120`) are the qualification targets.

3. Make the SONATA network available as `GLIF_network/` in the repository root, or point `--data_dir` to the network directory you built or downloaded by following the main repo instructions.
   The network should contain the usual SONATA subdirectories such as `network/` and `components/`.

4. Install LaTeX (optional, for publication-quality plots):
   - Ubuntu/Debian: `sudo apt-get install texlive-latex-base texlive-fonts-recommended`
   - Fedora/RHEL/CentOS: `sudo dnf install texlive-latex`
   - Arch Linux: `sudo pacman -S texlive-core`

   You can disable LaTeX for plots by setting `plt.rcParams['text.usetex'] = False` in your code.

### Running the Model

#### Testing with Drifting Gratings

To run a short smoke test against the full V1 network, use the network you built or downloaded via the main repo instructions and keep the simulated branch duration short:

```bash
python osi_dsi_estimator.py --data_dir GLIF_network \
    --results_dir Simulation_results/smoke_osi --neurons 0 --n_trials_per_angle 1 \
    --seq_len 50 --spont_duration 50 --evoked_duration 50 --n_runs 1 --n_epochs 1 \
    --batch_size 1 --nocalculate_osi_dsi --restore_from ''
```

Parameters:
- `--data_dir`: Path to the SONATA network directory
- `--results_dir`: Output directory for checkpoints and metrics
- `--neurons`: Number of V1 neurons to simulate (`0` means all neurons)
- `--n_trials_per_angle`: Number of trials per orientation
- `--seq_len`: Length of each simulated branch in milliseconds
- `--spont_duration` / `--evoked_duration`: Duration of the spontaneous and evoked windows
- `--nocalculate_osi_dsi`: Skip the final OSI/DSI plotting pass for a smoke test

#### Training the Model

To train the model to match experimental V1 tuning properties:

```bash
python multi_training.py --data_dir GLIF_network \
    --neurons 0 --seq_len 500 --n_epochs 75 --steps_per_epoch 25 \
    --batch_size 5 --optimizer exp_adam --learning_rate 0.005 \
    --rate_cost 10000 --voltage_cost 1 --sync_cost 1.5 --osi_cost 20 \
    --train_recurrent --gradient_checkpointing
```

`multi_training.py` is also the production multi-GPU entry point. It runs one
process holding one replica per visible GPU and all-reduces gradients with
NCCL. Batch flags remain per-replica for backward compatibility:

```bash
CUDA_VISIBLE_DEVICES=0,1 python multi_training.py --n_gpus 2 \
    --batch_size 32 --grating_batch_size 16 --gray_batch_size 16 \
    --data_dir GLIF_network --neurons 0 --seq_len 500 \
    --train_recurrent --train_noise --gradient_checkpointing
```

This example has global batch 64. The equivalent convenience form is
`--n_gpus 2 --global_batch_size 64`; the grating/gray ratio must resolve to
integer per-replica batches. Multi-GPU CUDA training requires GPUs with the
same compute capability. Only the chief replica writes checkpoints, TensorBoard
data, plots, and final artifacts.

**Scale the global batch with the GPU count, not the other way round.** The
recurrent operators are specialized for a static batch of 32 per replica, and
step time is nearly flat in batch size below 16, so two GPUs sharing a *fixed*
global batch of 32 are slower than one GPU doing all of it. On two RTX PRO 6000
GPUs at 32 per replica the 66,652-neuron model reaches 34.3 samples/s against
19.1 on one GPU, with balanced 17.3 GiB per GPU.

`--distributed_mode` selects how `--n_gpus > 1` executes: `mirrored` (default)
uses one process, and `multi_worker` forks one process per GPU with
`MultiWorkerMirroredStrategy`. Multi-worker measured a few percent faster per
step but builds the network once per GPU in host memory; its non-chief workers
write to `Distributed_worker_logs/worker_<n>.log`. See
`distributed_training_investigation/REPORT.md` for the measurements behind both
of these.

The CUDA operators are built automatically when the architecture-specific
cache is missing or stale. Caches for A100 (`sm80`), RTX 3090 (`sm86`), L40S /
RTX 6000 Ada (`sm89`), and RTX Pro 6000 Blackwell (`sm120`) coexist; building
for one architecture does not replace the others.

Note: Running with batch size 5 requires ~40 GB of VRAM. If you have less VRAM, reduce the batch size.

Parameters:
- `--rate_cost`: Cost weight for firing-rate regularization
- `--voltage_cost`: Cost weight for voltage dynamics
- `--sync_cost`: Cost weight for synchronization regularization
- `--osi_cost`: Cost weight for matching orientation/direction selectivity to experimental data
- `--train_recurrent`: Enable training of recurrent connections
- `--optimizer`: Optimizer used for the paper-aligned runs (`exp_adam`)
- `--learning_rate`: Learning rate used in the reported configuration (`0.005`)
- `--gradient_checkpointing`: Enable exact-BPTT activation recomputation. The
  default `segmented` implementation stores recurrent state only at temporal
  chunk boundaries and recomputes the chunks in reverse during backpropagation.
- `--gradient_checkpoint_chunk_size`: Number of timesteps per recomputed chunk
  (default: `25`). Smaller chunks reduce peak VRAM at the cost of additional
  loop overhead.
- `--pack_spike_checkpoints`: Store the binary recurrent spike-delay state at
  temporal checkpoint boundaries in a bit-packed `int32` representation. This
  is opt-in while its end-to-end memory and runtime trade-off is evaluated.
- `--n_gpus`: Number of already-visible GPUs to use (default: `1`).
- `--global_batch_size`: Optional global batch divided evenly across replicas;
  when zero, the existing per-replica batch flags are used.
- `--distributed_mode`: `mirrored` (default, one process) or `multi_worker`
  (one process per GPU) when `--n_gpus > 1`.

For the 203,816-neuron model at sequence length 500 and batch size 1, chunk
sizes 25, 50, and 100 measured approximately 17.9, 26.2, and 43.6 GiB of peak
TensorFlow allocation, respectively, with similar steady-state step times.
Whole-sequence recomputation and chunk size 250 exceeded 96 GiB. These figures
depend on the connectivity, losses, TensorFlow build, and GPU architecture.

#### Evaluating Orientation and Direction Selectivity

To evaluate OSI/DSI metrics after training, or to rerun a checkpointed model:

```bash
python osi_dsi_estimator.py --data_dir GLIF_network \
    --neurons 0 --n_trials_per_angle 10 --restore_from "/path/to/model/checkpoint" \
    --track_core_only
```

`output_spike_dtype` controls only the spike sequence exposed by the recurrent
layer. The OSI/DSI core-neuron path uses `tf.float32` inside the GPU RNN and
converts completed chunks to `uint8`; this avoids per-timestep host transfers.
Other callers can request `tf.uint8`, `tf.float16`, or `tf.float32` explicitly.
The recurrent state remains in the model's configured compute dtype.

Pass `--track_voltage` to save the selected neurons' first-trial membrane
voltage trace to `voltage_trace.npy`. Voltage sequences are not materialized
when this flag is disabled.

`create_model(..., acceleration="auto")` selects the CUDA synaptic-current and
state-transition implementations when the visible GPUs are compatible, and
otherwise uses TensorFlow. Use `acceleration="cuda"` to require CUDA or
`acceleration="tensorflow"` for the reference implementation. The fused state
kernel supports `surrogate_gradient="triangular"`, `"gaussian"`, and
`"slayer"`; the legacy `pseudo_gauss=True` option selects Gaussian.

Compiled CUDA operators are stored outside the repository in an ABI- and
architecture-keyed cache. They are built with `nvcc` and the C++ compiler from
the active Conda environment. Set `V1_CUDA_CACHE_DIR` to override the default
`~/.cache/v1_glif/cuda` location. Missing or stale operators rebuild
automatically; intermediate object files are discarded after linking.

## Visual Stimuli

The model currently supports grating stimuli for testing V1 responses:

- **Drifting gratings**: Sinusoidal gratings moving in different directions
  - Control parameters: orientation, spatial frequency, temporal frequency, and contrast
- **Static gratings**: Non-moving sinusoidal gratings
<!-- - **Natural images**: Support for natural image processing (partial implementation) -->

## Data Analysis

The model provides several analysis tools:

- `v1_model_utils/model_metrics_analysis.py`: Calculate metrics like OSI/DSI
- `osi_dsi_analysis.ipynb`: Notebook for exploring OSI/DSI analysis and tuning-angle statistics
- `v1_model_utils/plotting_utils.py`: Visualization of neural activity
  - `LaminarPlot`: Plot activity across cortical layers
  - `PopulationActivity`: Plot population activity over time
  - `LGN_sample_plot`: Visualize LGN responses to stimuli

## Simulation Results

Results are saved in the `Simulation_results` directory.
 <!-- with the following structure:
- `Images_general/`: Visualization plots
- `Data/`: Raw simulation data
  - Membrane potentials (`v`)
  - Spike activity (`z`)
  - Input currents (`input_current`, `recurrent_current`)
  - LGN activity (`z_lgn`) -->

## Notes
In the new V1 model, the LGN coordinates are defined as the visual field coordinates (elevation and azimuth), though these coordinates are not zero-centered. The LGN’s elevation axis is oriented upward, aligning with the V1 model’s z-axis; however, this may conflict with conventional image coordinate systems, where row indices increase downward. Note that this definition differs from our previous model (Billeh et al., 2020), which defined the LGN coordinate axis as downward. Use the y_dir and flip_y options in BMTK to control image orientation when presenting data to this network.

## Reference Data

The `Neuropixels_data` directory contains experimental recordings that the model can be trained to match, including:
- Orientation and direction selectivity indices
- Firing rates across cell types

## Dependencies

The model requires specific package versions. Key dependencies include:
- Python 3.12
- TensorFlow 2.21.0 and Keras 3
- NumPy 2.5.3
- BMTK 1.2.0 (Brain Modeling Toolkit)
- See `environment.yml` for the complete list

## Citations

Based on the Allen Institute models and experimental data of mouse V1:
- Billeh et al. (2020), "Systematic Integration of Structural and Functional Data into Multi-Scale Models of Mouse Primary Visual Cortex", Neuron
- Siegle et al. (2021), "Survey of spiking in the mouse visual system reveals functional hierarchy", Nature

## Contributors

- Allen Institute for Brain Science
- Institute for Cross-Disciplinary Physics and Complex Systems (IFISC)

## License

<!-- [Specify the license] -->
