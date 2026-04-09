# V1_GLIF_model

A TensorFlow implementation of a biologically realistic mouse primary visual cortex (V1) model with Lateral Geniculate Nucleus (LGN) input, based on Allen Institute models. This project simulates V1 neurons' responses to visual stimuli using Generalized Leaky Integrate-and-Fire (GLIF; specifically the GLIF_3 variant) neurons.

![Fig1](https://github.com/user-attachments/assets/167be7dd-4723-48db-9166-4e5b38df8c23)

The software in this repository requires SONATA format network files. These can be either
generated or downloaded. Please refer to the main repository of the project, [biorealistic-v1-model](https://github.com/AllenInstitute/biorealistic-v1-model), for network-building instructions and download links.

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
- `stim_dataset.py`: Functions for generating visual stimuli (drifting gratings, etc.)
- `drifting_gratings.py`: Script to run the model with drifting grating stimuli
- `multi_training.py`: Distributed-GPU training implementation
- `osi_dsi_estimator.py`: Tool for measuring orientation and direction selectivity
- `METHODS_V1_MODEL_AND_TRAINING.md`: Paper-oriented inventory of model/training methodologies used in code
- `Neuropixels_data/`: Reference data from Neuropixels recordings for comparison

## Getting Started

### Prerequisites

- Python 3.11+
- CUDA 11.8 and cuDNN 8.8.0 (for GPU acceleration)
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
conda activate tf215
```

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
    --batch_size 1 --optimizer exp_adam --learning_rate 0.005 \
    --rate_cost 10000 --voltage_cost 1 --sync_cost 1.5 --osi_cost 20 \
    --train_recurrent
```

Parameters:
- `--rate_cost`: Cost weight for firing-rate regularization
- `--voltage_cost`: Cost weight for voltage dynamics
- `--sync_cost`: Cost weight for synchronization regularization
- `--osi_cost`: Cost weight for matching orientation/direction selectivity to experimental data
- `--train_recurrent`: Enable training of recurrent connections
- `--optimizer`: Optimizer used for the paper-aligned runs (`exp_adam`)
- `--learning_rate`: Learning rate used in the reported configuration (`0.005`)

#### Evaluating Orientation and Direction Selectivity

To evaluate OSI/DSI metrics after training, or to rerun a checkpointed model:

```bash
python osi_dsi_estimator.py --data_dir GLIF_network \
    --neurons 0 --n_trials_per_angle 10 --restore_from "/path/to/model/checkpoint"
```

## Visual Stimuli

The model supports various visual stimuli for testing V1 responses:

- **Drifting gratings**: Sinusoidal gratings moving in different directions
  - Control parameters: orientation, spatial frequency, temporal frequency, contrast
- **Static gratings**: Non-moving sinusoidal gratings
<!-- - **Natural images**: Support for natural image processing (partial implementation) -->

## Data Analysis

The model provides several analysis tools:

- `v1_model_utils/model_metrics_analysis.py`: Calculate metrics like OSI/DSI
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
- TensorFlow 2.15.0
- NumPy 1.23.5
- BMTK 1.0.8+ (Brain Modeling Toolkit)
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
