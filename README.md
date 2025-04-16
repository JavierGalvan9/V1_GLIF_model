# V1_GLIF_model

A TensorFlow implementation of a biologically plausible model of mouse primary visual cortex (V1) with Lateral Geniculate Nucleus (LGN) input, based on Allen Institute models. This project simulates V1 neurons' responses to visual stimuli using Generalized Leaky Integrate and Fire (GLIF) neurons.

![Fig1](https://github.com/user-attachments/assets/167be7dd-4723-48db-9166-4e5b38df8c23)

## Model Overview

This model simulates a cortical column in mouse V1, processing visual inputs from LGN and with background activity. Key features include:

- **LGN input processing**: Visual stimuli converted to neural activity patterns in LGN
- **Poisson background activity**: 100 Poisson nodes providing background noise
- **Different cell types**: Excitatory and inhibitory neurons across cortical layers
- **Synaptic double alpha implementation**: Realistic synaptic dynamics modeling
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
- `multi_training_single_gpu_split.py`: Distributed-GPU training implementation
- `osi_dsi_estimator.py`: Tool for measuring orientation and direction selectivity
- `Neuropixels_data/`: Reference data from Neuropixels recordings for comparison

## Getting Started

### Prerequisites

- Python 3.11+
- CUDA 11.8 and cuDNN 8.8.0 (for GPU acceleration)
- LaTeX (optional, for high-quality plot rendering)

### Installation

1. Clone this repository:
```bash
git clone https://github.com/username/V1_GLIF_model.git
cd V1_GLIF_model
```

2. Install dependencies using pip:
```bash
pip install -r requirements.txt
```

3. Install LaTeX (optional, for publication-quality plots):
   - Ubuntu/Debian: `sudo apt-get install texlive-latex-base texlive-fonts-recommended`
   - Fedora/RHEL/CentOS: `sudo dnf install texlive-latex`
   - Arch Linux: `sudo pacman -S texlive-core`

   You can disable LaTeX for plots by setting `plt.rcParams['text.usetex'] = False` in your code.

### Running the Model

#### Testing with Drifting Gratings

To run a simulation with drifting grating stimuli:

```bash
python drifting_gratings.py --neurons 65871 --n_input 17400 --seq_len 2500 \
    --gratings_orientation 0 --gratings_frequency 2 --batch_size 1
```

Parameters:
- `--neurons`: Number of V1 neurons to simulate (e.g., 65871 for full core model)
- `--n_input`: Number of LGN input neurons (default: 17400)
- `--seq_len`: Length of simulation in milliseconds
- `--gratings_orientation`: Orientation angle in degrees
- `--gratings_frequency`: Temporal frequency in Hz
- `--hard_reset`: Use hard reset for neurons (default: True)

#### Training the Model

To train the model to match experimental V1 tuning properties:

```bash
python multi_training_single_gpu_split.py --neurons 65871 --n_input 17400 \
    --seq_len 200 --loss_core_radius 200 --rate_cost 100.0 --voltage_cost 1.0 \
    --osi_cost 1.0 --train_recurrent --train_noise
```

Parameters:
- `--rate_cost`: Cost weight for firing rate regularization
- `--voltage_cost`: Cost weight for voltage dynamics
- `--osi_cost`: Cost weight for matching orientation/direction selectivity to experimental data
- `--train_recurrent`: Enable training of recurrent connections
- `--train_noise`: Enable training of noise input scaling

#### Evaluating Orientation and Direction Selectivity

To evaluate OSI/DSI metrics after training:

```bash
python osi_dsi_estimator.py --neurons 65871 --n_trials_per_angle 10 \
    --restore_from "/path/to/model/checkpoint"
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

## Reference Data

The `Neuropixels_data` directory contains experimental recordings that the model can be trained to match, including:
- Orientation and direction selectivity indices
- Firing rates across cell types

## Dependencies

The model requires specific package versions. Key dependencies include:
- TensorFlow 2.15.0
- NumPy 1.23.5
- BMTK 1.0.8+ (Brain Modeling Toolkit)
- See `requirements.txt` for the complete list

## Citations

Based on the Allen Institute models of mouse V1:
- Billeh et al. (2020), "Systematic Integration of Structural and Functional Data into Multi-Scale Models of Mouse Primary Visual Cortex", Neuron
- Siegle et al. (2021), "Survey of spiking in the mouse visual system reveals functional hierarchy", Nature

## Contributors

- Allen Institute for Brain Science
- Institute for Cross-Disciplinary Physics and Complex Systems (IFISC)

## License

<!-- [Specify the license] -->