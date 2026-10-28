# PRG analysis

Record continuous gray-screen activity for future renormalization group analysis.
This folder currently provides recording and data-loading tools; PRG analysis
algorithms are not implemented yet.

## Files

- [long_gray_simulation.py](long_gray_simulation.py): simulate one continuous
  trajectory and stream core-neuron spikes and voltages to HDF5.
- [long_gray_simulation.sh](long_gray_simulation.sh): submit a recording job on
  Nuredduna.
- [analysis.ipynb](analysis.ipynb): load recording windows and reconstruct the
  aligned cell-type vector from saved neuron IDs.
- [checkpoints/](checkpoints/README.md): local reference checkpoints, their
  original training settings, checksums, and validation results.

## Launch a recording

Run the following from the repository root on Nuredduna to restore the trained
203,816-neuron reference. It is TalaIA run `b_u05m`, checkpoint `ckpt-2`, copied
into `PRG/checkpoints/crowd_osi_ckpt/`:

```bash
bash PRG/long_gray_simulation.sh --restore_from PRG/checkpoints/crowd_osi_ckpt
```

See [checkpoints/README.md](checkpoints/README.md) for its training settings,
source, retrieval instructions, and checksums. The files are stored locally and
are excluded from Git because the data shard is approximately 2.95 GB.

Each checkpoint folder contains `checkpoint`, `ckpt-*.index`, and
`ckpt-*.data-*` files. Pass the **folder**, rather than an individual `.index`
or data file, to `--restore_from`. The copied `config.json` records the original
training settings; it is not loaded automatically. This reference was trained
with soft reset, while PRG intentionally uses its learned weights with hard
reset and float32 inference.

For another checkpoint stored in that folder hierarchy:

```bash
bash PRG/long_gray_simulation.sh --restore_from PRG/checkpoints/another_checkpoint
```

Alternatively, specify the checkpoint root and the reference subdirectory:

```bash
bash PRG/long_gray_simulation.sh \
    --ckpt_dir PRG/checkpoints --restore_from crowd_osi_ckpt
```

The launcher activates `neuro_tf221` and submits through `run`, requesting one
RTX 3090 GPU, eight CPUs, 80 GB RAM, and one hour. Scheduler logs are written to
`PRG/Out/` and `PRG/Error/`. Recording paths use
`PRG/results/prg_YYYYMMDD_HHMMSS/simulation_data.hdf5`, with the launch timestamp
in UTC. Log filenames use the same session name.

Omit `--restore_from` to record with the untrained model. The shell launcher
shows help when invoked with no arguments, so pass any desired simulation flag;
for example:

```bash
bash PRG/long_gray_simulation.sh --duration_ms 60000
```

### Defaults and overrides

| Setting | Default |
| --- | --- |
| Network data | `GLIF_network_nll_full` |
| Simulated neurons | 203,816 |
| Recorded core radius | 400 µm |
| Discarded warm-up | 5,000 ms |
| Recorded duration | 60,000 ms |
| Rollout chunk length | 500 ms |
| Sampling interval | 1 ms |
| Inference precision | float32 |
| Reset scheme | Hard reset (`--hard_reset`) |
| Network seed / recording session | 3000 / 0 |

The network state continues from warm-up through every recording chunk. Warm-up
samples are discarded, and the final chunk is shortened when needed to preserve
the exact requested duration. The full selected network is simulated; only core
neurons are recorded.

Progress lines report each chunk's wall-clock time, total recording time since
warm-up ended, and TensorFlow's peak GPU allocator usage in GiB. The memory peak
includes initialization and warm-up; it is shown as `N/A` when unavailable.
Each chunk is compressed on all allocated CPUs and written while the next chunk
is simulated, so the file's `completed_steps` trails the last progress line by
one chunk until the recording ends.

Pass Python flags through the launcher to change settings:

```bash
bash PRG/long_gray_simulation.sh \
    --restore_from PRG/checkpoints/crowd_osi_ckpt \
    --warmup_ms 1000 --duration_ms 60000 \
    --seed 3000 --run_session 1 \
    --output_file PRG/results/repeat_1/simulation_data.hdf5
```

Keep `--seed` equal to the training seed when restoring a network. Change
`--run_session` for a different LGN/background noise stream of that same network.
Network selection, readout settings, integration scheme, and online voltage
penalty settings must also match training. Training `config.json` files are not
loaded automatically. Existing output files are never overwritten.
Hard reset is enabled by default for PRG recordings. Use `--nohard_reset` only
when intentionally recording a model with soft-reset dynamics.

For all Python options, activate `neuro_tf221` and inspect:

```bash
python3 PRG/long_gray_simulation.py --helpfull
```

## HDF5 format

Each recording contains one trajectory, with no trial dimension:

| Dataset | Shape | Contents |
| --- | --- | --- |
| `spikes` | `(time, core_neurons)` | Binary uint8 spikes |
| `voltages` | `(time, core_neurons)` | Float16 normalized membrane voltages |
| `neuron_ids` | `(core_neurons,)` | Canonical network neuron IDs |
| `bmtk_ids` | `(core_neurons,)` | Corresponding source SONATA/BMTK IDs |

Spike and voltage datasets use chunked gzip level 1 compression. Their columns
and both ID vectors share the same canonical neuron order. Voltages are
dimensionless model values, with reset/rest at 0 and threshold at 1; they are
not millivolts. Cell types are not stored in the recording.

File attributes include `data_dir`, the resolved `checkpoint` prefix (empty for
untrained runs), `dt_ms`, `warmup_ms`, `duration_ms`, `completed_steps`, and the
serialized CLI settings in `flags_json`. Voltage units and normalization are
also recorded on the voltage dataset. The first saved sample is one step after
the end of warm-up.

Data are written incrementally. If a session stops early, use only rows below
`completed_steps`, rather than assuming the entire allocated dataset is valid.

## Load data and recover cell types

Open [analysis.ipynb](analysis.ipynb) using the `neuro_tf221` environment. Set
`recording_path` to the desired HDF5 file, then choose `start_ms`, `window_ms`,
and `max_neurons` to bound the amount of data loaded into memory.

The notebook uses `load_workers=4` CPU threads to decompress the recording's
gzip chunks, without creating a disk cache. Set it to `1` to use the standard
HDF5 reader. Figures reuse loaded windows when possible. Loading the full
60-second, 66,408-neuron recording still requires about 11.1 GiB of RAM for
spikes and voltages combined; parallel loading does not reduce that size.

The notebook uses the saved `bmtk_ids` with the existing `pop_names` and
`pop_name_to_cell_type` utilities to recover one cell type per recorded column,
preserving inhibitory and L5 excitatory subtypes. Canonical `neuron_ids` are
model indices and must not be substituted for source BMTK IDs in this lookup.
No network reconstruction or checkpoint loading is needed.

The original network's `network/v1_nodes.h5` and `network/v1_node_types.csv`
files must remain accessible. If the recording has moved between machines,
update `data_dir` in the notebook to the matching local network data directory.
