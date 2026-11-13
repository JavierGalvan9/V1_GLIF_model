"""Small CPU checks for continuous PRG recordings and their analysis notebook."""

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from PRG import long_gray_simulation as gray


REPO_ROOT = Path(__file__).resolve().parents[1]


def recording_flags(tmp_path, **overrides):
    return SimpleNamespace(
        **dict(seq_len=4, warmup_ms=5, duration_ms=11, data_dir=str(tmp_path))
        | overrides
    )


def test_exact_durations_state_continuity_and_canonical_columns(tmp_path):
    flags = recording_flags(tmp_path)
    calls = []

    def step(state, n_steps, record):
        calls.append((state, n_steps, record))
        times = np.arange(state + 1, state + n_steps + 1)[:, None]
        values = times + np.arange(3)[None, :]
        return state + n_steps, values % 2, values / 10

    output = tmp_path / "recording.hdf5"
    final_state = gray.record_simulation(
        output, flags, np.array([5, 1, 3]), np.array([2, 0, 1]), 0, step,
        checkpoint="test/ckpt-1",
    )
    assert final_state == 16
    assert calls == [
        (0, 4, False), (4, 1, False), (5, 4, True),
        (9, 4, True), (13, 3, True),
    ]
    expected = np.arange(6, 17)[:, None] + np.array([1, 2, 0])[None, :]
    with h5py.File(output, "r") as saved:
        assert set(saved) == {"spikes", "voltages", "neuron_ids", "bmtk_ids"}
        np.testing.assert_array_equal(saved["neuron_ids"][:], [1, 3, 5])
        np.testing.assert_array_equal(saved["bmtk_ids"][:], [0, 1, 2])
        np.testing.assert_array_equal(saved["spikes"][:], expected % 2)
        np.testing.assert_array_equal(saved["voltages"][:], (expected / 10).astype("float16"))
        assert saved["spikes"].dtype == np.uint8
        assert saved["voltages"].dtype == np.float16
        for name in ("spikes", "voltages"):
            assert saved[name].shape == (11, 3)
            assert saved[name].compression == "gzip"
            assert saved[name].compression_opts == 1
        assert saved.attrs["completed_steps"] == 11
        assert saved.attrs["dt_ms"] == 1
        assert saved.attrs["warmup_ms"] == 5
        assert saved.attrs["checkpoint"] == "test/ckpt-1"
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        gray.record_simulation(output, flags, [0], [0], 0, step)
    assert output.read_bytes() == original


def test_incomplete_recording_reports_only_committed_chunks(tmp_path):
    flags = recording_flags(tmp_path, warmup_ms=0)

    def step(state, n_steps, record):
        if state:
            raise RuntimeError("interrupted")
        return n_steps, np.zeros((n_steps, 1)), np.ones((n_steps, 1))

    output = tmp_path / "partial.hdf5"
    with pytest.raises(RuntimeError, match="interrupted"):
        gray.record_simulation(output, flags, [0], [0], 0, step)
    with h5py.File(output, "r") as saved:
        assert saved.attrs["completed_steps"] == 4
        np.testing.assert_array_equal(saved["voltages"][:4], 1)


@pytest.mark.parametrize("full_window", [False, True])
def test_notebook_loads_windows_and_derives_types_from_shuffled_bmtk_ids(tmp_path, monkeypatch, full_window):
    network_dir = tmp_path / "network"
    network_dir.mkdir()
    (network_dir / "v1_node_types.csv").write_text(
        "node_type_id pop_name\n10 e5ET\n20 i4Pvalb\n30 i23Sst\n"
    )
    with h5py.File(network_dir / "v1_nodes.h5", "w") as nodes:
        nodes.create_dataset("nodes/v1/node_type_id", data=[10, 20, 30])
        nodes.create_dataset("nodes/v1/0/tuning_angle", data=[90.0, 45.0, 0.0])
    output = tmp_path / "recording.hdf5"

    def step(state, n_steps, record):
        return state + n_steps, np.ones((n_steps, 3)), np.zeros((n_steps, 3))

    gray.record_simulation(
        output, recording_flags(tmp_path), [0, 1, 2], [2, 0, 1], 0, step,
    )
    namespace = dict(
        recording_path=output, data_dir_override=None,
        start_ms=0 if full_window else 2, window_ms=11 if full_window else 5,
        max_neurons=3 if full_window else 2, load_workers=2,
    )
    notebook = json.loads((REPO_ROOT / "PRG/analysis.ipynb").read_text())
    monkeypatch.chdir(REPO_ROOT / "PRG")
    for index, cell in enumerate(notebook["cells"]):
        source = "".join(cell["source"])
        if cell["cell_type"] == "code" and cell.get("id") != "prg-02":
            exec(compile(source, f"analysis.ipynb:cell{index}", "exec"), namespace)
    expected_shape = (11, 3) if full_window else (5, 2)
    assert namespace["spikes"].shape == namespace["voltages"].shape == expected_shape
    np.testing.assert_array_equal(namespace["cell_types"], ["L2/3 SST", "L5 ET", "L4 PV"])
    np.testing.assert_array_equal(namespace["neuron_info"]["bmtk_id"], [2, 0, 1])
    expected_times = np.arange(1, 12) if full_window else np.arange(3, 8)
    np.testing.assert_array_equal(namespace["time_ms"], expected_times)
    assert len(namespace["fig"].axes) == 2
    assert namespace["figure_spikes"].shape == (11, expected_shape[1])
    assert np.shares_memory(namespace["figure_spikes"], namespace["spikes"]) == full_window
    np.testing.assert_array_equal(namespace["trace_columns"], [0, 1, 2] if full_window else [0, 2])
    np.testing.assert_array_equal(namespace["raster_network"]["tf_id_to_bmtk_id"], [2, 0, 1] if full_window else [2, 1])
    np.testing.assert_array_equal(namespace["raster_network"]["tuning_angle"], [0, 90, 45] if full_window else [0, 45])


def test_tiny_glif_state_only_warmup_matches_uninterrupted_rollout(tmp_path):
    import tensorflow as tf

    from tests.test_inference_mode import _tiny_network
    from v1_model_utils import models

    network, lgn, background = _tiny_network(tmp_path)
    model = models.create_model(
        network, lgn, background, seq_len=4, n_input=1,
        batch_size=1, dtype=tf.float32, use_state_input=True, return_state=True,
        add_metric=False, current_input=True, acceleration="tensorflow",
        train_input=False, train_noise=False, train_recurrent=False,
        output_spike_dtype=tf.uint8, output_neuron_ids=np.array([1, 0]), seed=7,
    )
    rsnn = model.get_layer("rsnn")
    sequence_model = models.build_sequence_and_state_model(model, rsnn)
    zero_state = tuple(rsnn.cell.zero_state(1, tf.float32))
    step = gray.make_step(model, tf.ones((4, 1), tf.float32), tf.float32, seed=7)

    output = tmp_path / "tiny.hdf5"
    final_state = gray.record_simulation(
        output, recording_flags(tmp_path), [1, 0], [1, 0], zero_state, step,
    )
    assert int(rsnn.cell.noise_stream) == 5  # Warm-up and recording use fresh draws.
    reference = tf.nest.flatten(sequence_model([tf.ones((1, 16, 1)), *zero_state]))
    with h5py.File(output, "r") as saved:
        np.testing.assert_array_equal(saved["spikes"][:], reference[0][0, 5:, ::-1])
        np.testing.assert_array_equal(
            saved["voltages"][:], reference[1][0, 5:, ::-1].numpy().astype(np.float16),
        )
    for actual, expected in zip(final_state, reference[2:], strict=True):
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("arguments", [
    [], ["--duration_ms=0", "--evaluate_untrained"],
    ["--warmup_ms=-1", "--evaluate_untrained"],
    ["--seq_len=0", "--evaluate_untrained"],
    ["--n_input=17401", "--evaluate_untrained"],
    ["--restore_from=test", "--evaluate_untrained"],
])
def test_invalid_cli_settings_are_rejected(arguments):
    from absl import flags

    options = gray.register_flags(flags.FlagValues())
    options(["long_gray_simulation.py", *arguments])
    with pytest.raises(ValueError):
        gray.validate_flags(options)


def test_cli_defaults():
    from absl import flags

    options = gray.register_flags(flags.FlagValues())
    options(["long_gray_simulation.py", "--evaluate_untrained"])
    gray.validate_flags(options)
    assert (options.warmup_ms, options.duration_ms, options.seq_len) == (5000, 60000, 500)
    assert (options.neurons, options.plot_core_radius, options.dtype) == (203816, 400, "float32")
    assert options.hard_reset is True


def test_main_restores_a_checkpoint_into_compact_core_output(tmp_path, monkeypatch):
    import tensorflow as tf
    from absl import flags

    import stim_dataset
    from tests.test_inference_mode import _tiny_network
    from v1_model_utils import cuda_caches, load_sparse, models, other_v1_utils, tf_utils

    network, lgn, background = _tiny_network(tmp_path)
    network["tf_id_to_bmtk_id"] = np.array([2, 0])
    source = models.create_model(
        network, lgn, background, seq_len=4, n_input=1, batch_size=1,
        dtype=tf.float32, use_state_input=True, return_state=True,
        add_metric=False, acceleration="tensorflow", track_voltage_penalty=True,
        train_input=False, train_noise=False, train_recurrent=False, hard_reset=True, seed=7,
    )
    source.get_layer("rsnn").cell.input_weight_values.assign([3.5, 4.5])
    checkpoint_dir = tmp_path / "Best_model"
    checkpoint_dir.mkdir()
    checkpoint = tf.train.Checkpoint(model=source).save(str(checkpoint_dir / "ckpt"))
    output = tmp_path / "main.hdf5"
    options = gray.register_flags(flags.FlagValues())
    options([
        "long_gray_simulation.py", f"--restore_from={checkpoint_dir}",
        f"--data_dir={tmp_path}", f"--output_file={output}",
        "--n_input=1", "--neurons=2", "--seq_len=4", "--warmup_ms=3",
        "--duration_ms=7", "--acceleration=tensorflow", "--neuron_layout=canonical",
        "--lgn_backend=tensorflow", "--seed=7",
    ])
    monkeypatch.setattr(gray, "absl_flags", SimpleNamespace(FLAGS=options))
    monkeypatch.setattr(cuda_caches, "configure_cuda_caches", lambda: None)
    monkeypatch.setattr(tf_utils, "print_tensorflow_runtime_info", lambda: None)
    monkeypatch.setattr(load_sparse, "cached_load_v1", lambda *a, **k: (network, lgn, background))
    monkeypatch.setattr(other_v1_utils, "isolate_core_neurons", lambda *a, **k: np.array([False, True]))
    monkeypatch.setattr(
        stim_dataset, "load_or_compute_spontaneous_lgn_probabilities",
        lambda **kwargs: tf.ones((4, 1), tf.float32),
    )
    built = []
    create_model = models.create_model

    def capture_model(*args, **kwargs):
        model = create_model(*args, **kwargs)
        built.append(model)
        return model

    monkeypatch.setattr(models, "create_model", capture_model)
    gray.main(None)
    assert len(built) == 2  # Compact recorder and full-width checkpoint donor.
    np.testing.assert_array_equal(
        built[0].get_layer("rsnn").cell.input_weight_values, [3.5, 4.5],
    )
    with h5py.File(output, "r") as saved:
        assert saved.attrs["checkpoint"] == checkpoint
        assert saved.attrs["completed_steps"] == 7
        assert saved["spikes"].shape == saved["voltages"].shape == (7, 1)
        np.testing.assert_array_equal(saved["neuron_ids"][:], [1])
        np.testing.assert_array_equal(saved["bmtk_ids"][:], [0])


@pytest.mark.parametrize("with_compiler_hook", [False, True])
def test_launcher_quotes_arguments_without_submitting_a_job(tmp_path, with_compiler_hook):
    project = tmp_path / "project with spaces"
    scripts = project / "PRG"
    scripts.mkdir(parents=True)
    launcher = scripts / "long_gray_simulation.sh"
    launcher.write_text((REPO_ROOT / "PRG/long_gray_simulation.sh").read_text())
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    conda_base = tmp_path / "conda"
    (conda_base / "etc/profile.d").mkdir(parents=True)
    conda_init = conda_base / "etc/profile.d/conda.sh"
    compiler_hook = Path(sys.prefix) / "etc/conda/deactivate.d/deactivate-gxx_linux-64.sh"
    if with_compiler_hook:
        if not compiler_hook.is_file():
            pytest.skip("Conda compiler deactivation hook is not installed.")
        conda_init.write_text(f"conda() {{ source {shlex.quote(str(compiler_hook))}; }}\n")
    else:
        conda_init.touch()
    fake_conda = bin_dir / "conda"
    fake_conda.write_text(
        f"#!{sys.executable}\nimport sys\n"
        f"if sys.argv[1:] == ['info', '--base']: print({str(conda_base)!r})\n"
    )
    captured = tmp_path / "submitted.json"
    fake_run = bin_dir / "run"
    fake_run.write_text(
        f"#!{sys.executable}\nimport json, sys\nfrom pathlib import Path\n"
        f"Path({str(captured)!r}).write_text(json.dumps(sys.argv[1:]))\n"
    )
    for executable in (fake_conda, fake_run):
        executable.chmod(0o755)
    checkpoint = str(tmp_path / "checkpoint with spaces; $(false) `false`")
    environment = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}")
    if with_compiler_hook:
        environment = {
            key: value for key, value in environment.items()
            if not key.startswith("CONDA_BACKUP_")
        }
        environment["CONDA_PREFIX"] = sys.prefix
    result = subprocess.run(
        ["bash", str(launcher), "--restore_from", checkpoint],
        env=environment, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    submitted = json.loads(captured.read_text())
    assert submitted[:10] == [
        "-g", "1", "-G", "rtx3090", "-c", "8", "-m", "80", "-t", "1:00",
    ]
    command = shlex.split(submitted[-1])
    assert command[:3] == ["python3", "-u", "PRG/long_gray_simulation.py"]
    assert command[-2:] == ["--restore_from", checkpoint]
    assert command[3] == "--output_file"
    assert str(project / "PRG/results") in command[4]
    assert submitted[submitted.index("-o") + 1].startswith("PRG/Out/")
    assert submitted[submitted.index("-e") + 1].startswith("PRG/Error/")
    assert (project / "PRG/Out").is_dir() and (project / "PRG/Error").is_dir()
