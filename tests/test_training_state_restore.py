"""train_end_data's running statistics stay paired with the checkpoint they are restored with."""
import pickle as pkl

import numpy as np

from multi_training import load_checkpoint_training_state, neuron_state_to_save


def _write_run(run, state):
    (run / "Intermediate_checkpoints").mkdir(parents=True)
    with open(run / "train_end_data.pkl", "wb") as handle:
        pkl.dump(state, handle)
    return str(run / "Intermediate_checkpoints" / "ckpt-3")


def test_state_comes_from_the_run_of_the_restored_checkpoint(tmp_path):
    source = {"v1_ema": np.full(3, 0.004), "rolling_osi_emd_state": {"core": {"format": 2}}}
    _write_run(tmp_path / "own_run", {"v1_ema": np.full(3, 0.001)})
    # --restore_from another run: that run's state, not the session's own logdir.
    other_checkpoint = _write_run(tmp_path / "other_run", source)
    loaded = load_checkpoint_training_state(other_checkpoint)
    np.testing.assert_array_equal(loaded["v1_ema"], source["v1_ema"])
    assert loaded["rolling_osi_emd_state"] == source["rolling_osi_emd_state"]
    # No restored checkpoint (fresh weights): no restored statistics either.
    assert load_checkpoint_training_state(None) == {}
    assert load_checkpoint_training_state(str(tmp_path / "missing" / "Best_model" / "ckpt-1")) == {}


def test_diverged_session_keeps_the_state_of_its_restored_checkpoint():
    restored = {
        "v1_ema": np.full(3, 0.002),
        "rolling_osi_emd_state": {"core": {"weight_sum": np.array([0.5])}},
        "epoch_metric_values": {"train_loss": [1.0]},
    }
    current = {"v1_ema": np.full(3, 0.009), "rolling_osi_emd_state": {"core": {}}}
    assert neuron_state_to_save(current, restored, diverged=False) is current
    kept = neuron_state_to_save(current, restored, diverged=True)
    assert set(kept) == {"v1_ema", "rolling_osi_emd_state"}
    assert kept["v1_ema"] is restored["v1_ema"]
    assert kept["rolling_osi_emd_state"] is restored["rolling_osi_emd_state"]
    # A diverged first session (nothing restored) saves no statistics.
    assert neuron_state_to_save(current, {}, diverged=True) == {}
