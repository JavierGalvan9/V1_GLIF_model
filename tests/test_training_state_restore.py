"""The running statistics saved beside a checkpoint are restored with it."""
import numpy as np
import tensorflow as tf

from v1_model_utils import tf_utils


def test_each_checkpoint_keeps_its_own_state_and_deleted_ones_lose_theirs(tmp_path):
    checkpoint = tf.train.Checkpoint(step=tf.Variable(0))
    manager = tf.train.CheckpointManager(checkpoint, str(tmp_path / "Best_model"), max_to_keep=2)
    paths = []
    for epoch in range(1, 4):
        paths.append(manager.save(checkpoint_number=epoch))
        tf_utils.save_checkpoint_neuron_state(manager, paths[-1], {"v1_ema": np.full(3, float(epoch))})

    # A restore of an older (e.g. best) checkpoint gets that checkpoint's state.
    np.testing.assert_array_equal(tf_utils.load_checkpoint_neuron_state(paths[1])["v1_ema"], 2.0)
    np.testing.assert_array_equal(tf_utils.load_checkpoint_neuron_state(paths[2])["v1_ema"], 3.0)
    # The checkpoint the manager deleted takes its state with it.
    assert tf_utils.load_checkpoint_neuron_state(paths[0]) == {}
    assert sorted(p.name for p in (tmp_path / "Best_model").glob("*" + tf_utils.NEURON_STATE_SUFFIX)) == [
        "ckpt-2" + tf_utils.NEURON_STATE_SUFFIX, "ckpt-3" + tf_utils.NEURON_STATE_SUFFIX]
    # No restored checkpoint (fresh weights): no restored statistics either.
    assert tf_utils.load_checkpoint_neuron_state(None) == {}
