"""The vectorized OSI/DSI EMD loss against a per-cell-type NumPy reference.

The reference is the loop the loss replaced: per type, keep the active neurons,
resample the type's sorted Neuropixels sample to their count with
``np.linspace``, and average |sorted values - targets|. Gradients are compared
per (cell type, value) tie group as sorted lists, since a tie group may be
assigned to its target ranks in any order.
"""
import numpy as np
import pytest
import tensorflow as tf

from v1_model_utils.loss_functions import OrientationSelectivityLoss


def build(groups, osi_emp, dsi_emp, osi_cost=2.0, alignment=1.0):
    loss = OrientationSelectivityLoss.__new__(OrientationSelectivityLoss)
    loss._dtype = tf.float32
    loss._osi_cost = osi_cost
    loss._batch_emd_alignment_weight = alignment
    loss._rolling_epsilon = tf.constant(1e-6, tf.float32)
    loss._emd_group_indices = tf.ragged.constant(groups, dtype=tf.int32)
    loss._osi_empirical_distributions = tf.ragged.constant(osi_emp, dtype=tf.float32)
    loss._dsi_empirical_distributions = tf.ragged.constant(dsi_emp, dtype=tf.float32)
    loss.cell_type_count = tf.constant([float(len(g)) for g in groups], tf.float32)
    loss._n_node_types = tf.constant(len(groups), tf.int32)
    return loss


def resample(sorted_values, n):
    positions = np.linspace(0.0, sorted_values.size - 1, n)
    lower = np.floor(positions).astype(int)
    upper = np.minimum(lower + 1, sorted_values.size - 1)
    weight = positions - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def reference(groups, osi_emp, dsi_emp, inputs, active, osi_cost=2.0, alignment=1.0):
    """Loss value and gradients of the per-type loop, in float64."""
    inputs = [x.astype(np.float64) for x in inputs]
    grads = [np.zeros_like(x) for x in inputs]
    losses, weights = [], []
    for group, o_emp, d_emp in zip(groups, osi_emp, dsi_emp):
        ids = group[active[group]]
        n = ids.size
        weights.append(float(len(group)) if n else 0.0)
        if not n:
            losses.append(0.0)
            continue
        value = 0.0
        for k, emp in ((0, o_emp), (2, d_emp)):
            order = np.argsort(inputs[k][ids], kind="stable")
            diff = inputs[k][ids][order] - resample(np.asarray(emp, np.float64), n)
            value += np.abs(diff).mean()
            grads[k][ids[order]] += np.sign(diff) / n
        for k in (1, 3):
            value += alignment * np.abs(inputs[k][ids]).mean()
            grads[k][ids] += alignment * np.sign(inputs[k][ids]) / n
        losses.append(value)
    weights = np.asarray(weights)
    total = max(weights.sum(), 1e-6)
    scale = np.zeros(len(inputs[0]))
    for group, w in zip(groups, weights):
        scale[group] = w / total * osi_cost
    return (np.dot(losses, weights) / total * osi_cost, [g * scale for g in grads])


def make_case(seed, sizes, active_frac, quantize=None):
    rng = np.random.default_rng(seed)
    n_neurons = int(np.sum(sizes))
    perm = rng.permutation(n_neurons)
    bounds = np.cumsum([0, *sizes])
    groups = [np.sort(perm[a:b]) for a, b in zip(bounds[:-1], bounds[1:])]
    osi_emp = [np.sort(rng.uniform(-0.2, 1.0, rng.integers(1, 300))) for _ in sizes]
    dsi_emp = [np.sort(rng.uniform(-0.2, 1.0, rng.integers(1, 300))) for _ in sizes]
    inputs = [rng.uniform(-1, 1, n_neurons).astype(np.float32) for _ in range(4)]
    if quantize:
        inputs = [(np.round(x * quantize) / quantize).astype(np.float32) for x in inputs]
    active = rng.uniform(size=n_neurons) < active_frac
    active[groups[0]] = False  # one silent cell type
    active[groups[1]] = False
    active[groups[1][0]] = True  # one type with a single active neuron
    return groups, osi_emp, dsi_emp, inputs, active


@pytest.mark.parametrize("quantize", [None, 8])
def test_vectorized_emd_matches_per_type_reference(quantize):
    groups, osi_emp, dsi_emp, inputs, active = make_case(
        3, [50, 30, 400, 1, 2, 900, 120], 0.6, quantize
    )
    variables = [tf.Variable(x) for x in inputs]
    loss = build(groups, osi_emp, dsi_emp)  # tf.function methods hold `self` weakly
    with tf.GradientTape() as tape:
        value = loss._signed_selectivity_emd_loss(
            *variables, tf.constant(active)
        )
    grads = [tf.convert_to_tensor(g).numpy() for g in tape.gradient(value, variables)]
    ref_value, ref_grads = reference(groups, osi_emp, dsi_emp, inputs, active)

    np.testing.assert_allclose(value.numpy(), ref_value, rtol=1e-5)
    for grad, ref_grad, x in zip(grads, ref_grads, inputs):
        for group in groups:
            keys = np.stack([x[group], active[group]], axis=1)
            _, tie_id = np.unique(keys, axis=0, return_inverse=True)
            for t in np.unique(tie_id):
                members = group[tie_id == t]
                np.testing.assert_allclose(
                    np.sort(grad[members]), np.sort(ref_grad[members]), rtol=1e-5, atol=1e-8
                )


def test_all_silent_types_give_zero_loss_and_gradient():
    groups, osi_emp, dsi_emp, inputs, active = make_case(4, [10, 20], 0.0)
    active[:] = False
    variables = [tf.Variable(x) for x in inputs]
    loss = build(groups, osi_emp, dsi_emp)  # tf.function methods hold `self` weakly
    with tf.GradientTape() as tape:
        value = loss._signed_selectivity_emd_loss(
            *variables, tf.constant(active)
        )
    assert value.numpy() == 0.0
    for grad in tape.gradient(value, variables):
        assert not np.any(tf.convert_to_tensor(grad).numpy())
