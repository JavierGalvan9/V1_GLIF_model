import numpy as np
import tensorflow as tf
import os
import pickle as pkl
from tensorflow.python.eager import record
from .cuda_synaptic_currents import (
    build_csr_connectivity,
    calculate_synaptic_currents as calculate_synaptic_currents_cuda,
)
from .cuda_external_currents import (
    build_csr_connectivity as build_external_csr_connectivity,
    calculate_external_currents,
)
from numba import njit

# import subprocess
# from . import other_v1_utils

# Define a custom gradient for the spike function.
# Diverse functions can be used to define the gradient.
# Here we provide variations depending on the gradient type.
def gauss_pseudo(v_scaled, sigma, amplitude):
    dtype = v_scaled.dtype
    sigma = tf.cast(sigma, dtype)
    amplitude = tf.cast(amplitude, dtype)
    return tf.math.exp(-tf.square(v_scaled) / tf.square(sigma)) * amplitude

def pseudo_derivative(v_scaled, dampening_factor):
    dtype = v_scaled.dtype
    dampening_factor = tf.cast(dampening_factor, dtype)
    one = tf.cast(1.0, dtype)
    zero = tf.cast(0.0, dtype)
    return dampening_factor * tf.maximum(one - tf.abs(v_scaled), zero)

def slayer_pseudo(v_scaled, sigma, amplitude):
    dtype = v_scaled.dtype
    sigma = tf.cast(sigma, dtype)
    amplitude = tf.cast(amplitude, dtype)
    return tf.math.exp(-sigma * tf.abs(v_scaled)) * amplitude


@tf.custom_gradient
def _range_voltage_penalty_mean(voltage, inverse_n_neurons):
    """Compute the neuron mean and return it in the voltage compute dtype."""
    centered = voltage - tf.cast(0.5, voltage.dtype)
    outside = tf.nn.relu(tf.abs(centered) - tf.cast(0.5, voltage.dtype))
    mean_penalty = (
        tf.reduce_sum(tf.cast(tf.square(outside), tf.float32), axis=1)
        * inverse_n_neurons
    )
    mean_penalty = tf.cast(mean_penalty, voltage.dtype)

    def grad(dy):
        gradient_factor = (
            tf.cast(2.0, voltage.dtype) * outside * tf.sign(centered)
        )
        reduction_gradient = dy[:, None] * tf.cast(inverse_n_neurons, voltage.dtype)
        return reduction_gradient * gradient_factor, None

    return mean_penalty, grad


@tf.custom_gradient
def _threshold_voltage_penalty_mean(voltage, inverse_n_neurons):
    """Compute the threshold mean and return it in the voltage compute dtype."""
    offset = voltage - tf.cast(1.0, voltage.dtype)
    mean_penalty = (
        tf.reduce_sum(tf.cast(tf.square(offset), tf.float32), axis=1)
        * inverse_n_neurons
    )
    mean_penalty = tf.cast(mean_penalty, voltage.dtype)

    def grad(dy):
        # Compute the gradient of the mean penalty with respect to the voltage.
        gradient_factor = tf.cast(2.0, voltage.dtype) * offset
        reduction_gradient = dy[:, None] * tf.cast(inverse_n_neurons, voltage.dtype)
        return reduction_gradient * gradient_factor, None

    return mean_penalty, grad


def voltage_penalty_mean_step(voltage, n_neurons, penalty_mode="range"):
    """Return the compute-dtype neuron-mean penalty for one timestep."""
    inverse_n_neurons = tf.math.reciprocal(tf.cast(n_neurons, tf.float32))
    if penalty_mode == "range":
        return _range_voltage_penalty_mean(voltage, inverse_n_neurons)
    if penalty_mode == "threshold":
        return _threshold_voltage_penalty_mean(voltage, inverse_n_neurons)
    raise ValueError("penalty_mode must be 'range' or 'threshold'.")


def reset_voltage_penalty_state(cell, state):
    """Reset only an online voltage accumulator, preserving all neural state."""
    state = tuple(state)
    if not getattr(cell, "_track_voltage_penalty", False):
        return state
    return state[:-1] + (tf.zeros_like(state[-1]),)

@tf.custom_gradient
def spike_gauss(v_scaled, sigma, amplitude):
    dtype = v_scaled.dtype
    z_ = tf.greater(v_scaled, tf.cast(0.0, dtype))
    z_ = tf.cast(z_, dtype)

    def grad(dy):
        # de_dz = tf.cast(dy, dtype)
        de_dz = dy
        dz_dv_scaled = gauss_pseudo(v_scaled, sigma, amplitude)
        de_dv_scaled = de_dz * dz_dv_scaled

        return [de_dv_scaled, None, None]

    return tf.identity(z_, name='spike_gauss'), grad

@tf.custom_gradient
def spike_function(v_scaled, dampening_factor):
    dtype = v_scaled.dtype
    z_ = tf.greater(v_scaled, tf.cast(0.0, dtype))
    z_ = tf.cast(z_, dtype)

    def grad(dy):
        # de_dz = tf.cast(dy, dtype)
        de_dz = dy
        dz_dv_scaled = pseudo_derivative(v_scaled, dampening_factor)
        de_dv_scaled = de_dz * dz_dv_scaled

        return [de_dv_scaled, None]

    return tf.identity(z_, name="spike_function"), grad

@tf.custom_gradient
def spike_slayer(v_scaled, sigma, amplitude):
    dtype = v_scaled.dtype
    z_ = tf.greater(v_scaled, tf.cast(0.0, dtype))
    z_ = tf.cast(z_, dtype)

    def grad(dy):
        # de_dz = tf.cast(dy, dtype)
        de_dz = dy
        dz_dv_scaled = slayer_pseudo(v_scaled, sigma, amplitude)
        de_dv_scaled = de_dz * dz_dv_scaled

        return [de_dv_scaled, tf.zeros_like(sigma), tf.zeros_like(amplitude)]

    return tf.identity(z_, name="spike_slayer"), grad


# def _coalesce_indexed_slices_1d(values, indices, dense_size, out_index_dtype=tf.int32):
#     """Coalesce duplicate indices for a 1D target variable gradient."""
#     indices = tf.cast(indices, out_index_dtype)
#     unique_idx, inverse_pos = tf.unique(indices, out_idx=tf.int32)
#     num_unique = tf.shape(unique_idx, out_type=tf.int32)[0]
#     coalesced_values = tf.math.unsorted_segment_sum(
#         data=values,
#         segment_ids=inverse_pos,
#         num_segments=num_unique
#     )
#     dense_shape = tf.cast(tf.reshape(dense_size, [1]), out_index_dtype)
#     return tf.IndexedSlices(
#         values=coalesced_values,
#         indices=unique_idx,
#         dense_shape=dense_shape
#     )

# @tf.function(jit_compile=True) # (No registered 'RaggedGather' OpKernel for XLA_GPU_JIT devices)
@tf.custom_gradient
def calculate_synaptic_currents(rec_z_buf, synapse_indices, weight_values,
                                weight_values_compute, dense_shape, synaptic_basis_weights,
                                syn_ids, pre_ind_table, dampening_factor):
    """
    Optimized synaptic current calculation with memory-efficient gradients for RSNN timestep iteration.

    Mathematical formulation:
    - Forward: I[b,post,r] = sum_pre(spike[b,pre] * W[post,pre] * basis[type[post,pre], r])
    - Grad w.r.t spike: dL/dspike[b,pre] = dampening_factor
      * sum_post sum_r(dL/dI[b,post,r] * W[post,pre] * basis[type,r])
    - Grad w.r.t W: dL/dW[post,pre] = sum_b sum_r(dL/dI[b,post,r] * spike[b,pre] * basis[type,r])

    Key optimizations:
    1. Use int32 for GPU-bound operations (segment_ids, arithmetic, unsorted_segment_sum)
    2. Use int64 for CPU-bound operations (RaggedTensor gather) and SparseTensor indices (required)
    3. Recompute cheap operations in backward pass to minimize saved activations (VRAM)
    4. Use einsum for fused multiply-sum operations on weight gradients
    """
    # Get batch size and network dimensions
    # tf.shape returns int32, dense_shape is int64 (for SparseTensor compatibility)
    batch_size = tf.cast(tf.shape(rec_z_buf)[0], dtype=tf.int64)
    n_post_neurons = dense_shape[0]  # int64
    compute_dtype = rec_z_buf.dtype  # e.g., float16 for mixed precision

    # Find non-zero spike indices
    # tf.where() returns int64 by default (GPU optimized for this operation)
    non_zero_indices = tf.where(rec_z_buf > 0)  # [num_spikes, 2], int64
    batch_indices = non_zero_indices[:, 0]  # int64
    pre_neuron_indices = non_zero_indices[:, 1]  # keep int64 for RaggedTensor gather (CPU-optimized)

    # Retrieve connections and weights for active presynaptic neurons
    # This uses pre_ind_table (RaggedTensor), which benefits from int64 on CPU
    post_neuron_indices, new_weights, new_syn_ids, post_in_degree = (
        get_active_synapse_data_table(
            synapse_indices,
            weight_values_compute,
            syn_ids,
            pre_neuron_indices,
            pre_ind_table,
        )
    )
    # new_syn_ids = tf.cast(new_syn_ids, dtype=tf.int32)  # int32 to reduce VRAM since its reused in backward pass

    # Returns: post_neuron_indices (int64), new_syn_ids (int64), post_in_degree (int32),
    # all_synaptic_inds (int32)

    # Build segment IDs for unsorted_segment_sum using int32 for the GPU kernel.
    batch_indices_per_connection = tf.repeat(batch_indices, post_in_degree)  # int64
    num_segments = batch_size * n_post_neurons  # int64
    segment_ids = batch_indices_per_connection * n_post_neurons + post_neuron_indices  # int64
    segment_ids = tf.cast(segment_ids, dtype=tf.int32)
    num_segments = tf.cast(num_segments, dtype=tf.int32)

    # Compute weighted basis factors for active synapses
    # Note: basis_factors will be recomputed in grad() to save VRAM (cheap gather operation)
    # basis_factors = tf.cast(
    #     tf.gather(synaptic_basis_weights, new_syn_ids, axis=0), compute_dtype
    # )  # [n_active, n_basis], compute_dtype
    basis_factors = tf.gather(synaptic_basis_weights, new_syn_ids, axis=0)  # [n_active, n_basis]
    # new_syn_ids = tf.cast(new_syn_ids, dtype=tf.int32)  # int32 to reduce VRAM since its reused in backward pass
    # Mixed precision: cast gathered subsets to compute_dtype (float16) AFTER gather.
    # This avoids creating a temporary float16 copy of the full weight table (~23M elements)
    # and only casts the ~1-2M active connections — 10-20x less cast work and no extra VRAM.
    # new_weights = tf.cast(new_weights, compute_dtype)
    weighted_basis = new_weights[:, tf.newaxis] * basis_factors  # [n_active, n_basis], compute_dtype
    # weighted_basis = new_weights[:, tf.newaxis] * basis_factors  # [n_active, n_basis], compute_dtype --- IGNORE ---

    # if per_type_training:
    #     per_type_weights = tf.expand_dims(tf.gather(recurrent_per_type_weight_values,
    #                                                 tf.gather(connection_type_ids, all_synaptic_inds)), axis=1)
    #     new_weights = new_weights * per_type_weights

    # Sum to get currents per (batch, neuron, receptor) — float32
    i_rec_flat = tf.math.unsorted_segment_sum(weighted_basis, segment_ids, num_segments)

    # if i_rec_flat.dtype != compute_dtype:
    #     i_rec_flat = tf.cast(i_rec_flat, dtype=compute_dtype)

    def grad(dy):
        # Gradient computation - recompute cheap operations to save VRAM since this runs every timestep
        # dy arrives in compute_dtype (float16) since i_rec_flat is in compute_dtype

        # =================================================================
        # GRADIENT W.R.T. INPUT SPIKES (rec_z_buf)
        # =================================================================
        # dL/dspike[b,pre] = sum_post,r( dy[b,post,r] * W[post,pre] * basis[type,r] )
        n_syn_basis = tf.shape(synaptic_basis_weights, out_type=tf.int32)[1]
        n_post_neurons_g = dense_shape[0]  # int64
        n_pre_neurons_g = dense_shape[1]  # int64
        # weight_values_g = weight_values_compute

        def per_receptor_accum(r_id, acc):
            dy_r = tf.reshape(dy[:, r_id], [batch_size, n_post_neurons_g])
            recurrent_weights_factors = tf.gather(synaptic_basis_weights[:, r_id], syn_ids, axis=0)
            weights_syn_receptors = weight_values_compute * recurrent_weights_factors
            sparse_w_rec = tf.sparse.SparseTensor(synapse_indices, weights_syn_receptors, dense_shape)
            de_dv_rid = tf.sparse.sparse_dense_matmul(dy_r, sparse_w_rec, adjoint_a=False)

            return r_id + 1, acc + de_dv_rid

        init_acc = tf.zeros((batch_size, n_pre_neurons_g), dtype=compute_dtype)
        _, de_dv= tf.while_loop(
            lambda r_id, _: r_id < n_syn_basis,
            per_receptor_accum,
            [tf.constant(0, dtype=tf.int32), init_acc],
            parallel_iterations=1,
        )
        # de_dv = tf.cast(de_dv, rec_z_buf.dtype)
        de_dv *= tf.cast(dampening_factor, de_dv.dtype)

        # # Extract the gradient for this receptor type (shape: [batch_size, n_post_neurons])
        # r_id = 0
        # dy_r = tf.reshape(dy[:, r_id], [batch_size, n_post_neurons])
        # # dy_r = dy_reshaped[:, :, r_id]
        # # Compute gradient w.r.t rec_z_buf for this receptor type
        # recurrent_weights_factors = tf.gather(synaptic_basis_weights[:, r_id], syn_ids, axis=0)
        # weights_syn_receptors = weight_values * recurrent_weights_factors
        # sparse_w_rec = tf.sparse.SparseTensor(synapse_indices, weights_syn_receptors, dense_shape)
        # de_dv_rid = tf.sparse.sparse_dense_matmul(dy_r, sparse_w_rec, adjoint_a=False)
        # de_dv = tf.cast(de_dv_rid, dtype=rec_z_buf.dtype)

        # =================================================================
        # GRADIENT W.R.T. WEIGHTS
        # =================================================================
        # For active synapses: dL/dW[syn] = sum_b,r( dy[b,post[syn],r] * basis[type[syn],r] )

        # Rebuild active synapse metadata here instead of keeping large per-connection
        # indexing tensors alive across the forward pass.
        post_neuron_indices_g, new_syn_ids_g, post_in_degree_g, all_synaptic_inds_g = (
            get_active_synapse_metadata_table(
                synapse_indices,
                syn_ids,
                pre_neuron_indices,
                pre_ind_table,
            )
        )
        batch_indices_per_connection_g = tf.repeat(batch_indices, post_in_degree_g)
        segment_ids_g = batch_indices_per_connection_g * n_post_neurons_g + post_neuron_indices_g
        segment_ids_g = tf.cast(segment_ids_g, dtype=tf.int32)

        # Gather gradients for active connections using reconstructed segment ids.
        dnew_weights = tf.gather(dy, segment_ids_g)  # [n_active, n_basis]
        basis_factors_grad = tf.gather(synaptic_basis_weights, new_syn_ids_g, axis=0)
        # Compute weight gradients in master-weight dtype (typically float32) to avoid
        # quantizing dW in mixed precision before optimizer update.
        # basis_factors_grad = tf.cast(basis_factors_grad, weight_values.dtype)
        de_dweight_values_connection = tf.einsum('cr,cr->c', dnew_weights, basis_factors_grad)
        # de_dweight_values_connection = tf.reduce_sum(dnew_weights * basis_factors_grad, axis=1)  # [n_active], compute_dtype

        # Accumulate to original synapse positions
        # Instead of tensor_scatter_nd_add, use unsorted_segment_sum:
        de_dweight_values = tf.math.unsorted_segment_sum(
            data=de_dweight_values_connection,
            segment_ids=all_synaptic_inds_g,
            num_segments=tf.shape(weight_values)[0]
        )
        de_dweight_values = tf.cast(de_dweight_values, dtype=weight_values.dtype)

        # de_dweight_values_connection = tf.cast(de_dweight_values_connection, dtype=weight_values.dtype)
        # de_dweight_values = _coalesce_indexed_slices_1d(
        #     values=de_dweight_values_connection,
        #     indices=all_synaptic_inds,
        #     dense_size=tf.shape(weight_values, out_type=tf.int32)[0],
        #     out_index_dtype=tf.int32
        # )

        return [
            de_dv,              # Gradient w.r.t rec_z_buf (compute_dtype)
            None,               # synapse_indices (constant)
            de_dweight_values,  # Gradient w.r.t weight_values (float32, matches master weights)
            None,               # weight_values_compute (non-trainable shadow copy)
            None,               # dense_shape[0] (constant)
            None,               # dense_shape[1] (constant)
            None,               # synaptic_basis_weights (constant)
            None,               # syn_ids (constant)
            None,               # pre_ind_table (constant)
            None                # dampening_factor (constant)
        ]

    return i_rec_flat, grad


def exp_convolve(tensor, decay=0.8, reverse=False, initializer=None, axis=0):
    rank = len(tensor.get_shape())
    perm = np.arange(rank)
    perm[0], perm[axis] = perm[axis], 0
    tensor = tf.transpose(tensor, perm)

    if initializer is None:
        initializer = tf.zeros_like(tensor[0])

    def scan_fun(_acc, _t):
        return _acc * decay + _t

    filtered = tf.scan(scan_fun, tensor, reverse=reverse,
                       initializer=initializer)

    filtered = tf.transpose(filtered, perm)
    return filtered

def straight_through_dampen(x, dampening):
    """
    Keep the forward value unchanged while scaling the backward pass by (1 - dampening).

    We use this on the membrane self-loop after detaching the reset term. That targets
    the long voltage chain directly without weakening the synaptic/current gradients.
    """
    dtype = x.dtype
    dampening = tf.cast(dampening, dtype)
    zero = tf.cast(0.0, dtype)
    one = tf.cast(1.0, dtype)
    dampening = tf.clip_by_value(dampening, zero, one)
    return x * (one - dampening) + tf.stop_gradient(x * dampening)


def _validate_inputs(indices, n_source_neurons):
    n_source_neurons = int(n_source_neurons)
    if n_source_neurons <= 0:
        raise ValueError(f'`n_source_neurons` must be > 0, got {n_source_neurons}.')

    indices_np = np.asarray(indices)
    if indices_np.ndim != 2 or indices_np.shape[1] < 2:
        raise ValueError(f'`indices` must have shape [n_synapses, >=2], got {indices_np.shape}.')

    pre_ids = indices_np[:, 1].astype(np.int64, copy=False)
    invalid = (pre_ids < 0) | (pre_ids >= n_source_neurons)
    if np.any(invalid):
        bad = int(pre_ids[np.flatnonzero(invalid)[0]])
        raise ValueError(
            f'Presynaptic index {bad} is out of bounds for `n_source_neurons={n_source_neurons}`.'
        )
    return pre_ids, n_source_neurons

@njit(cache=True)
def _build_csr_order_numba(pre_ids, n_source_neurons):
    """O(n_syn + n_source) stable bucket build for CSR order/row_splits."""
    n_syn = pre_ids.shape[0]
    counts = np.zeros(n_source_neurons, dtype=np.int64)
    for i in range(n_syn):
        counts[pre_ids[i]] += 1

    row_splits = np.empty(n_source_neurons + 1, dtype=np.int64)
    row_splits[0] = 0
    for i in range(n_source_neurons):
        row_splits[i + 1] = row_splits[i] + counts[i]

    write_ptr = np.empty(n_source_neurons, dtype=np.int64)
    for i in range(n_source_neurons):
        write_ptr[i] = row_splits[i]

    order = np.empty(n_syn, dtype=np.int64)
    for syn_idx in range(n_syn):
        p = pre_ids[syn_idx]
        pos = write_ptr[p]
        order[pos] = syn_idx
        write_ptr[p] = pos + 1

    return order, row_splits


def _csr_index_dtype(n_synapses):
    """Return an index dtype that represents every CSR offset."""
    return tf.int32 if n_synapses <= np.iinfo(np.int32).max else tf.int64


def make_pre_ind_table(indices, n_source_neurons=197613, order_keys=None):
    """
    Build CSR-style row_splits and sorted post indices by presynaptic index.
    Numba provides a linear-time stable bucket build when no within-row ordering
    is requested.
    This function creates a table that maps presynaptic indices to
    the indices of the recurrent_indices tensor using a RaggedTensor.
    This approach ensures that every presynaptic neuron, even those with no
    postsynaptic connections, has an entry in the RaggedTensor.
    """
    pre_ids, n_source_neurons = _validate_inputs(indices, n_source_neurons)

    if pre_ids.size == 0:
        order_np = np.empty((0,), dtype=np.int32)
        row_splits_np = np.zeros((n_source_neurons + 1,), dtype=np.int64)
    elif order_keys is not None:
        # Keep each presynaptic row contiguous while improving within-row locality
        # for gather/segment operations.
        if not isinstance(order_keys, (tuple, list)):
            order_keys = (order_keys,)
        normalized_keys = []
        for key in order_keys:
            key_np = np.asarray(key)
            if key_np.shape[0] != pre_ids.shape[0]:
                raise ValueError(
                    f"`order_keys` entries must have length {pre_ids.shape[0]}, "
                    f"got {key_np.shape[0]}."
                )
            normalized_keys.append(key_np.astype(np.int64, copy=False))
        stable_index = np.arange(pre_ids.shape[0], dtype=np.int64)
        order_np = np.lexsort((stable_index, *reversed(normalized_keys), pre_ids))
        counts_np = np.bincount(pre_ids[order_np], minlength=n_source_neurons)
        row_splits_np = np.empty((n_source_neurons + 1,), dtype=np.int64)
        row_splits_np[0] = 0
        np.cumsum(counts_np, dtype=np.int64, out=row_splits_np[1:])
    else:
        order_np, row_splits_np = _build_csr_order_numba(pre_ids, n_source_neurons)

    index_dtype = _csr_index_dtype(order_np.size)
    order_tf = tf.convert_to_tensor(order_np, dtype=index_dtype)
    row_splits_tf = tf.convert_to_tensor(row_splits_np, dtype=index_dtype)

    return tf.RaggedTensor.from_row_splits(order_tf, row_splits_tf, validate=False)

# def make_pre_ind_table(indices, n_source_neurons=197613):
#     """
#     Build CSR-style row_splits and sorted post indices by presynaptic index.
#     This function creates a table that maps presynaptic indices to
#     the indices of the recurrent_indices tensor using a RaggedTensor.
#     This approach ensures that every presynaptic neuron, even those with no
#     postsynaptic connections, has an entry in the RaggedTensor.
#     """
#     # Extract presynaptic IDs
#     pre_ids = indices[:, 1]  # shape: [num_synapses]
#     # Sort the synapses by presynaptic ID
#     sort_idx = tf.argsort(pre_ids, axis=0)
#     sorted_pre = tf.gather(pre_ids, sort_idx)
#     # Count how many synapses belong to each presynaptic neuron
#     # (We cast to int32 for tf.math.bincount.)
#     counts = tf.math.bincount(tf.cast(sorted_pre, tf.int32), minlength=n_source_neurons)
#     # Build row_splits to define a RaggedTensor from these sorted indices
#     row_splits = tf.concat([[0], tf.cumsum(counts)], axis=0)
#     row_splits = tf.cast(row_splits, dtype=tf.int32) # int32 for efficient indexing in RaggedTensor
#     counts = tf.cast(counts, dtype=tf.int32) # int32 for efficient indexing in RaggedTensor
#     # The values of the RaggedTensor are the original synapse-array row indices,
#     # but sorted by presyn neuron
#     # For efficiency, rt should be in int32 dtype
#     return tf.RaggedTensor.from_row_splits(sort_idx, row_splits, validate=False)

def get_active_synapse_data_table(indices, weights, syn_ids, non_zero_cols, pre_ind_table):
    """Gather active post ids, weights, and synapse ids without materializing full index rows."""
    selected_rows = tf.gather(pre_ind_table, non_zero_cols)
    all_synapse_inds = selected_rows.flat_values
    post_in_degree = selected_rows.row_lengths()  # int64
    active_post_ids = tf.gather(indices, all_synapse_inds)[:, 0]
    active_weights = tf.gather(weights, all_synapse_inds)
    active_syn_ids = tf.gather(syn_ids, all_synapse_inds)

    return active_post_ids, active_weights, active_syn_ids, post_in_degree


def get_active_synapse_metadata_table(indices, syn_ids, non_zero_cols, pre_ind_table):
    """Gather only the metadata needed to rebuild active recurrent indexing in backward."""
    selected_rows = tf.gather(pre_ind_table, non_zero_cols)
    all_synapse_inds = selected_rows.flat_values
    post_in_degree = selected_rows.row_lengths()  # int64
    active_post_ids = tf.gather(indices, all_synapse_inds)[:, 0]
    active_syn_ids = tf.gather(syn_ids, all_synapse_inds)

    return active_post_ids, active_syn_ids, post_in_degree, all_synapse_inds

class SignedConstraint(tf.keras.constraints.Constraint):
    def __init__(self, positive):
        # self._positive = positive
        self.condition = positive

    def __call__(self, w):
        # condition = tf.greater(self._positive, 0)  # yields bool
        sign_corrected_w = tf.where(self.condition, tf.nn.relu(w), -tf.nn.relu(-w))
        return sign_corrected_w


class SparseSignedConstraint(tf.keras.constraints.Constraint):
    def __init__(self, mask, positive):
        self._mask = mask
        self._positive = positive

    def __call__(self, w):
        condition = tf.greater(self._positive, 0)  # yields bool
        sign_corrected_w = tf.where(condition, tf.nn.relu(w), -tf.nn.relu(-w))
        return tf.where(self._mask, sign_corrected_w, tf.zeros_like(sign_corrected_w))


class ClipConstraint(tf.keras.constraints.Constraint):
    def __init__(self, lower_limit, upper_limit):
        self._lower_limit = lower_limit
        self._upper_limit = upper_limit

    def __call__(self, w):
        return tf.clip_by_value(w, self._lower_limit, self._upper_limit)


class V1Column(tf.keras.layers.Layer):
    def __init__(
        self,
        network,
        lgn_input,
        bkg_input,
        dt=1.0,
        gauss_std=0.5,
        dampening_factor=0.3,
        recurrent_dampening_factor=0.5,
        voltage_gradient_dampening=0.5,
        input_weight_scale=1.0,
        recurrent_weight_scale=1.0,
        lr_scale=1.0,
        spike_gradient=False,
        max_delay=0,
        batch_size=1,
        bkg_firing_rate=250,
        pseudo_gauss=False,
        train_recurrent=True,
        train_recurrent_per_type=True,
        train_input=False,
        train_noise=True,
        noise_seed=0,
        hard_reset=False,
        current_input=False,
        synaptic_current_backend="cuda",
        track_voltage_penalty=False,
        voltage_penalty_mode="range",
        return_voltage_sequences=True,
        inference_mode=False,
    ):
        super().__init__()
        # Disable Keras layer autocast so tensors keep explicit dtypes:
        # recurrent state buffers in compute_dtype, selected threshold-critical
        # parameters in variable_dtype.
        self._autocast = False
        _params = dict(network["node_params"])
        # Rescale the voltages to have them near 0, as we wanted the effective step size
        # for the weights to be normalized when learning (weights are scaled similarly)
        voltage_scale = _params["V_th"] - _params["E_L"]
        # voltage_offset = _params["E_L"] # dead write
        # _params["V_th"] = (_params["V_th"] - voltage_offset) / voltage_scale # dead write
        # _params["E_L"] = (_params["E_L"] - voltage_offset) / voltage_scale  # dead write
        # _params["V_reset"] = (_params["V_reset"] - voltage_offset) / voltage_scale # dead write since E_L = V_reset
        _params["asc_amps"] = (_params["asc_amps"] / voltage_scale[..., None])  # _params['asc_amps'] has shape (111, 2)
        # Define the other model variables
        self._node_type_ids = np.array(network["node_type_ids"])
        self._dt = tf.constant(dt, self.compute_dtype)
        self._recurrent_dampening = tf.constant(recurrent_dampening_factor, self.compute_dtype)
        self._dampening_factor = tf.constant(dampening_factor, self.compute_dtype)
        self._voltage_gradient_dampening = tf.constant(
            voltage_gradient_dampening, self.compute_dtype
        )
        self._pseudo_gauss = pseudo_gauss
        self._lr_scale = tf.constant(lr_scale, dtype=self.compute_dtype)
        # self._spike_gradient = spike_gradient
        # Updated by the training loop before each logical forward call.
        # Combined with per-timestep noise_step state to keep recompute_grad deterministic.
        # Runtime RNG state for stateless background sampling.
        # Keep untracked so legacy checkpoints remain loadable with assert_consumed().
        self.noise_seed = tf.Variable(int(noise_seed), trainable=False, dtype=tf.int64, name="noise_seed")
        self.noise_stream = tf.Variable(0, trainable=False, dtype=tf.int64, name="noise_stream")
        self._hard_reset = hard_reset
        self._current_input = current_input
        if voltage_penalty_mode not in ("range", "threshold"):
            raise ValueError("voltage_penalty_mode must be 'range' or 'threshold'.")
        self._track_voltage_penalty = bool(track_voltage_penalty)
        self._voltage_penalty_mode = voltage_penalty_mode
        self._return_voltage_sequences = bool(return_voltage_sequences)
        self._inference_mode = bool(inference_mode)
        if synaptic_current_backend not in ("cuda", "tensorflow"):
            raise ValueError(
                "synaptic_current_backend must be 'cuda' or 'tensorflow', got "
                f"{synaptic_current_backend!r}"
            )
        self.synaptic_current_backend = synaptic_current_backend
        self._n_neurons = int(network["n_nodes"])
        self._gauss_std = tf.constant(gauss_std, self.compute_dtype)
        # Determine the membrane time decay constant
        tau = _params["C_m"] / _params["g"]
        membrane_decay = np.exp(-dt / tau)
        current_factor = (1 - membrane_decay) / _params["g"]

        # Determine the synaptic dynamic parameters for each of the 4 basis receptors.
        path='synaptic_data/tau_basis.npy'
        tau_syns = np.load(path)
        self._n_syn_basis = tau_syns.size
        syn_decay_np = np.exp(-dt / tau_syns)
        syn_decay_np = np.tile(syn_decay_np, self._n_neurons)
        self.syn_decay = tf.constant(syn_decay_np[None, :], dtype=self.compute_dtype) # expand the dimension for processing different receptor types
        psc_initial_np = np.e / tau_syns
        psc_initial_np = np.tile(psc_initial_np, self._n_neurons)
        self.psc_initial = tf.constant(psc_initial_np[None, :], dtype=self.compute_dtype) # expand the dimension for processing different receptor types

        # Determine the maximum delay from the network data
        rec_delays = np.array(network["synapses"]["delays"])
        all_delays = [rec_delays]
        if "delays" in lgn_input:
            all_delays.append(np.array(lgn_input["delays"]))
        if "delays" in bkg_input:
            all_delays.append(np.array(bkg_input["delays"]))
        data_max_delay = int(np.ceil(np.max(np.concatenate(all_delays)) / dt))
        if max_delay > 0:
            self.max_delay = min(data_max_delay, max_delay)
        else:
            self.max_delay = data_max_delay
        print(f"    > max_delay={self.max_delay} ms "
              f"(data max={data_max_delay} ms{f', capped by flag={max_delay}' if max_delay > 0 and max_delay < data_max_delay else ''})")
        # self.batch_size = batch_size # Batch size is now determined by the input tensors in the call() method, not fixed at initialization, to allow for flexible batching during training and inference.

        def _gather(prop):
            return tf.gather(prop, self._node_type_ids)

        def _f(_v, trainable=False, dtype=None):
            if dtype is None:
                dtype = self.compute_dtype
            return tf.Variable(
                tf.cast(_gather(_v), dtype),
                trainable=trainable,
                dtype=dtype,
            )

        def inv_sigmoid(_x):
            return tf.math.log(_x / (1 - _x))

        def custom_val(_v, trainable=False):
            _v = tf.Variable(
                tf.cast(inv_sigmoid(_gather(_v)), self.compute_dtype),
                trainable=trainable,
            )
            def _g():
                return tf.nn.sigmoid(_v.read_value())

            # return _v, _g
            return _g

        # Gather the neuron parameters for every neuron
        # self.t_ref = _f(_params["t_ref"])  # refractory time
        # Store refractory time as integer number of simulation steps.
        # Using ceil preserves the same blocking behavior as float dynamics.
        t_ref_per_neuron = _params["t_ref"][self._node_type_ids]
        t_ref_steps = np.ceil(t_ref_per_neuron / dt).astype(np.int16)
        t_ref_steps = np.maximum(t_ref_steps, 1)
        max_ref_steps = int(np.max(t_ref_steps))
        if max_ref_steps > 127:
            self._refractory_state_dtype = tf.int16
            print(f"Warning: max refractory period is {max_ref_steps} steps, which exceeds int8 capacity. Using int16 for refractory state.")
        else:
            self._refractory_state_dtype = tf.int8
        self.t_ref_steps = tf.constant(t_ref_steps, dtype=self._refractory_state_dtype)

        # Keep threshold-critical membrane parameters in variable_dtype (fp32 under mixed policy)
        # to reduce spike-timing jitter while keeping recurrent state buffers compact.
        self.asc_amps = _f(_params["asc_amps"], trainable=False)
        _k = tf.cast(_params['k'], self.compute_dtype)
        # inverse sigmoid of the adaptation rate constant (1/ms)
        # param_k, param_k_read = custom_val(_k, trainable=False)
        param_k_read = custom_val(_k, trainable=False)
        k = param_k_read()
        self.asc_decay = tf.exp(-self._dt * k)
        self.v_th = tf.constant(1.0, dtype=self.compute_dtype)
        self.v_reset = tf.constant(0.0, dtype=self.compute_dtype)
        # After per-type normalization:
        # - E_L is exactly 0
        # V_th is exactly 1
        # - V_th - E_L is exactly 1
        # E_L - V_th is exactly -1
        # Keep only the threshold offset needed for spike generation.
        # e_l = _f(_params["E_L"])
        # self.normalizer = self.v_th - e_l
        # param_g = _f(_params["g"])
        # self.gathered_g = param_g * e_l
        # self.v_th = _f(_params["V_th"], dtype=self.compute_dtype)
        # self.v_reset = _f(_params["V_reset"], dtype=self.compute_dtype)
        # self.v_gap = self.v_reset - self.v_th

        self.decay = _f(membrane_decay, dtype=self.compute_dtype)
        self.current_factor = _f(current_factor, dtype=self.compute_dtype)
        # self.voltage_scale = _f(voltage_scale)
        # self.voltage_offset = _f(voltage_offset)

        # Find the synaptic basis representation for each synaptic type
        # path = os.path.join('GLIF_network', 'syn_id_to_syn_weights_dict.pkl')
        path = os.path.join(network["data_dir"], 'tf_data', 'syn_id_to_syn_weights_dict.pkl')
        with open(path, "rb") as f:
            syn_id_to_syn_weights_dict = pkl.load(f)
        synaptic_basis_weights = np.array(list(syn_id_to_syn_weights_dict.values()))
        # self.synaptic_basis_weights = tf.constant(synaptic_basis_weights, dtype=self.compute_dtype)
        self.synaptic_basis_weights = tf.constant(synaptic_basis_weights, dtype=self.compute_dtype)

        ### Network recurrent connectivity ###
        indices = np.array(network["synapses"]["indices"])
        weights = np.array(network["synapses"]["weights"])
        dense_shape = np.array(network["synapses"]["dense_shape"])
        syn_ids = np.array(network["synapses"]["syn_ids"])
        delays = np.array(network["synapses"]["delays"])
        # Scale down the recurrent weights
        weights = (weights/voltage_scale[self._node_type_ids[indices[:, 0]]])
        # Use the maximum delay to clip the synaptic delays
        delays = np.round(np.clip(delays, dt, self.max_delay)/dt).astype(np.int32)
        # Introduce the delays in the presynaptic neuron indices
        indices[:, 1] = indices[:, 1] + self._n_neurons * (delays - 1)
        self.recurrent_dense_shape = dense_shape[0], self.max_delay * dense_shape[1]
        #the first column (presynaptic neuron) has size n_neurons and the second column (postsynaptic neuron) has size max_delay*n_neurons
        if self.synaptic_current_backend == "cuda":
            self.recurrent_csr = build_csr_connectivity(
                indices,
                syn_ids,
                n_pre=self.recurrent_dense_shape[1],
                n_post=self.recurrent_dense_shape[0],
            )
        else:
            self.recurrent_indices = tf.Variable(
                indices, dtype=tf.int64, trainable=False
            )
            self.pre_ind_table = make_pre_ind_table(
                indices,
                n_source_neurons=self.recurrent_dense_shape[1],
                order_keys=(indices[:, 0], syn_ids),
            )
        # add dimension for the weights factors - TensorShape([23525415, 1])
        # weights = tf.expand_dims(weights, axis=1)
        # Set the sign of the connections (exc or inh)
        # recurrent_weight_positive = tf.Variable(
        #     weights >= 0.0, name="recurrent_weights_sign", trainable=False)
        # recurrent_weight_positive = tf.constant(weights >= 0, dtype=tf.int8)
        recurrent_weight_positive = tf.constant(weights >= 0, dtype=tf.bool)

        # if training the recurrent connection per type, turn off recurrent training
        # of individual connections
        if train_recurrent:
            if train_recurrent_per_type:
                individual_training = False
            else:
                individual_training = True
        else:
            individual_training = False

        # Scale the weights
        self.recurrent_weight_values = tf.Variable(
            weights * recurrent_weight_scale / lr_scale,
            name="sparse_recurrent_weights",
            constraint=SignedConstraint(recurrent_weight_positive),
            trainable=individual_training,
            dtype=self.variable_dtype
        ) # shape = (n_synapses,)

        # # prepare per_type variable, if required
        # if per_type_training:
        #     self.per_type_training = True
        #     self.connection_type_ids = other_v1_utils.connection_type_ids(network)
        #     max_id = np.max(self.connection_type_ids) + 1
        #     # prepare a variable and gather with type ids.
        #     self.recurrent_per_type_weight_values = tf.Variable(
        #         tf.ones(max_id),
        #         name="recurrent_per_type_weights",
        #         constraint=ClipConstraint(0.2, 5.0),
        #         trainable=True,
        #         dtype=self.variable_dtype
        #     ) # shape = (n_connection_types (21 * 21))
        #     # multiply this to the weights (this needs to be done in the loop)
        # else:
        #     self.per_type_training = False

        if self.synaptic_current_backend == "tensorflow":
            self.syn_ids = tf.constant(syn_ids, dtype=tf.int64)
        # self.recurrent_weights_factors = tf.gather(self.synaptic_basis_weights, self.syn_ids, axis=0) # TensorShape([23525415, 5])
        print(f"    > # Recurrent synapses: {len(indices)}")

        del indices, weights, dense_shape, delays, syn_ids, recurrent_weight_positive

        ### LGN input connectivity ###
        self.input_dim = lgn_input["n_inputs"]
        self.lgn_input_dense_shape = (self._n_neurons, self.input_dim,)
        input_indices = np.array(lgn_input["indices"])
        input_weights = np.array(lgn_input["weights"])
        input_syn_ids = np.array(lgn_input["syn_ids"])
        # Scale down the input weights
        input_weights = (input_weights/ voltage_scale[self._node_type_ids[input_indices[:, 0]]])
        # # Introduce the delays in the postsynaptic neuron indices
        # input_delays = np.array(lgn_input["delays"])
        # input_delays = np.round(np.clip(input_delays, dt, self.max_delay)/dt).astype(np.int32)
        # input_indices[:, 1] = input_indices[:, 1] + self._n_neurons * (input_delays - 1)
        if self.synaptic_current_backend == "cuda":
            # Retain the legacy checkpoint object without occupying GPU memory;
            # CUDA execution consumes the compact CSR tensors below instead.
            with tf.device("/CPU:0"):
                self.input_indices = tf.Variable(
                    input_indices, trainable=False, dtype=tf.int64
                )
        else:
            self.input_indices = tf.Variable(
                input_indices, trainable=False, dtype=tf.int64
            )

        # Define the Tensorflow variables
        # input_weight_positive = tf.Variable(
        #     input_weights >= 0.0, name="input_weights_sign", trainable=False)
        # input_weight_positive = tf.constant(input_weights >= 0, dtype=tf.int8)
        input_weight_positive = tf.constant(input_weights >= 0, dtype=tf.bool)
        self.input_weight_values = tf.Variable(
            input_weights * input_weight_scale / lr_scale,
            name="sparse_input_weights",
            constraint=SignedConstraint(input_weight_positive),
            trainable=train_input,
            dtype=self.variable_dtype
        )
        if self.synaptic_current_backend == "cuda" and not self._current_input:
            self.input_csr = build_external_csr_connectivity(
                input_indices,
                input_syn_ids,
                n_pre=self.lgn_input_dense_shape[1],
                n_post=self.lgn_input_dense_shape[0],
            )
        else:
            self.input_syn_ids = tf.constant(input_syn_ids, dtype=tf.int64)
        if self.synaptic_current_backend == "tensorflow" and not self._current_input:
            self.pre_input_ind_table = make_pre_ind_table(input_indices, n_source_neurons=self.lgn_input_dense_shape[1])

        print(f"    > # LGN input synapses {len(input_indices)}")
        del input_indices, input_weights, input_syn_ids, input_weight_positive #, input_delays

        ### BKG input connectivity ###
        self.bkg_spike_prob = tf.constant(bkg_firing_rate * 0.001, dtype=self.compute_dtype)
        self.bkg_input_dense_shape = (self._n_neurons, bkg_input["n_inputs"],)
        bkg_input_indices = np.array(bkg_input['indices'])
        bkg_input_weights = np.array(bkg_input['weights'])
        bkg_input_syn_ids = np.array(bkg_input['syn_ids'])
        # Scale down the background input weights
        bkg_input_weights = (bkg_input_weights/voltage_scale[self._node_type_ids[bkg_input_indices[:, 0]]])
        # # Introduce the delays in the postsynaptic neuron indices
        # bkg_input_delays = np.array(bkg_input['delays'])
        # bkg_input_delays = np.round(np.clip(bkg_input_delays, dt, self.max_delay)/dt).astype(np.int32)
        # bkg_input_indices[:, 1] = bkg_input_indices[:, 1] + self._n_neurons * (bkg_input_delays - 1)
        if self.synaptic_current_backend == "cuda":
            with tf.device("/CPU:0"):
                self.bkg_input_indices = tf.Variable(
                    bkg_input_indices, trainable=False, dtype=tf.int64
                )
        else:
            self.bkg_input_indices = tf.Variable(
                bkg_input_indices, trainable=False, dtype=tf.int64
            )
        # self.bkg_input_indices = tf.Variable(bkg_input_indices, trainable=False, dtype=tf.int32)
        if self.synaptic_current_backend == "cuda":
            self.bkg_input_csr = build_external_csr_connectivity(
                bkg_input_indices,
                bkg_input_syn_ids,
                n_pre=self.bkg_input_dense_shape[1],
                n_post=self.bkg_input_dense_shape[0],
            )
        else:
            self.pre_bkg_ind_table = make_pre_ind_table(
                bkg_input_indices,
                n_source_neurons=self.bkg_input_dense_shape[1],
            )

        # Define Tensorflow variables
        # bkg_input_weight_positive = tf.Variable(
        #     bkg_input_weights >= 0.0, name="bkg_input_weights_sign", trainable=False)
        # bkg_input_weight_positive = tf.constant(bkg_input_weights >= 0, dtype=tf.int8)
        bkg_input_weight_positive = tf.constant(bkg_input_weights >= 0, dtype=tf.bool)
        self.bkg_input_weights = tf.Variable(
            bkg_input_weights * input_weight_scale / lr_scale,
            name="rest_of_brain_weights",
            constraint=SignedConstraint(bkg_input_weight_positive),
            trainable=train_noise,
            dtype=self.variable_dtype
        )

        if self.synaptic_current_backend == "tensorflow":
            self.bkg_input_syn_ids = tf.constant(bkg_input_syn_ids, dtype=tf.int64)
        # self.bkg_input_weights_factors = tf.gather(self.synaptic_basis_weights, bkg_input_syn_ids, axis=0)

        print(f"    > # BKG input synapses {len(bkg_input_indices)}")
        del bkg_input_indices, bkg_input_weights, bkg_input_syn_ids, bkg_input_weight_positive #, bkg_input_delays

    def calculate_input_current_from_firing_probabilities(self, x_t):
        """
        Calculate the input current to the LGN neurons from the input layer.
        """
        batch_size = tf.shape(x_t)[0]
        # This function performs the tensor multiplication to calculate the recurrent currents at each timestep
        # Compute in float32 — sparse_dense_matmul lacks optimized float16 GPU kernels.
        # Cast output once at the end instead.
        # if x_t.dtype != self.variable_dtype:
        #     x_t = tf.cast(x_t, dtype=self.variable_dtype)

        i_in = tf.TensorArray(dtype=self.variable_dtype, size=self._n_syn_basis)
        for r_id in range(self._n_syn_basis):
            input_weights_factors = tf.gather(self.synaptic_basis_weights[:, r_id], self.input_syn_ids, axis=0) # shape (n_input_synapses,)
            weights_syn_receptors = tf.cast(self.input_weight_values, self.compute_dtype) * input_weights_factors
            sparse_w_in = tf.sparse.SparseTensor(
                self.input_indices,
                weights_syn_receptors,
                self.lgn_input_dense_shape
            )
            i_receptor = tf.sparse.sparse_dense_matmul(
                                                sparse_w_in,
                                                x_t,
                                                adjoint_b=True
                                                )
            # Append i_receptor to the TensorArray
            i_in = i_in.write(r_id, i_receptor)
        # Stack the TensorArray into a single tensor
        i_in = i_in.stack()
        # flat the output
        i_in = tf.transpose(i_in)
        i_in_flat = tf.reshape(i_in, [batch_size * self._n_neurons, self._n_syn_basis])

        # Cast output to compute_dtype
        # if i_in_flat.dtype != self.compute_dtype:
        #     i_in_flat = tf.cast(i_in_flat, dtype=self.compute_dtype)

        return i_in_flat

    def calculate_input_current_from_spikes(self, x_t):
        """
        Calculate the input current from the LGN neurons, given the spikes at time t (x_t).
        Use int64 for indexing to optimize GPU gather and segment_sum performance, which are critical operations in this function.
        """
        if self.synaptic_current_backend == "cuda":
            active_inputs = x_t if x_t.dtype == tf.bool else x_t > 0
            activity = tf.stop_gradient(
                tf.cast(active_inputs, dtype=self.compute_dtype)
            )
            return calculate_external_currents(
                activity,
                self.input_weight_values,
                self.synaptic_basis_weights,
                self.input_csr,
                compute_activity_gradient=False,
            )

        # x_t: Shape [batch_size, input_dim]
        batch_size = tf.cast(tf.shape(x_t)[0], dtype=tf.int64) # int64
        n_post_neurons = self.lgn_input_dense_shape[0] #int64
        # Find the indices of non-zero inputs
        non_zero_indices = tf.where(x_t > 0) if x_t.dtype != tf.bool else tf.where(x_t)  # shape (n_active_inputs, 2)

        batch_indices = non_zero_indices[:, 0]  # int64
        pre_neuron_indices = non_zero_indices[:, 1] #int64

        # Get the indices into self.recurrent_indices for each pre_neuron_index
        # self.pre_ind_table is a RaggedTensor or a list of lists mapping pre_neuron_index to indices in recurrent_indices
        post_neuron_indices, new_weights, new_syn_ids, post_in_degree = get_active_synapse_data_table(
            self.input_indices,
            self.input_weight_values,
            self.input_syn_ids,
            pre_neuron_indices,
            self.pre_input_ind_table
        )

        # Expand batch_indices to match the length of inds_flat
        batch_indices_per_connection = tf.repeat(batch_indices, post_in_degree) #int64
        # Compute segment IDs
        segment_ids = batch_indices_per_connection * n_post_neurons + post_neuron_indices
        segment_ids = tf.cast(segment_ids, dtype=tf.int32)
        num_segments = tf.cast(batch_size * n_post_neurons, dtype=tf.int32)
        # Compute weighted basis factors
        basis_factors = tf.gather(self.synaptic_basis_weights, new_syn_ids, axis=0)  # shape (n_active_connections, n_syn_basis)
        # Accumulate currents in variable_dtype for better numerical fidelity, then cast once.
        new_weights = tf.cast(new_weights, self.compute_dtype)
        new_weights_final = new_weights[:, tf.newaxis] * basis_factors
        # new_weights_final = new_weights[:, tf.newaxis] * basis_factors

        # Calculate input currents
        i_in_flat = tf.math.unsorted_segment_sum(new_weights_final, segment_ids, num_segments)

        # Cast back to compute_dtype to keep downstream state buffers compact.
        # if i_in_flat.dtype != self.compute_dtype:
        #     i_in_flat = tf.cast(i_in_flat, dtype=self.compute_dtype)

        return i_in_flat

    def calculate_noise_current(self, batch_size, noise_step):
        n_post_neurons = self.bkg_input_dense_shape[0]
        step_seed = tf.cast(noise_step[0], tf.int32)
        base_seed = tf.cast(self.noise_seed, tf.int32)
        replica_context = tf.distribute.get_replica_context()
        if replica_context is None:
            replica_id = tf.constant(0, dtype=tf.int32)
        else:
            replica_id = tf.cast(replica_context.replica_id_in_sync_group, tf.int32)
        noise_seed = tf.stack(
            [base_seed + replica_id * tf.constant(1000003, dtype=tf.int32), step_seed],
            axis=0,
        )
        poisson_shape = tf.stack(
            [tf.cast(batch_size, tf.int32), tf.cast(self.bkg_input_dense_shape[1], tf.int32)],
            axis=0,
        )
        rest_of_brain = tf.random.stateless_poisson(
            shape=poisson_shape,
            seed=noise_seed,
            lam=self.bkg_spike_prob,
            dtype=tf.int32,
        )

        if self.synaptic_current_backend == "cuda":
            activity = tf.stop_gradient(
                tf.cast(rest_of_brain, dtype=self.compute_dtype)
            )
            return calculate_external_currents(
                activity,
                self.bkg_input_weights,
                self.synaptic_basis_weights,
                self.bkg_input_csr,
                compute_activity_gradient=False,
            )

        # Keep noise indexing in int64
        non_zero_indices = tf.where(rest_of_brain > 0)

        batch_indices = non_zero_indices[:, 0] #int64
        pre_neuron_indices = non_zero_indices[:, 1] #int64
        # Get the indices into self.recurrent_indices for each pre_neuron_index
        # self.pre_ind_table is a RaggedTensor or a list of lists mapping pre_neuron_index to indices in recurrent_indices
        post_neuron_indices, new_weights, new_syn_ids, post_in_degree = get_active_synapse_data_table(
            self.bkg_input_indices,
            self.bkg_input_weights,
            self.bkg_input_syn_ids,
            pre_neuron_indices,
            self.pre_bkg_ind_table
        )
        # Expand batch_indices to match the length of inds_flat
        batch_indices_per_connection = tf.repeat(batch_indices, post_in_degree)
        # Compute segment IDs
        segment_ids = batch_indices_per_connection * n_post_neurons + post_neuron_indices
        segment_ids = tf.cast(segment_ids, dtype=tf.int32)
        num_segments = tf.cast(batch_size * n_post_neurons, dtype=tf.int32)

        # # Alternative (slower): Gather spike counts once per active input and broadcast by connection degree.
        # # Keeping it here only as reference.
        # active_spike_counts = tf.gather_nd(rest_of_brain, non_zero_indices)
        # n_pre_spikes = tf.cast(tf.repeat(active_spike_counts, post_in_degree), dtype=self.variable_dtype)

        presynaptic_indices = tf.stack(
            [batch_indices_per_connection, tf.repeat(pre_neuron_indices, post_in_degree)],
            axis=1
        ) # gather works better with int64
        n_pre_spikes = tf.cast(tf.gather_nd(rest_of_brain, presynaptic_indices), dtype=self.compute_dtype)
        # n_pre_spikes = tf.cast(tf.gather_nd(rest_of_brain, presynaptic_indices), dtype=self.compute_dtype)

        # Compute weighted basis factors
        basis_factors = tf.gather(self.synaptic_basis_weights, new_syn_ids, axis=0)
        # Accumulate currents in variable_dtype for better numerical fidelity, then cast once.
        new_weights = tf.cast(new_weights, dtype=self.compute_dtype) # shape (n_active_connections,)
        # new_weights = tf.cast(new_weights * n_pre_spikes, self.compute_dtype)
        # new_weights_final = new_weights[:, tf.newaxis] * basis_factors
        # new_weights_final = tf.cast(new_weights * n_pre_spikes, dtype=self.compute_dtype)
        new_weights_final = (new_weights * n_pre_spikes)[:, tf.newaxis] * basis_factors

        # Calculate input currents
        i_in_flat = tf.math.unsorted_segment_sum(new_weights_final, segment_ids, num_segments)

        # # Cast back to compute_dtype to keep downstream state buffers compact.
        # if i_in_flat.dtype != self.compute_dtype:
        #     i_in_flat = tf.cast(i_in_flat, dtype=self.compute_dtype)

        return i_in_flat

    def calculate_i_rec_with_custom_grad(self, rec_z_buf):

        # Calculate recurrent currents with the selected fused CUDA or reference
        # TensorFlow implementation.
        if self.synaptic_current_backend == "cuda":
            i_rec_flat = calculate_synaptic_currents_cuda(
                rec_z_buf,
                self.recurrent_weight_values,
                self.synaptic_basis_weights,
                self._recurrent_dampening,
                self.recurrent_csr,
            )
        else:
            i_rec_flat = calculate_synaptic_currents(
                rec_z_buf,
                self.recurrent_indices,
                self.recurrent_weight_values,
                tf.cast(self.recurrent_weight_values, self.compute_dtype),
                self.recurrent_dense_shape,
                self.synaptic_basis_weights,
                self.syn_ids,
                self.pre_ind_table,
                self._recurrent_dampening,
            )

        # Cast back to compute_dtype to keep downstream state buffers compact.
        # if i_rec_flat.dtype != self.compute_dtype:
        #     i_rec_flat = tf.cast(i_rec_flat, dtype=self.compute_dtype)

        return i_rec_flat

    # @tf.function(jit_compile=True) does not work here because it breaks the graph structure
    def update_psc(self, psc, psc_rise, rec_inputs):
        new_psc_rise = psc_rise * self.syn_decay + rec_inputs * self.psc_initial
        new_psc = psc * self.syn_decay + self._dt * self.syn_decay * psc_rise
        return new_psc, new_psc_rise

    # @tf.function(jit_compile=True)  does not work here because it breaks the graph structure
    def _dense_update_impl(self, batch_size, prev_z, v, r, asc, psc_rise, psc, rec_inputs):
        """
        Compute the dense update of the neuron states for one timestep.
        """
        # Calculate the new psc variables
        new_psc, new_psc_rise = self.update_psc(psc, psc_rise, rec_inputs)
        # Calculate the ASC variables
        asc = tf.reshape(asc, (batch_size, self._n_neurons, 2))
        # new_asc = self.asc_decay * asc + tf.expand_dims(prev_z, axis=-1) * self.asc_amps
        new_asc = self.asc_decay * asc + tf.expand_dims(tf.stop_gradient(prev_z), axis=-1) * self.asc_amps
        new_asc = tf.reshape(new_asc, (batch_size, self._n_neurons * 2))
        # Calculate the postsynaptic current
        input_current = tf.reshape(psc, (batch_size, self._n_neurons, self._n_syn_basis))
        input_current = tf.reduce_sum(input_current, -1)
        # Add all the postsynaptic current sources
        c1 = input_current + tf.reduce_sum(asc, axis=-1) # + self.gathered_g
        # Compute membrane update in variable_dtype (fp32 under mixed policy) for
        # more stable threshold crossings, then store state in compute_dtype.
        # decayed_v = self.decay * v
        # reset_current = prev_z * self.v_gap
        # new_v = decayed_v + self.current_factor * c1 + reset_current
        # new_v = self.decay * v + self.current_factor * c1 - prev_z
        # new_v = self.decay * v + self.current_factor * c1 - tf.stop_gradient(prev_z)
        # Damp only the voltage self-loop. We intentionally leave
        # current_factor * c1 untouched so recurrent/input pathways keep full credit.
        dampened_v = straight_through_dampen(v, self._voltage_gradient_dampening)
        new_v = self.decay * dampened_v + self.current_factor * c1 - tf.stop_gradient(prev_z)
        # new_v = self.decay * dampened_v + self.current_factor * c1 - prev_z
        # Update the voltage according to the LIF equation and the refractory period
        # New r is a variable that accounts for the refractory period in which a neuron cannot spike
        # prev_spike = tf.cast(prev_z > 0, self._refractory_state_dtype)
        prev_spike = tf.cast(prev_z, dtype=self._refractory_state_dtype)
        # new_r = tf.maximum(r + prev_spike * self.t_ref_steps - 1, 0)
        new_r = tf.stop_gradient(tf.maximum(r + prev_spike * self.t_ref_steps - 1, 0)) # prevent gradients from flowing through the refractory state

        if self._hard_reset:
            # Here we keep the voltage at the reset value during the refractory period
            refractory_mask = tf.greater(new_r, 0)
            new_v = tf.where(refractory_mask, self.v_reset, new_v)
            # Here we make a hard reset and let the voltage freely evolve but we do not let the
            # neuron spike during the refractory period

        # new_v_state = tf.cast(new_v, self.compute_dtype)
        return new_v, new_r, new_asc, new_psc_rise, new_psc

    @property
    def state_size(self):
        # Define the state size of the network
        state_size = (
            self._n_neurons * self.max_delay,  # z buffer
            self._n_neurons,  # v
            self._n_neurons,  # r
            self._n_neurons * 2,  # asc
            self._n_neurons * self._n_syn_basis,  # psc rise
            self._n_neurons * self._n_syn_basis,  # psc
            tf.TensorShape([]),  # noise timestep counter
        )
        if self._track_voltage_penalty:
            state_size += (tf.TensorShape([]),)
        return state_size

    def zero_state(self, batch_size, dtype=tf.float32):
        # Keep recurrent state buffers in compute_dtype to control VRAM.
        # The neurons membrane voltage start the simulation at their reset value
        z0_buf = tf.zeros((batch_size, self._n_neurons * self.max_delay), self.compute_dtype)
        v0 = tf.zeros((batch_size, self._n_neurons), self.compute_dtype)
        # v0 = tf.ones((batch_size, self._n_neurons), self.compute_dtype) * self.v_reset
        r0 = tf.zeros((batch_size, self._n_neurons), self._refractory_state_dtype)
        asc = tf.zeros((batch_size, self._n_neurons * 2), self.compute_dtype)
        psc_rise0 = tf.zeros((batch_size, self._n_neurons * self._n_syn_basis), self.compute_dtype)
        psc0 = tf.zeros((batch_size, self._n_neurons * self._n_syn_basis), self.compute_dtype)
        noise_step0 = tf.zeros((batch_size,), tf.int32)

        state = (z0_buf, v0, r0, asc, psc_rise0, psc0, noise_step0)
        if self._track_voltage_penalty:
            state += (tf.zeros((batch_size,), self.compute_dtype),)
        return state

    def _voltage_penalty_mean_step(self, voltage):
        return voltage_penalty_mean_step(
            voltage, self._n_neurons, self._voltage_penalty_mode
        )

    # @tf.function # dont use it in here because it breaks the graph structure and the custom gradients
    def call(self, inputs, state, constants=None):

        # Get all the model inputs
        # external_current = inputs[:, :self._n_neurons*self._n_syn_basis] # external inputs shape (1, 399804)
        # bkg_noise = inputs[:, self._n_neurons*self._n_syn_basis:-self._n_neurons]
        lgn_input = inputs[:, :self.input_dim]

        batch_size = tf.cast(tf.shape(inputs)[0], dtype=tf.int64)

        # Extract the network variables from the state
        z_buf, v, r, asc, psc_rise, psc, noise_step = state[:7]
        voltage_penalty = state[7] if self._track_voltage_penalty else None
        # Get previous spikes
        prev_z = z_buf[:, :self._n_neurons] # Shape: [batch_size, n_neurons]

        # Calculate the recurrent postsynaptic currents
        i_rec = self.calculate_i_rec_with_custom_grad(z_buf)
        # Calculate the postsynaptic current from the external input
        if self._current_input:
            external_current = self.calculate_input_current_from_firing_probabilities(lgn_input)
        else:
            external_current = self.calculate_input_current_from_spikes(lgn_input)

        i_noise = self.calculate_noise_current(batch_size, noise_step)
        # Add all the current sources
        rec_inputs = i_rec + external_current + i_noise
        # Reshape i_rec_flat back to [batch_size, num_neurons]
        rec_inputs = tf.reshape(rec_inputs, [batch_size, self._n_neurons * self._n_syn_basis])
        # Scale with the learning rate
        rec_inputs = rec_inputs * self._lr_scale

        new_v, new_r, new_asc, new_psc_rise, new_psc = self._dense_update_impl(
            batch_size, prev_z, v, r, asc, psc_rise, psc, rec_inputs
        )

        # Generate spikes from a high-fidelity membrane lane before state quantization.
        # v_sc = (new_v - self.v_th) / self.normalizer # normalized is 1 for scaled voltage
        v_sc = new_v - self.v_th
        if self._pseudo_gauss:
            new_z = spike_gauss(v_sc, self._gauss_std, self._dampening_factor)
        else:
            new_z = spike_function(v_sc, self._dampening_factor)

        # Generate the new spikes if the refractory period is concluded
        # new_z = tf.cast(new_z, self.compute_dtype)
        refractory_active = tf.greater(new_r, 0)
        new_z = tf.where(refractory_active, tf.zeros_like(new_z), new_z)

        # Add current spikes to the buffer
        new_z_buf = tf.concat([new_z, z_buf[:, :-self._n_neurons]], axis=1)  # Shift buffer

        # Keep model outputs in compute_dtype for mixed-precision efficiency.
        # output_v = new_v
        # if output_v.dtype != self.compute_dtype:
        #     output_v = tf.cast(output_v, self.compute_dtype)

        # Define the model outputs and the new state of the network
        # Training keeps floating spikes for surrogate gradients. Inference can expose
        # the same binary values as uint8 without changing the floating delay state.
        visible_z = tf.cast(new_z, tf.uint8) if self._inference_mode else new_z
        outputs = (
            (visible_z, new_v)
            if self._return_voltage_sequences
            else (visible_z,)
        )
        new_noise_step = noise_step + 1
        new_state = (new_z_buf, new_v, new_r, new_asc, new_psc_rise, new_psc, new_noise_step)
        if self._track_voltage_penalty:
            new_state += (voltage_penalty + self._voltage_penalty_mean_step(new_v),)

        return outputs, new_state


class ClassificationReadoutLayer(tf.keras.layers.Layer):
    """
    Readout layer for classification from spiking neural network outputs.

    This layer:
    1. Gathers spikes from designated readout neurons for each class
    2. Applies dampening scale correction for gradient flow
    3. Applies a learnable output scale
    4. Temporally bins and averages the outputs
    5. Applies softmax for classification probabilities
    """
    def __init__(self, readout_neuron_ids, n_output, dampening_factor,
                 seq_len, down_sample, **kwargs):
        super().__init__(**kwargs)
        self.readout_neuron_ids = readout_neuron_ids  # List of neuron id arrays per class
        self.n_output = n_output
        self.dampening_factor = dampening_factor
        self.seq_len = seq_len
        self.down_sample = down_sample

    def build(self, input_shape):
        self.output_scale = self.add_weight(
            name='output_scale',
            shape=(),
            initializer='zeros',
            trainable=True,
            dtype=tf.float32
        )
        super().build(input_shape)

    def call(self, spikes):
        damp_scale = tf.cast(1.0 / self.dampening_factor, spikes.dtype)
        # scale = 1 + tf.nn.softplus(self.output_scale)
        scale = tf.cast(1 + tf.nn.softplus(self.output_scale), spikes.dtype)

        class_outputs = []
        for i in range(self.n_output):
            # Gather spikes from readout neurons for this class
            t_spikes = tf.gather(spikes, self.readout_neuron_ids[i], axis=2)
            # Apply dampening correction for proper gradient scaling
            t_spikes = damp_scale * t_spikes + (1 - damp_scale) * tf.stop_gradient(t_spikes)
            # t_spikes = tf.cast(t_spikes, tf.float32)
            # Average over readout neurons
            t_output = tf.reduce_mean(t_spikes, axis=-1)
            class_outputs.append(t_output)

        # Concatenate class outputs and apply scale: [batch, time, n_output]
        output = tf.concat(class_outputs, axis=-1) * scale

        # Temporal binning: reshape to [batch, n_bins, bin_size, n_output]
        mean_output = tf.reshape(
            output,
            (-1, int(self.seq_len / self.down_sample), self.down_sample, self.n_output)
        )
        # Average within each bin: [batch, n_bins, n_output]
        mean_output = tf.reduce_mean(mean_output, axis=2)
        # Apply softmax for classification probabilities
        mean_output = tf.nn.softmax(mean_output, axis=-1)
        mean_output = tf.cast(mean_output, dtype=tf.float32)

        return mean_output

    def get_config(self):
        config = super().get_config()
        config.update({
            'readout_neuron_ids': self.readout_neuron_ids,
            'n_output': self.n_output,
            'dampening_factor': self.dampening_factor,
            'seq_len': self.seq_len,
            'down_sample': self.down_sample,
        })
        return config


# @profile
def create_model(
    network,
    lgn_input,
    bkg_input,
    seq_len=100,
    n_input=10,
    n_output=2,
    dtype=tf.float32,
    input_weight_scale=1.0,
    gauss_std=0.5,
    dampening_factor=0.2,
    recurrent_dampening_factor=0.5,
    voltage_gradient_dampening=0.5,
    lr_scale=800.0,
    train_recurrent=True,
    train_recurrent_per_type=False,
    train_input=True,
    train_noise=True,
    neuron_output=False,
    use_state_input=False,
    return_state=False,
    return_sequences=False,
    down_sample=50,
    cue_duration=20,
    add_metric=True,
    max_delay=0,
    batch_size=None,
    pseudo_gauss=False,
    hard_reset=False,
    current_input=False,
    use_dummy_state_input=False,
    seed=42,
    synaptic_current_backend="cuda",
    track_voltage_penalty=False,
    voltage_penalty_mode="range",
    return_voltage_sequences=True,
    inference_mode=False,
):

    # Create the input layer of the model
    x = tf.keras.layers.Input(shape=(None, n_input,), dtype=dtype)
    neurons = network["n_nodes"]
    full_inputs = x

    # Optional dummy input for legacy voltage-gradient experiments.
    if use_dummy_state_input:
        state_input_holder = tf.keras.layers.Input(shape=(None, neurons), dtype=dtype)
        full_inputs = tf.concat((full_inputs, state_input_holder), -1)
    else:
        state_input_holder = None

    # If batch_size is not provided as an argument, it is automatically inferred from the
    # first dimension of x using tf.shape().
    if batch_size is None:
        batch_size = tf.shape(x)[0]
    else:
        batch_size = batch_size

    # Create the V1Column cell
    print('Creating the V1 column...')
    # time0 = time()
    cell = V1Column(
        network,
        lgn_input,
        bkg_input,
        gauss_std=gauss_std,
        dampening_factor=dampening_factor,
        input_weight_scale=input_weight_scale,
        lr_scale=lr_scale,
        spike_gradient=True,
        recurrent_dampening_factor=recurrent_dampening_factor,
        voltage_gradient_dampening=voltage_gradient_dampening,
        max_delay=max_delay,
        pseudo_gauss=pseudo_gauss,
        batch_size=batch_size,
        train_recurrent=train_recurrent,
        train_recurrent_per_type=train_recurrent_per_type,
        train_input=train_input,
        train_noise=train_noise,
        noise_seed=seed,
        hard_reset=hard_reset,
        current_input=current_input,
        synaptic_current_backend=synaptic_current_backend,
        track_voltage_penalty=track_voltage_penalty,
        voltage_penalty_mode=voltage_penalty_mode,
        return_voltage_sequences=return_voltage_sequences,
        inference_mode=inference_mode,
    )

    # initialize the RNN state to zero using the zero_state() method of the V1Column class.
    zero_state = cell.zero_state(batch_size, dtype)

    if use_state_input:
        # The shape of each input tensor matches the shape of the corresponding
        # tensor in the zero_state tuple, except for the batch dimension. The batch
        # dimension is left unspecified, allowing the tensor to be fed variable-sized
        # batches of data.
        initial_state_holder = tuple(
            tf.keras.layers.Input(shape=s.shape[1:], dtype=s.dtype)
            for s in zero_state
        )
        # The code then copies the input tensors into the rnn_initial_state variable
        # using tf.nest.map_structure(). This creates a nested structure of tensors with
        # the same shape as the original zero_state structure.
        rnn_initial_state = tf.nest.map_structure(tf.identity, initial_state_holder)
    else:
        rnn_initial_state = zero_state
        initial_state_holder = None

    # Create the RNN layer of the model using the V1Column cell
    # The RNN layer returns the output of the RNN layer and the final state of the RNN
    # layer. The output of the RNN layer is a tensor of shape (batch_size, seq_len,
    # neurons). The final state of the RNN layer is a tuple of tensors, each of shape
    # (batch_size, neurons).
    rnn = tf.keras.layers.RNN(cell, return_sequences=True, return_state=return_state, name="rsnn")
    # Keep provided state dtypes (notably fp32 v/r) instead of autocasting to compute_dtype.
    rnn._autocast = False

    # Apply the rnn layer to the full_inputs tensor
    rsnn_out = rnn(full_inputs, initial_state=rnn_initial_state)

    # Check if the return_state argument is True or False and assign the output of the
    # RNN layer to the hidden variable accordingly.
    if return_state:
        hidden = rsnn_out[0]
        # new_state = out[1:]
    else:
        hidden = rsnn_out

    spikes_dict = {}
    spikes_dict['v1'] = (
        tf.cast(hidden[0], dtype) if inference_mode else hidden[0]
    )
    # voltage = hidden[1]

    outputs_dict = {}
    for area in ['v1']: #['v1', 'lm']:
        if neuron_output:
            # Collect readout neuron ids for each output class
            # check if readout_neuron_ids_{i} exist in network dict
            if f'readout_neuron_ids_{0}' in network:
                readout_ids = [network[f'readout_neuron_ids_{i}'] for i in range(n_output)]
            else:
                readout_ids = [network["readout_neuron_ids"][i] for i in range(n_output)]

            # Create and apply the classification readout layer
            readout_layer = ClassificationReadoutLayer(
                readout_neuron_ids=readout_ids,
                n_output=n_output,
                dampening_factor=dampening_factor,
                seq_len=seq_len,
                down_sample=down_sample,
                name=f'classification_readout_{area}'
            )
            outputs_dict[area] = readout_layer(spikes_dict[area])
        else:
            # output = tf.keras.layers.Dense(n_output, name=f'projection_{area}', trainable=True)(spikes_dict[area])
            projection_initializer = tf.keras.initializers.GlorotUniform(seed=int(seed) + 17)
            output = tf.keras.layers.Dense(
                n_output,
                name=f'projection_{area}',
                trainable=False,
                kernel_initializer=projection_initializer,
                bias_initializer="zeros",
            )(spikes_dict[area])
            mean_output = tf.reshape(output, (-1, int(seq_len / down_sample), down_sample, n_output))
            mean_output = tf.reduce_mean(mean_output, axis=2)
            mean_output = tf.nn.softmax(mean_output, axis=-1)
            outputs_dict[area] = mean_output

    if use_state_input:
        if use_dummy_state_input:
            inputs = [x, state_input_holder, initial_state_holder]
        else:
            inputs = [x, initial_state_holder]
    else:
        if use_dummy_state_input:
            inputs = [x, state_input_holder]
        else:
            inputs = [x]

    # many_input_model = tf.keras.Model(inputs=inputs, outputs=[rsnn_out, outputs_dict['v1']])
    many_input_model = tf.keras.Model(inputs=inputs, outputs=[outputs_dict['v1']])

    if add_metric:
        # add the firing rate of the neurons as a metric to the model
        # computes the mean of the spikes tensor along the second and third dimensions
        # (which represent time and neurons),
        rate = tf.reduce_mean(spikes_dict['v1'])
        many_input_model.add_metric(rate, name="rate")

    return many_input_model

def build_sequence_only_model(model, rsnn_layer, name="rsnn_sequences"):
    """Create a training model that returns only the RNN sequence outputs."""
    return tf.keras.Model(
        inputs=model.inputs,
        outputs=rsnn_layer.output[0],
        name=name,
    )


def build_sequence_and_state_model(model, rsnn_layer, name="rsnn_sequence_and_state"):
    """Expose the RNN sequence outputs and final state without the readout."""
    return tf.keras.Model(
        inputs=model.inputs,
        outputs=rsnn_layer.output,
        name=name,
    )


class SegmentedRecomputeRunner:
    """Run an RNN in recomputed temporal chunks while preserving exact BPTT."""

    def __init__(
        self,
        core_model,
        sequence_length,
        chunk_size,
        differentiate_inputs=True,
    ):
        sequence_length = int(sequence_length)
        chunk_size = int(chunk_size)
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive.")
        if chunk_size < 1 or chunk_size > sequence_length:
            raise ValueError(
                "chunk_size must satisfy 1 <= chunk_size <= sequence_length."
            )

        self.core_model = core_model
        self.sequence_length = sequence_length
        self.chunk_size = chunk_size
        self.differentiate_inputs = bool(differentiate_inputs)
        self.n_full_chunks = sequence_length // chunk_size
        self.remainder_size = sequence_length % chunk_size
        self.chunk_sizes = tuple(
            min(chunk_size, sequence_length - start)
            for start in range(0, sequence_length, chunk_size)
        )
        self.n_chunks = len(self.chunk_sizes)
        self.n_sequence_outputs = len(tf.nest.flatten(core_model.output[0]))

    def _run_chunk(self, inputs, *state):
        outputs = self.core_model((inputs, tuple(state)))
        return tuple(tf.nest.flatten(outputs[0])) + tuple(outputs[1:])

    def _slice_chunk(self, inputs, chunk_index, chunk_length, time_check=None):
        start = chunk_index * self.chunk_size
        dependencies = () if time_check is None else (time_check,)
        with tf.control_dependencies(dependencies):
            chunk = tf.identity(inputs[:, start:start + chunk_length, ...])
        if inputs.shape.rank is not None:
            chunk_shape = inputs.shape.as_list()
            chunk_shape[1] = chunk_length
            chunk = tf.ensure_shape(chunk, chunk_shape)
        return chunk

    def _assemble_chunks(self, array, output_spec, inputs):
        chunks = tuple(array.read(index) for index in range(self.n_chunks))
        sequence = chunks[0] if self.n_chunks == 1 else tf.concat(chunks, axis=1)
        if output_spec.shape.rank is not None:
            output_shape = output_spec.shape.as_list()
            output_shape[0] = inputs.shape[0]
            output_shape[1] = self.sequence_length
            sequence = tf.ensure_shape(sequence, output_shape)
        return sequence

    def _forward(self, inputs, initial_state, time_check):
        sequence_specs = tf.nest.flatten(self.core_model.output[0])
        sequence_arrays = tuple(
            tf.TensorArray(
                dtype=spec.dtype,
                size=self.n_chunks,
                infer_shape=self.remainder_size == 0,
                clear_after_read=True,
            )
            for spec in sequence_specs
        )
        state_arrays = tuple(
            tf.TensorArray(dtype=value.dtype, size=self.n_chunks, clear_after_read=True)
            for value in initial_state
        )

        def run_full_chunk(chunk_index, arrays, states, history):
            chunk = self._slice_chunk(inputs, chunk_index, self.chunk_size, time_check)
            history = tuple(
                array.write(chunk_index, value)
                for array, value in zip(history, states)
            )
            flat_outputs = self._run_chunk(chunk, *states)
            chunk_sequences = flat_outputs[:self.n_sequence_outputs]
            next_state = tuple(flat_outputs[self.n_sequence_outputs:])
            arrays = tuple(
                array.write(chunk_index, value)
                for array, value in zip(arrays, chunk_sequences)
            )
            return chunk_index + 1, arrays, next_state, history

        _, sequence_arrays, state, state_arrays = tf.while_loop(
            lambda chunk_index, _arrays, _state, _history: (
                chunk_index < self.n_full_chunks
            ),
            run_full_chunk,
            (tf.constant(0), sequence_arrays, tuple(initial_state), state_arrays),
            parallel_iterations=1,
        )

        if self.remainder_size:
            remainder_index = tf.constant(self.n_full_chunks)
            state_arrays = tuple(
                array.write(remainder_index, value)
                for array, value in zip(state_arrays, state)
            )
            chunk = self._slice_chunk(
                inputs, remainder_index, self.remainder_size, time_check
            )
            flat_outputs = self._run_chunk(chunk, *state)
            chunk_sequences = flat_outputs[:self.n_sequence_outputs]
            state = tuple(flat_outputs[self.n_sequence_outputs:])
            sequence_arrays = tuple(
                array.write(remainder_index, value)
                for array, value in zip(sequence_arrays, chunk_sequences)
            )

        sequences = tuple(
            self._assemble_chunks(array, spec, inputs)
            for array, spec in zip(sequence_arrays, sequence_specs)
        )
        return sequences, state, state_arrays

    def _backward(
        self,
        inputs,
        state_arrays,
        final_state,
        sequence_gradients,
        final_state_gradients,
        variables,
    ):
        inputs_are_differentiable = (
            self.differentiate_inputs
            and (inputs.dtype.is_floating or inputs.dtype.is_complex)
        )
        input_gradients = (
            tf.TensorArray(
                dtype=inputs.dtype,
                size=self.n_chunks,
                infer_shape=self.remainder_size == 0,
                clear_after_read=True,
            ),
        ) if inputs_are_differentiable else ()
        state_dtypes = tuple(array.dtype for array in state_arrays)
        state_cotangents = tuple(
            (
                gradient if gradient is not None else tf.zeros_like(final_state[index])
            )
            if dtype.is_floating or dtype.is_complex
            else tf.zeros(tf.shape(final_state[index]), tf.float32)
            for index, (dtype, gradient) in enumerate(
                zip(state_dtypes, final_state_gradients)
            )
        )
        variable_gradients = tuple(tf.zeros_like(value) for value in variables)

        def reverse_chunk(
            chunk_index,
            chunk_length,
            input_history,
            output_state_cotangents,
            accumulated_variable_gradients,
        ):
            start = chunk_index * self.chunk_size
            chunk = self._slice_chunk(inputs, chunk_index, chunk_length)
            boundary_state = tuple(
                array.read(chunk_index) for array in state_arrays
            )
            differentiable_state_indices = tuple(
                index for index, value in enumerate(boundary_state)
                if value.dtype.is_floating or value.dtype.is_complex
            )
            differentiable_state = tuple(
                boundary_state[index] for index in differentiable_state_indices
            )

            with tf.GradientTape() as tape:
                if inputs_are_differentiable:
                    tape.watch(chunk)
                tape.watch(differentiable_state)
                tape.watch(variables)
                recomputed = self._run_chunk(chunk, *boundary_state)

            chunk_sequence_gradients = tuple(
                None if gradient is None
                else gradient[:, start:start + chunk_length, ...]
                for gradient in sequence_gradients
            )
            state_output_gradients = tuple(
                output_state_cotangents[index]
                if dtype.is_floating or dtype.is_complex else None
                for index, dtype in enumerate(state_dtypes)
            )
            input_targets = (chunk,) if inputs_are_differentiable else ()
            targets = input_targets + differentiable_state + variables
            gradients = tape.gradient(
                recomputed,
                targets,
                output_gradients=chunk_sequence_gradients + state_output_gradients,
                unconnected_gradients=tf.UnconnectedGradients.ZERO,
            )

            target_offset = int(inputs_are_differentiable)
            if inputs_are_differentiable:
                chunk_gradient = gradients[0]
            state_gradient_values = gradients[
                target_offset:target_offset + len(differentiable_state)
            ]
            variable_gradient_values = gradients[
                target_offset + len(differentiable_state):
            ]
            next_state_cotangents = list(output_state_cotangents)
            for index, gradient in zip(
                differentiable_state_indices, state_gradient_values
            ):
                next_state_cotangents[index] = gradient
            next_variable_gradients = tuple(
                accumulated + gradient
                for accumulated, gradient in zip(
                    accumulated_variable_gradients, variable_gradient_values
                )
            )
            if inputs_are_differentiable:
                input_history = (
                    input_history[0].write(chunk_index, chunk_gradient),
                )
            return input_history, tuple(next_state_cotangents), next_variable_gradients

        if self.remainder_size:
            input_gradients, state_cotangents, variable_gradients = reverse_chunk(
                tf.constant(self.n_full_chunks),
                self.remainder_size,
                input_gradients,
                state_cotangents,
                variable_gradients,
            )

        def run_reverse_full_chunk(
            chunk_index, input_history, output_state_cotangents,
            accumulated_variable_gradients,
        ):
            input_history, state_values, variable_values = reverse_chunk(
                chunk_index,
                self.chunk_size,
                input_history,
                output_state_cotangents,
                accumulated_variable_gradients,
            )
            return chunk_index - 1, input_history, state_values, variable_values

        _, input_gradients, state_cotangents, variable_gradients = tf.while_loop(
            lambda chunk_index, _inputs, _states, _variables: chunk_index >= 0,
            run_reverse_full_chunk,
            (
                tf.constant(self.n_full_chunks - 1),
                input_gradients,
                state_cotangents,
                variable_gradients,
            ),
            parallel_iterations=1,
        )
        if inputs_are_differentiable:
            input_chunks = tuple(
                input_gradients[0].read(index) for index in range(self.n_chunks)
            )
            input_gradient = (
                input_chunks[0] if self.n_chunks == 1
                else tf.concat(input_chunks, axis=1)
            )
            input_gradient = tf.ensure_shape(input_gradient, inputs.shape)
        else:
            input_gradient = None
        initial_state_gradients = tuple(
            gradient if dtype.is_floating or dtype.is_complex else None
            for gradient, dtype in zip(state_cotangents, state_dtypes)
        )
        return input_gradient, initial_state_gradients, variable_gradients

    def __call__(self, inputs, initial_state):
        time_check = tf.debugging.assert_equal(
            tf.shape(inputs)[1],
            self.sequence_length,
            message="Segmented input length does not match sequence_length.",
        )

        @tf.custom_gradient
        def segmented_rollout(inputs, *initial_state):
            with record.stop_recording():
                sequences, state, state_arrays = self._forward(
                    inputs, initial_state, time_check
                )
            flat_outputs = sequences + state

            def grad(*output_gradients, variables=None):
                sequence_gradients = output_gradients[:self.n_sequence_outputs]
                final_state_gradients = output_gradients[self.n_sequence_outputs:]
                variables = tuple(variables or ())
                input_gradient, state_gradients, variable_gradients = self._backward(
                    inputs,
                    state_arrays,
                    state,
                    sequence_gradients,
                    final_state_gradients,
                    variables,
                )
                argument_gradients = (input_gradient,) + state_gradients
                if variables:
                    return argument_gradients, list(variable_gradients)
                return argument_gradients

            return flat_outputs, grad

        return segmented_rollout(inputs, *tuple(initial_state))


def build_state_only_model(model, rsnn_layer, name="rsnn_state"):
    """Create a model that returns only the final RNN state (no sequences)."""
    rnn_inputs = rsnn_layer.input
    if isinstance(rnn_inputs, (list, tuple)):
        full_inputs = rnn_inputs[0]
        state_inputs = list(rnn_inputs[1:])
    else:
        full_inputs = rnn_inputs
        state_inputs = []
    state_rnn = tf.keras.layers.RNN(
        rsnn_layer.cell, return_sequences=False, return_state=True, name=name
    )
    # Preserve heterogeneous state dtypes for the state rollout path as well.
    state_rnn._autocast = False
    if state_inputs:
        state_out = state_rnn(full_inputs, initial_state=state_inputs)
    else:
        state_out = state_rnn(full_inputs)
    return tf.keras.Model(inputs=model.inputs, outputs=state_out[1:])



# if name is main run the create model function
if __name__ == "__main__":
    # load the network
    import load_sparse

    n_input = 17400
    n_neurons = 1574

    network, lgn_input, bkg_input = load_sparse.cached_load_v1(
        n_input,
        n_neurons,
        True,
        "GLIF_network",
        seed=3000,
        connected_selection=False,
        n_output=2,
        neurons_per_output=16,
    )
    # create the model
    model = create_model(
        network,
        lgn_input,
        bkg_input,
        seq_len=100,
        n_input=n_input,
        n_output=2,
        cue_duration=20,
        dtype=tf.float32,
        input_weight_scale=1.0,
        gauss_std=0.5,
        dampening_factor=0.2,
        lr_scale=800.0,
        train_recurrent=True,
        train_recurrent_per_type=False,
        train_input=True,
        neuron_output=False,
        recurrent_dampening_factor=0.5,
        use_state_input=False,
        return_state=False,
        return_sequences=False,
        down_sample=50,
        add_metric=True,
        max_delay=0,
        batch_size=1,
        pseudo_gauss=False,
        hard_reset=True,
    )
