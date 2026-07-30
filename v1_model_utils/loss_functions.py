import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
import pandas as pd
import pickle as pkl
try:
    from numba import njit
    HAS_NUMBA = True
except Exception:
    HAS_NUMBA = False

import os
import sys
sys.path.append(os.path.join(os.getcwd(), "v1_model_utils"))
import other_v1_utils


@njit(cache=True)
def _build_group_order_numba(group_ids, n_groups):
    """Stable bucket ordering by group id with CSR row_splits."""
    n = group_ids.shape[0]
    counts = np.zeros(n_groups, dtype=np.int64)
    for i in range(n):
        counts[group_ids[i]] += 1

    row_splits = np.empty(n_groups + 1, dtype=np.int64)
    row_splits[0] = 0
    for g in range(n_groups):
        row_splits[g + 1] = row_splits[g] + counts[g]

    write_ptr = np.empty(n_groups, dtype=np.int64)
    for g in range(n_groups):
        write_ptr[g] = row_splits[g]

    order = np.empty(n, dtype=np.int64)
    for i in range(n):
        g = group_ids[i]
        pos = write_ptr[g]
        order[pos] = i
        write_ptr[g] = pos + 1

    return order, row_splits

@njit(cache=True)
def _sort_initial_values_by_group_numba(initial_values, order, row_splits):
    """Sort initial values independently inside each group segment."""
    out = np.empty(initial_values.shape[0], dtype=initial_values.dtype)
    n_groups = row_splits.shape[0] - 1
    for g in range(n_groups):
        start = row_splits[g]
        end = row_splits[g + 1]
        size = end - start
        buf = np.empty(size, dtype=initial_values.dtype)
        for j in range(size):
            buf[j] = initial_values[order[start + j]]
        buf.sort()
        for j in range(size):
            out[start + j] = buf[j]
    return out


def _build_group_order_numpy(group_ids, n_groups):
    order = np.argsort(group_ids, kind='stable')
    counts = np.bincount(group_ids, minlength=n_groups)
    row_splits = np.empty(n_groups + 1, dtype=np.int64)
    row_splits[0] = 0
    np.cumsum(counts, dtype=np.int64, out=row_splits[1:])
    return order.astype(np.int64, copy=False), row_splits


def _sort_initial_values_by_group_numpy(initial_values, order, row_splits):
    out = np.empty(initial_values.shape[0], dtype=initial_values.dtype)
    for g in range(row_splits.shape[0] - 1):
        start = row_splits[g]
        end = row_splits[g + 1]
        out[start:end] = np.sort(initial_values[order[start:end]])
    return out


@njit(cache=True)
def _group_mean_and_count_numba(values, group_ids, n_groups):
    sums = np.zeros(n_groups, dtype=values.dtype)
    counts = np.zeros(n_groups, dtype=np.int64)
    n = values.shape[0]
    for i in range(n):
        g = group_ids[i]
        sums[g] += values[i]
        counts[g] += 1

    means = np.zeros(n_groups, dtype=values.dtype)
    for g in range(n_groups):
        if counts[g] > 0:
            means[g] = sums[g] / counts[g]
    return means, counts


def _group_mean_and_count_numpy(values, group_ids, n_groups):
    sums = np.zeros(n_groups, dtype=values.dtype)
    counts = np.zeros(n_groups, dtype=np.int64)
    for i in range(values.shape[0]):
        g = group_ids[i]
        sums[g] += values[i]
        counts[g] += 1

    means = np.zeros(n_groups, dtype=values.dtype)
    non_zero = counts > 0
    means[non_zero] = sums[non_zero] / counts[non_zero].astype(values.dtype, copy=False)
    return means, counts


@njit(cache=True)
def _build_lognormal_targets_numba(initial_value, group_ids, n_groups, log_epsilon, std_epsilon):
    log_initial_value = np.log(np.abs(initial_value) + log_epsilon)
    log_mean_all, counts = _group_mean_and_count_numba(log_initial_value, group_ids, n_groups)

    squared_diff = np.empty_like(log_initial_value)
    for i in range(log_initial_value.shape[0]):
        g = group_ids[i]
        d = log_initial_value[i] - log_mean_all[g]
        squared_diff[i] = d * d

    log_var_all, _ = _group_mean_and_count_numba(squared_diff, group_ids, n_groups)
    log_std_all = np.sqrt(log_var_all + std_epsilon)

    n_valid = 0
    for g in range(n_groups):
        if counts[g] > 2:
            n_valid += 1
    valid_indices = np.empty(n_valid, dtype=np.int32)
    j = 0
    for g in range(n_groups):
        if counts[g] > 2:
            valid_indices[j] = g
            j += 1

    return log_mean_all, log_std_all, counts, valid_indices


def _build_lognormal_targets_numpy(initial_value, group_ids, n_groups, log_epsilon, std_epsilon):
    log_initial_value = np.log(np.abs(initial_value) + log_epsilon).astype(initial_value.dtype, copy=False)
    log_mean_all, counts = _group_mean_and_count_numpy(log_initial_value, group_ids, n_groups)

    gathered_means = log_mean_all[group_ids]
    squared_diff = np.square(log_initial_value - gathered_means, dtype=initial_value.dtype)
    log_var_all, _ = _group_mean_and_count_numpy(squared_diff, group_ids, n_groups)
    log_std_all = np.sqrt(log_var_all + std_epsilon).astype(initial_value.dtype, copy=False)

    valid_indices = np.where(counts > 2)[0].astype(np.int32, copy=False)
    return log_mean_all, log_std_all, counts, valid_indices


# class StiffRegularizer(tf.keras.regularizers.Regularizer):
#     def __init__(self, strength, initial_value):
#         super().__init__()
#         self._strength = strength
#         self._initial_value = tf.Variable(initial_value, trainable=False)

#     def __call__(self, x):
#         return self._strength * tf.reduce_mean(tf.square(x - self._initial_value))

# class L2Regularizer(tf.keras.regularizers.Regularizer):
#     def __init__(self, strength, initial_value):
#         super().__init__()
#         self._strength = strength
#         self._initial_value = tf.Variable(initial_value, trainable=False)

#     def __call__(self, x):
#         return self._strength * tf.reduce_mean(tf.square(x))

class MeanStiffRegularizer(Layer):
    def __init__(self, strength, network, penalize_relative_change=False, dtype=tf.float32):
        super().__init__()
        self._strength = tf.cast(strength, dtype)
        self._dtype = dtype
        self._penalize_relative_change = penalize_relative_change
        # Compute voltage scale
        voltage_scale = (network['node_params']['V_th'] - network['node_params']['E_L']).astype(np.float32)
        # Get the initial weights and properly scale them down
        indices = network["synapses"]["indices"]
        weights = np.array(network["synapses"]["weights"], dtype=np.float32)
        edge_type_ids = network['synapses']['edge_type_ids']
        # Scale initial values by the voltage scale of the node IDs
        initial_value = weights / voltage_scale[network['node_type_ids'][indices[:, 0]]]
        # Find unique values and their first occurrence indices
        unique_edge_types, self.idx = np.unique(edge_type_ids, return_inverse=True)
        # Sort first_occurrence_indices to maintain the order of first appearances
        self.num_unique = unique_edge_types.shape[0]
        sum_weights = np.bincount(self.idx, weights=initial_value, minlength=self.num_unique)
        count_weights = np.bincount(self.idx, minlength=self.num_unique)
        initial_mean_weights = sum_weights / count_weights
        # Determine target mean weights
        if self._penalize_relative_change:
            epsilon = np.float32(1e-4)
            denominator = np.maximum(np.abs(initial_mean_weights), epsilon)
            self._denominator = tf.constant(denominator, dtype=self._dtype)

        self.idx = tf.constant(self.idx, dtype=tf.int32)
        self.num_unique = tf.constant(self.num_unique, dtype=tf.int32)
        self._target_mean_weights = tf.constant(initial_mean_weights, dtype=self._dtype)

    @tf.function(jit_compile=True)
    def __call__(self, x):

        if len(x.shape) > 1 and x.shape[1] == 1:
            x = tf.squeeze(x, axis=1)

        if x.dtype != self._dtype:
            x = tf.cast(x, self._dtype)

        mean_edge_type_weights = tf.math.unsorted_segment_mean(x, self.idx, self.num_unique)
        if self._penalize_relative_change:
            # return self._strength * tf.reduce_mean(tf.abs(x - self._initial_value))
            relative_deviation = (mean_edge_type_weights - self._target_mean_weights) / self._denominator
            # Penalize the relative deviation
            reg_loss = tf.sqrt(tf.reduce_mean(tf.square(relative_deviation)))
        else:
            reg_loss = tf.reduce_mean(tf.square(mean_edge_type_weights - self._target_mean_weights))

        return tf.cast(reg_loss, dtype=self._dtype) * self._strength

class MeanStdStiffRegularizer(Layer):
    def __init__(self, strength, network, penalize_relative_change=False,
                    std_weight=0.5, logspace=True, dtype=tf.float32):
        super().__init__()
        self._strength = tf.cast(strength, dtype)
        self._dtype = dtype
        self._penalize_relative_change = penalize_relative_change
        self._std_weight = tf.cast(std_weight, self._dtype)  # Weight for std deviation component
        self._logspace = logspace  # Whether to use logspace for std calculation
        self._epsilon = tf.constant(1e-6, dtype=self._dtype)  # Small value to avoid log(0)

        # Compute voltage scale
        voltage_scale = (network['node_params']['V_th'] - network['node_params']['E_L']).astype(np.float32)
        # Get the initial weights and properly scale them down
        indices = network["synapses"]["indices"]
        weights = np.array(network["synapses"]["weights"], dtype=np.float32)
        edge_type_ids = network['synapses']['edge_type_ids']
        # Scale initial values by the voltage scale of the node IDs
        initial_value = weights / voltage_scale[network['node_type_ids'][indices[:, 0]]]
        # Find unique values and their first occurrence indices
        unique_edge_types, self.idx = np.unique(edge_type_ids, return_inverse=True)
        # Sort first_occurrence_indices to maintain the order of first appearances
        self.num_unique = unique_edge_types.shape[0]
        sum_weights = np.bincount(self.idx, weights=initial_value, minlength=self.num_unique)
        count_weights = np.bincount(self.idx, minlength=self.num_unique)
        initial_mean_weights = sum_weights / count_weights

        # Calculate std deviation per edge type in logspace if requested
        self._target_std_weights = []
        for i in range(self.num_unique):
            edge_type_weights = initial_value[self.idx == i]
            if self._logspace:
                # Use abs to handle any potential negative weights
                log_weights = np.log(np.abs(edge_type_weights) + np.float32(1e-6))
                std = np.std(log_weights)
            else:
                std = np.std(edge_type_weights)
            self._target_std_weights.append(std)
        self._target_std_weights = tf.constant(self._target_std_weights, dtype=self._dtype)

        # Determine target mean weights and denominators for relative change
        if self._penalize_relative_change:
            epsilon = np.float32(1e-4)
            denominator = np.maximum(np.abs(initial_mean_weights), epsilon)
            self._denominator = tf.constant(denominator, dtype=self._dtype)
            # Also for std deviation
            epsilon = np.float32(1e-3)
            std_denominator = np.maximum(np.abs(self._target_std_weights.numpy()), epsilon)
            self._std_denominator = tf.constant(std_denominator, dtype=self._dtype)

        self.idx = tf.constant(self.idx, dtype=tf.int32)
        self.num_unique = tf.constant(self.num_unique, dtype=tf.int32)
        self._target_mean_weights = tf.constant(initial_mean_weights, dtype=self._dtype)

    @tf.function(jit_compile=True)
    def __call__(self, x):

        if len(x.shape) > 1 and x.shape[1] == 1:
            x = tf.squeeze(x, axis=1)

        if x.dtype != self._dtype:
            x = tf.cast(x, self._dtype)

        # Calculate mean per edge type
        mean_edge_type_weights = tf.math.unsorted_segment_mean(x, self.idx, self.num_unique)

        # Calculate std deviation per edge type
        if self._logspace:
            # Use abs for log transformation to handle potential negative weights
            abs_x = tf.abs(x)
            log_x = tf.math.log(abs_x + self._epsilon)
            mean_log_x = tf.math.unsorted_segment_mean(log_x, self.idx, self.num_unique)
            # Get log values for each input position
            gathered_mean_log_x = tf.gather(mean_log_x, self.idx)
            # Calculate squared differences
            squared_diffs = tf.square(log_x - gathered_mean_log_x)
            # Calculate variance and std dev with epsilon for stability
            log_var = tf.math.unsorted_segment_mean(squared_diffs, self.idx, self.num_unique)
            std_edge_type_weights = tf.sqrt(log_var + self._epsilon)        # prevent division by zero for edge_types with just 1 edge
        else:
            # Calculate variance directly with improved stability
            gathered_means = tf.gather(mean_edge_type_weights, self.idx)
            squared_diffs = tf.square(x - gathered_means)
            var_edge_type_weights = tf.math.unsorted_segment_mean(squared_diffs, self.idx, self.num_unique)
            std_edge_type_weights = tf.sqrt(var_edge_type_weights + self._epsilon) # prevent division by zero for edge_types with just 1 edge

        # Calculate losses with improved numerical stability
        if self._penalize_relative_change:
            # Mean deviation component - with safe division
            mean_relative_deviation = (mean_edge_type_weights - self._target_mean_weights) / self._denominator
            mean_loss = tf.sqrt(tf.reduce_mean(tf.square(mean_relative_deviation)))

            # Std deviation component - with safe division
            std_relative_deviation = (std_edge_type_weights - self._target_std_weights) / self._std_denominator
            std_loss = tf.sqrt(tf.reduce_mean(tf.square(std_relative_deviation)))
        else:
            # Mean deviation component
            mean_squared_error = tf.square(mean_edge_type_weights - self._target_mean_weights)
            mean_loss = tf.reduce_mean(mean_squared_error)

            # Std deviation component
            std_squared_error = tf.square(std_edge_type_weights - self._target_std_weights)
            std_loss = tf.reduce_mean(std_squared_error)

        # Combine losses with weighting
        total_loss = (tf.cast(1.0, self._dtype) - self._std_weight) * mean_loss + self._std_weight * std_loss
        return tf.cast(total_loss, dtype=self._dtype) * self._strength


class StiffKLLogNormalRegularizer(Layer):
    """Regularization using KL divergence for log-normal distributions

    Args:
        Layer (_type_): _description_
    """
    def __init__(self, strength, network, dtype=tf.float32):
        super().__init__()
        # Keep this regularizer in fp32 for numerical stability.
        self._dtype = dtype
        self._strength = tf.cast(strength, self._dtype)
        self.epsilon = tf.constant(1e-8, dtype=self._dtype)

        # Compute voltage scale and rescale initial weights
        voltage_scale = (network['node_params']['V_th'] - network['node_params']['E_L']).astype(np.float32)
        indices = np.asarray(network["synapses"]["indices"])
        weights = np.asarray(network["synapses"]["weights"], dtype=np.float32)
        edge_type_ids = np.asarray(network['synapses']['edge_type_ids'])
        initial_value = weights / voltage_scale[np.asarray(network['node_type_ids'])[indices[:, 0]]]

        # Edge type indexing
        _, idx_np = np.unique(edge_type_ids, return_inverse=True)
        idx_np = idx_np.astype(np.int32, copy=False)
        n_groups = int(np.max(idx_np) + 1) if idx_np.size else 0

        if idx_np.size == 0:
            log_mean_all_np = np.empty((0,), dtype=np.float32)
            log_std_all_np = np.empty((0,), dtype=np.float32)
            valid_indices_np = np.empty((0,), dtype=np.int32)
        else:
            log_epsilon = np.float32(1e-10)
            std_epsilon = np.float32(1e-8)
            if HAS_NUMBA:
                log_mean_all_np, log_std_all_np, count_weights, valid_indices_np = _build_lognormal_targets_numba(
                    initial_value, idx_np, n_groups, log_epsilon, std_epsilon
                )
            else:
                log_mean_all_np, log_std_all_np, count_weights, valid_indices_np = _build_lognormal_targets_numpy(
                    initial_value, idx_np, n_groups, log_epsilon, std_epsilon
                )

        self.idx = tf.constant(idx_np, dtype=tf.int32)
        self.num_unique = tf.constant(n_groups, dtype=tf.int32)
        self.valid_indices = tf.constant(valid_indices_np, dtype=tf.int32)
        self.num_valid = tf.constant(valid_indices_np.size, dtype=tf.int32)

        self._target_log_mean_all = tf.constant(log_mean_all_np, dtype=self._dtype)
        self._target_log_mean = tf.constant(log_mean_all_np[valid_indices_np], dtype=self._dtype)
        self._target_log_std = tf.constant(log_std_all_np[valid_indices_np], dtype=self._dtype)

    @tf.function(jit_compile=True)
    def __call__(self, x):
        if len(x.shape) > 1 and x.shape[1] == 1:
            x = tf.squeeze(x, axis=1)

        if x.dtype != self._dtype:
            x = tf.cast(x, self._dtype)

        # Calculate log of absolute values with epsilon for stability
        log_x = tf.math.log(tf.abs(x) + self.epsilon)
        # Calculate mean and std of log(x) per edge type
        log_mean_all = tf.math.unsorted_segment_mean(log_x, self.idx, self.num_unique)
        # Calculate std deviation with better numerical stability
        squared_diff = tf.square(log_x - tf.gather(log_mean_all, self.idx))
        log_var_all = tf.math.unsorted_segment_mean(squared_diff, self.idx, self.num_unique)
        log_std_all = tf.sqrt(log_var_all + self.epsilon)  # Add epsilon before sqrt
        # Use gather instead of boolean_mask
        log_mean = tf.gather(log_mean_all, self.valid_indices)
        log_std = tf.gather(log_std_all, self.valid_indices)
        # KL divergence calculation with improved numerical stability
        log_ratio = tf.math.log(self._target_log_std + self.epsilon) - tf.math.log(log_std + self.epsilon)
        denominator = 2.0 * tf.square(self._target_log_std) + self.epsilon
        std_ratio = tf.square(log_std) / denominator
        diff_ratio = tf.square(log_mean - self._target_log_mean) / denominator

        # Combine terms after stable calculations
        kl = log_ratio + std_ratio + diff_ratio - 0.5
        # Use reduce_mean without abs since KL should be positive
        kl_mean = tf.reduce_mean(kl)

        return self._strength * tf.cast(kl_mean, dtype=self._dtype)

class L2Regularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, strength, network, flags, penalize_relative_change=False, dtype=tf.float32):
        super().__init__()
        # Keep this regularizer in fp32 for numerical stability.
        self._dtype = dtype
        self._strength = tf.cast(strength, self._dtype)

        if penalize_relative_change:
            # Compute voltage scale
            voltage_scale = (network['node_params']['V_th'] - network['node_params']['E_L']).astype(np.float32)
            # Get the initial weights and properly scale them down
            indices = network["synapses"]["indices"]
            weights = np.array(network["synapses"]["weights"], dtype=np.float32)
            # Scale initial values by the voltage scale of the node IDs
            voltage_scale_node_ids = voltage_scale[network['node_type_ids'][indices[:, 0]]]
            initial_value = weights / voltage_scale_node_ids
            # using the edge_type ids group calculate the mean weight of each type of edge in the network and then create a constant with same shape as weights and with each value corresponding to the populations mean
            # Calculate mean weights for each edge type
            edge_type_ids = np.array(network['synapses']['edge_type_ids'])
            unique_edge_type_ids, inverse_indices = np.unique(edge_type_ids, return_inverse=True)
            mean_weights = np.array([np.mean(initial_value[edge_type_ids == edge_type_id]) for edge_type_id in unique_edge_type_ids])
            # Create target mean weights array based on the edge type indices
            target_mean_weights = mean_weights[inverse_indices]
            epsilon = 1e-4
            target_mean_weights = np.maximum(np.abs(target_mean_weights), epsilon) # Ensure no zero values for stability
            self._target_mean_weights = tf.constant(target_mean_weights, dtype=self._dtype)
        else:
            self._target_mean_weights = None

    @tf.function(jit_compile=True)
    def __call__(self, x):

        if len(x.shape) > 1 and x.shape[1] == 1:
            x = tf.squeeze(x, axis=1)

        if x.dtype != self._dtype:
            x = tf.cast(x, self._dtype)

        if self._target_mean_weights is None:
            return tf.cast(self._strength * tf.reduce_mean(tf.square(x)), dtype=self._dtype)
        else:
            relative_deviation = x / self._target_mean_weights
            mse = self._strength * tf.reduce_mean(tf.square(relative_deviation))
            return tf.cast(mse, dtype=self._dtype)

# class EarthMoversDistanceRegularizer(Layer):
#     """
#     Regularizer that penalizes the Earth Mover's Distance (Wasserstein-1) between the current and initial
#     synaptic weight distributions, per edge type, averaged over all edge types.
#     Uses TF operations for initialization and tf.map_fn in call for memory efficiency.
#     """
#     def __init__(self, strength, network, dtype=tf.float32):
#         super().__init__()
#         self._strength = tf.cast(strength, dtype)
#         self._dtype = dtype

#         # --- Original Initialization Logic ---
#         voltage_scale = (network['node_params']['V_th'] - network['node_params']['E_L']).astype(np.float32)
#         indices = network["synapses"]["indices"]
#         initial_value_np = np.array(network["synapses"]["weights"], dtype=np.float32)
#         # edge_type_ids_np = network['synapses']['edge_type_ids']
#         # use the connection_type_ids instead
#         edge_type_ids_np = other_v1_utils.connection_type_ids(network)
#         initial_value_np /= voltage_scale[network['node_type_ids'][indices[:, 0]]]
#         n_edges = len(initial_value_np)
#         # --- End Original Initialization Logic ---

#         # Convert to TF Tensors
#         self._initial_value = tf.constant(initial_value_np, dtype=self._dtype)
#         edge_type_ids = tf.constant(edge_type_ids_np, dtype=tf.int32)

#         unique_edge_types, idx = np.unique(edge_type_ids_np, return_inverse=True)
#         idx = tf.constant(idx, dtype=tf.int32)
#         self.num_unique = tf.constant(unique_edge_types.shape[0], dtype=tf.int32)

#         # presort the initial value
#         for i in tf.range(self.num_unique):
#             mask = tf.equal(idx, i)
#             y_i = tf.boolean_mask(self._initial_value, mask)
#             y_i = tf.sort(y_i)
#             self._initial_value = tf.tensor_scatter_nd_update(self._initial_value, tf.where(mask), y_i)

#         ### 2. Reorder original_indices and initial_value based on sorted edge types
#         original_indices = tf.range(n_edges, dtype=tf.int32)
#         sorted_indices = tf.argsort(edge_type_ids)
#         permuted_original_indices = tf.gather(original_indices, sorted_indices)
#         sorted_edge_type_ids = tf.gather(edge_type_ids, sorted_indices) # Needed for unique

#         # 3. Build row_splits directly from per-type counts.
#         # This avoids RaggedTensor.from_value_rowids -> DenseBincount, which raises
#         # under GPU deterministic mode in TF 2.15.
#         _, _, counts = tf.unique_with_counts(sorted_edge_type_ids)
#         row_splits = tf.concat(
#             [tf.zeros((1,), dtype=counts.dtype), tf.cumsum(counts)],
#             axis=0,
#         )

#         # 4. Construct group_indices RaggedTensor (indices into the *original* weight tensor)
#         self._group_indices = tf.RaggedTensor.from_row_splits(
#             values=permuted_original_indices,
#             row_splits=row_splits,
#             validate=False
#         )

    # @tf.function(jit_compile=False) # Do not use jit_compile=True. It uses a lot of memory.
    # def __call__(self, x):
    #     if x.dtype != self._dtype:
    #         x = tf.cast(x, self._dtype)
    #     if len(x.shape) > 1 and x.shape[1] == 1:
    #         x = tf.squeeze(x, axis=1)
    #     emd_losses = tf.TensorArray(self._dtype, size=self.num_unique)
    #     for i in tf.range(self.num_unique):
    #         x_i = tf.gather(x, self._group_indices[i])
    #         y_i = tf.gather(self._initial_value, self._group_indices[i])

    #         # y_i is presorted.
    #         emd = tf.reduce_mean(tf.abs(tf.sort(x_i) - y_i))
    #         emd_losses = emd_losses.write(i, emd)
    #     emd_losses = emd_losses.stack()
    #     reg_loss = tf.reduce_mean(emd_losses)
    #     return reg_loss * self._strength


class EarthMoversDistanceRegularizer(Layer):
    """
    EMD Regularizer that penalizes the Earth Mover's Distance (Wasserstein-1) between the current and initial
    synaptic weight distributions, per edge type, averaged over all edge types.
    Uses TF operations for execution and CPU numba for initialization.
    """

    def __init__(self, strength, network, dtype=tf.float32):
        super().__init__()
        self._strength = tf.cast(strength, dtype)
        self._dtype = dtype

        voltage_scale = (network['node_params']['V_th'] - network['node_params']['E_L']).astype(np.float32)
        indices = np.asarray(network['synapses']['indices'])
        initial_value_np = np.asarray(network['synapses']['weights'], dtype=np.float32).copy()
        # edge_type_ids_np = network['synapses']['edge_type_ids']
        # use the connection_type_ids instead
        edge_type_ids_np = other_v1_utils.connection_type_ids(network).astype(np.int64, copy=False)
        initial_value_np /= voltage_scale[np.asarray(network['node_type_ids'])[indices[:, 0]]]

        _, group_ids_np = np.unique(edge_type_ids_np, return_inverse=True)
        n_groups = int(np.max(group_ids_np) + 1) if group_ids_np.size else 0

        # presort the initial valueby group using the same order as group_ids_np, so that we can directly compare to sorted x_i in the call method without needing to sort y_i there.
        ### 2. Reorder original_indices and initial_value based on sorted edge types
        if group_ids_np.size == 0:
            order_np = np.empty((0,), dtype=np.int64)
            row_splits_np = np.zeros((1,), dtype=np.int64)
            sorted_initial_flat_np = np.empty((0,), dtype=np.float32)
        else:
            if HAS_NUMBA:
                order_np, row_splits_np = _build_group_order_numba(group_ids_np, n_groups)
                sorted_initial_flat_np = _sort_initial_values_by_group_numba(initial_value_np, order_np, row_splits_np)
            else:
                order_np, row_splits_np = _build_group_order_numpy(group_ids_np, n_groups)
                sorted_initial_flat_np = _sort_initial_values_by_group_numpy(initial_value_np, order_np, row_splits_np)

        if order_np.size <= np.iinfo(np.int32).max:
            order_tf = tf.convert_to_tensor(order_np, dtype=tf.int32)
        else:
            order_tf = tf.convert_to_tensor(order_np, dtype=tf.int64)
        row_splits_tf = tf.convert_to_tensor(row_splits_np, dtype=tf.int32)
        sorted_initial_tf = tf.convert_to_tensor(sorted_initial_flat_np, dtype=self._dtype)

        # 3. Build row_splits directly from per-type counts.
        # This avoids RaggedTensor.from_value_rowids -> DenseBincount, which raises
        # under GPU deterministic mode in TF 2.15.
        self.num_unique = tf.constant(n_groups, dtype=tf.int32)
        self._group_indices = tf.RaggedTensor.from_row_splits(order_tf, row_splits_tf, validate=False)
        self._sorted_initial_values = tf.RaggedTensor.from_row_splits(sorted_initial_tf, row_splits_tf, validate=False)

    @tf.function(jit_compile=False) # Do not use jit_compile=True. It uses a lot of memory.
    def __call__(self, x):
        if x.dtype != self._dtype:
            x = tf.cast(x, self._dtype)
        if len(x.shape) > 1 and x.shape[1] == 1:
            x = tf.squeeze(x, axis=1)

        emd_losses = tf.TensorArray(self._dtype, size=self.num_unique)
        for i in tf.range(self.num_unique):
            x_i = tf.gather(x, self._group_indices[i])
            y_i = self._sorted_initial_values[i] # Already presorted during initialization
            emd = tf.reduce_mean(tf.abs(tf.sort(x_i) - y_i))
            emd_losses = emd_losses.write(i, emd)
        reg_loss = tf.reduce_mean(emd_losses.stack())
        return reg_loss * self._strength


def spike_trimming(spikes, pre_delay=50, post_delay=50, trim=True):
    pre = pre_delay or 0
    if trim:
        post = -post_delay if post_delay else None
        spikes = spikes[:, pre:post, :]
    else:
        spikes = spikes[:, pre:, :]
    return spikes

def sample_firing_rates(firing_rates, n_neurons, rnd_seed):
    # Sort the original firing rates
    sorted_firing_rates = np.sort(firing_rates)
    # Calculate the empirical cumulative distribution function (CDF)
    percentiles = np.linspace(0, 1, sorted_firing_rates.size)
    # Generate random uniform values from 0 to 1
    rate_rd = np.random.RandomState(seed=rnd_seed)
    x_rand = rate_rd.uniform(low=0, high=1, size=n_neurons)
    # Use inverse transform sampling: interpolate the uniform values to find the firing rates
    target_firing_rates = np.sort(np.interp(x_rand, percentiles, sorted_firing_rates))
    # target_firing_rates = np.interp(x_rand, percentiles, sorted_firing_rates)
    return target_firing_rates


def resample_sorted_distribution(values, n_samples):
    sorted_values = np.sort(np.asarray(values, dtype=np.float32))
    if n_samples <= 0 or sorted_values.size == 0:
        return np.empty((0,), dtype=np.float32)
    if sorted_values.size == 1:
        return np.full((n_samples,), sorted_values[0], dtype=np.float32)

    source_quantiles = np.linspace(0.0, 1.0, sorted_values.size, dtype=np.float32)
    target_quantiles = np.linspace(0.0, 1.0, n_samples, dtype=np.float32)
    return np.interp(target_quantiles, source_quantiles, sorted_values).astype(np.float32)

def huber_quantile_loss(u, tau, kappa, dtype=tf.float32):
    tau = tf.cast(tau, dtype)
    abs_u = tf.abs(u)
    num = tf.abs(tau - tf.cast(u <= 0, dtype))
    branch_1 = num / (2 * kappa) * tf.square(u)
    branch_2 = num * (abs_u - 0.5 * kappa)
    return tf.where(abs_u <= kappa, branch_1, branch_2)

### To calculate the loss of firing rates between neuron types
def compute_spike_rate_target_loss(rates, target_rates, dtype=tf.float32):
    # TODO: define this function
    # target_rates is a dictionary that contains all the cell types.
    # I should iterate on them, and add the cost for each one at the end.
    # spikes will have a shape of (batch_size, n_steps, n_neurons)
    # rates = tf.reduce_mean(_spikes, (0, 1))
    total_loss = tf.constant(0.0, dtype=dtype)
    num_neurons = tf.constant(0, dtype=tf.int32)
    # if core_mask is not None:
    #     core_neurons_ids = np.where(core_mask)[0]

    for key, value in target_rates.items():
        neuron_ids = value["neuron_ids"]
        if len(neuron_ids) != 0:
            _rate_type = tf.gather(rates, neuron_ids)
            target_rate = value["sorted_target_rates"]
            # if core_mask is not None:
            #     key_core_mask = np.isin(value["neuron_ids"], core_neurons_ids)
            #     neuron_ids =  np.where(key_core_mask)[0]
            #     _rate_type = tf.gather(rates, neuron_ids)
            #     target_rate = value["sorted_target_rates"][key_core_mask]
            # else:
            #     _rate_type = tf.gather(rates, value["neuron_ids"])
            #     target_rate = value["sorted_target_rates"]

            loss_type = compute_spike_rate_distribution_loss(_rate_type, target_rate, dtype=dtype)
            total_loss += tf.reduce_sum(loss_type)
            num_neurons += tf.size(neuron_ids)

    total_loss /= tf.cast(num_neurons, dtype=dtype)

    return total_loss

def compute_spike_rate_distribution_loss(_rates, target_rate, dtype=tf.float32):
    # Firstly we shuffle the current model rates to avoid bias towards a particular tuning angles (inherited from neurons ordering in the network)
    ind = tf.range(target_rate.shape[0])
    rand_ind = tf.random.shuffle(ind)
    _rates = tf.gather(_rates, rand_ind)
    sorted_rate = tf.sort(_rates)
    # u = target_rate - sorted_rate
    u = sorted_rate - target_rate
    n = tf.shape(target_rate)[0]
    tau = (tf.cast(tf.range(n), dtype) + 1) / tf.cast(n, dtype)
    loss = huber_quantile_loss(u, tau, 0.002, dtype=dtype)
    # loss = huber_quantile_loss(u, tau, 0.1, dtype=dtype)

    return loss

def process_neuropixels_data(path=''):
    # Load data
    neuropixels_data_path = 'Neuropixels_data/cortical_metrics_1.4.csv'
    df_all = pd.read_csv(neuropixels_data_path, sep=",")
    # Exc and PV have sufficient number of cells, so we'll filter out non-V1 Exc and PV.
    # SST and VIP are small populations, so let's keep also non-V1 neurons
    exclude = (df_all["cell_type"].isnull() | df_all["cell_type"].str.contains("EXC") | df_all["cell_type"].str.contains("PV")) \
            & (df_all["ecephys_structure_acronym"] != 'VISp')
    df = df_all[~exclude]
    print(f"Original: {df_all.shape[0]} cells,   filtered: {df.shape[0]} cells")

    # Some cells have very large values of RF. They are likely not-good fits, so ignore.
    df.loc[(df["width_rf"] > 100), "width_rf"] = np.nan
    df.loc[(df["height_rf"] > 100), "height_rf"] = np.nan

    # Save the processed table
    df.to_csv('Neuropixels_data/v1_OSI_DSI_DF.csv', sep=" ", index=False)
    # return df

def neuropixels_cell_type_to_cell_type(pop_name):
    if not isinstance(pop_name, str):
        return pop_name
    if ' ' in pop_name:  # This is already new. No need to update.
        return pop_name

    # Convert pop_name in the neuropixels cell type to cell types. E.g, 'EXC_L23' -> 'L2/3 Exc', 'PV_L5' -> 'L5 PV'
    layer = pop_name.split('_')[1]
    class_name = pop_name.split('_')[0]
    if "2" in layer:
        layer = "L2/3"
    elif layer == "L1":
        return "L1 Htr3a"  # special case
    if class_name == "EXC":
        class_name = "Exc"
    if class_name == 'Htr3a':
        class_name = 'VIP'

    return f"{layer} {class_name}"


CELL_TYPE_QUERY_MAPPING = {
    "i1H": "L1 Htr3a",
    "e23": "L2/3 Exc",
    "i23P": "L2/3 PV",
    "i23S": "L2/3 SST",
    "i23V": "L2/3 VIP",
    "e4": "L4 Exc",
    "i4P": "L4 PV",
    "i4S": "L4 SST",
    "i4V": "L4 VIP",
    "e5": "L5 Exc",
    "i5P": "L5 PV",
    "i5S": "L5 SST",
    "i5V": "L5 VIP",
    "e6": "L6 Exc",
    "i6P": "L6 PV",
    "i6S": "L6 SST",
    "i6V": "L6 VIP",
}

CELL_TYPE_ORDER = tuple(CELL_TYPE_QUERY_MAPPING.values())


def _core_mask_to_numpy(core_mask, n_nodes):
    if core_mask is None:
        return np.ones(n_nodes, dtype=bool)
    if hasattr(core_mask, "numpy"):
        core_mask = core_mask.numpy()
    core_mask = np.asarray(core_mask, dtype=bool)
    if core_mask.shape[0] != n_nodes:
        raise ValueError(
            f"core_mask has length {core_mask.shape[0]}, expected {n_nodes}."
        )
    return core_mask


def get_population_neuron_ids(
    network, data_dir="GLIF_network", core_mask=None, reindex_selected=False
):
    pop_names = other_v1_utils.pop_names(network, data_dir=data_dir)
    np_core_mask = _core_mask_to_numpy(core_mask, len(pop_names))
    if reindex_selected:
        selected_ids = np.arange(np.count_nonzero(np_core_mask), dtype=np.int32)
    else:
        selected_ids = np.flatnonzero(np_core_mask)
    selected_pop_names = pop_names[np_core_mask]
    selected_cell_types = np.array(
        [
            other_v1_utils.pop_name_to_cell_type(pop_name, ignore_l5e_subtypes=True)
            for pop_name in selected_pop_names
        ]
    )

    grouped_ids = {}
    for cell_type in CELL_TYPE_ORDER:
        grouped_ids[cell_type] = selected_ids[selected_cell_types == cell_type]
    return grouped_ids


def compare_rate_mask_population_ids(network, data_dir="GLIF_network", core_mask=None):
    """Compare canonical population-id grouping against the legacy mask path.

    This is intentionally kept as a diagnostic helper: training should use
    ``get_population_neuron_ids`` directly, while tests and migration checks can
    use this function to verify that the rate-loss masks have not drifted.
    """
    canonical_ids = get_population_neuron_ids(
        network,
        data_dir=data_dir,
        core_mask=core_mask,
        reindex_selected=False,
    )

    pop_names = other_v1_utils.pop_names(network, data_dir=data_dir)
    np_core_mask = _core_mask_to_numpy(core_mask, len(pop_names))
    selected_ids = np.flatnonzero(np_core_mask)
    selected_pop_names = pop_names[np_core_mask]
    selected_cell_types = np.array(
        [
            other_v1_utils.pop_name_to_cell_type(pop_name, ignore_l5e_subtypes=True)
            for pop_name in selected_pop_names
        ]
    )

    comparison = {}
    for cell_type in CELL_TYPE_ORDER:
        legacy_ids = selected_ids[selected_cell_types == cell_type]
        canonical = np.asarray(canonical_ids[cell_type])
        legacy = np.asarray(legacy_ids)
        comparison[cell_type] = {
            "match": bool(np.array_equal(canonical, legacy)),
            "canonical_count": int(canonical.size),
            "legacy_count": int(legacy.size),
            "canonical_ids": canonical,
            "legacy_ids": legacy,
        }
    return comparison


def compute_rolling_decay_from_sample_ess(batch_size, target_sample_ess, max_decay=0.9999):
    """Compute EMA decay from target effective sample size (ESS) in samples.

    For EMA with decay ``d``, the effective sample size in update steps is
    ``ESS_steps ~= (1 + d) / (1 - d)``. Converting to samples gives
    ``ESS_samples ~= batch_size * ESS_steps``.
    """
    batch_size = float(batch_size)
    target_sample_ess = float(target_sample_ess)
    max_decay = float(max_decay)

    if batch_size <= 0.0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}.")
    if target_sample_ess <= 0.0:
        raise ValueError(
            f"target_sample_ess must be > 0, got {target_sample_ess}."
        )
    if not 0.0 <= max_decay < 1.0:
        raise ValueError(f"max_decay must be in [0, 1), got {max_decay}.")

    ess_steps = target_sample_ess / batch_size
    if ess_steps <= 1.0:
        return 0.0

    decay = (ess_steps - 1.0) / (ess_steps + 1.0)
    return float(np.clip(decay, 0.0, max_decay))


class SpikeRateDistributionTarget:
    """ Instead of regularization, treat it as a target.
        The main difference is that this class will calculate the loss
        for each subtypes of the neurons."""
    def __init__(self, network, stimulus_type='drifting_gratings', rate_cost=.5, pre_delay=None, post_delay=None,
                 data_dir='GLIF_network', core_mask=None, rates_dampening=1.0, seed=42, dtype=tf.float32,
                 neuropixels_df='Neuropixels_data/v1_OSI_DSI_DF.csv', **kwargs):
        self._network = network
        self._rate_cost = rate_cost
        self._pre_delay = pre_delay
        self._post_delay = post_delay
        self._rates_dampening = rates_dampening
        self._core_mask = core_mask
        self._data_dir = data_dir
        self._dtype = dtype
        self._seed = seed
        self._neuropixels_df = neuropixels_df

        # Mapping of stimulus type to neuropixels feature
        # If deprecated arguments are used, map them to stimulus_type
        if kwargs.get('spontaneous_fr'):
            stimulus_type = 'spontaneous'
        elif kwargs.get('natural_images'):
            stimulus_type = 'natural_stimuli'

        if stimulus_type in ['spontaneous', 'gray']:
            self.neuropixels_feature = 'firing_rate_sp'
        elif stimulus_type in ['natural_stimuli', 'natural_images']:
            self.neuropixels_feature = 'firing_rate_ns'
        elif stimulus_type == 'drifting_gratings':
            self.neuropixels_feature = 'Ave_Rate(Hz)'
        else:
            raise ValueError(f"Unknown stimulus_type: {stimulus_type}. Choose among 'spontaneous/gray', 'drifting_gratings', or 'natural_stimuli'.")

        self._target_rates = self.get_neuropixels_firing_rates()

    def get_neuropixels_firing_rates(self):
        """
        Processes neuropixels data to obtain neurons average firing rates.

        Returns:
            dict: Dictionary containing rates and node_type_ids for each population query.
        """
        # Load data
        # neuropixels_data_path = f'Neuropixels_data/v1_OSI_DSI_DF.csv'

        neuropixels_data_path = self._neuropixels_df
        if neuropixels_data_path == 'Neuropixels_data/v1_OSI_DSI_DF.csv':
            if not os.path.exists(neuropixels_data_path):
                process_neuropixels_data(path=neuropixels_data_path)
        else: # just inform the user that the custom file is loading.
            print(f"Using custom neuropixels data file for FR loss: {neuropixels_data_path}")

        # New dataset has Spont_Rate(Hz) instead of firing_rate_sp.
        # if reading firing_rate_sp fails, replace it with Spont_Rate(Hz) and try again.
        features_to_load = ['ecephys_unit_id', 'cell_type', 'firing_rate_sp', 'Ave_Rate(Hz)']
        try:
            np_df = pd.read_csv(neuropixels_data_path, index_col=0, sep=" ", usecols=features_to_load).dropna(how='all')
        except ValueError:
            print(f"Neuropixels data file {neuropixels_data_path} does not contain firing_rate_sp. Using Spont_Rate(Hz) instead.")
            features_to_load = ['ecephys_unit_id', 'cell_type', 'Spont_Rate(Hz)', 'Ave_Rate(Hz)']
            np_df = pd.read_csv(neuropixels_data_path, index_col=0, sep=" ", usecols=features_to_load).dropna(how='all')
            # Rename the column to match the original
            np_df.rename(columns={'Spont_Rate(Hz)': 'firing_rate_sp'}, inplace=True)
        # Ensure they use the new names
        np_df["cell_type"] = np_df["cell_type"].apply(neuropixels_cell_type_to_cell_type)
        type_rates_dict = {
            cell_type: np.append(subdf[self.neuropixels_feature].dropna().values / 1000, 0)
            for cell_type, subdf in np_df.groupby("cell_type")
        }
        population_ids = get_population_neuron_ids(
            self._network, data_dir=self._data_dir, core_mask=self._core_mask
        )

        target_firing_rates = {}
        for cell_type in CELL_TYPE_ORDER:
            rates = type_rates_dict.get(cell_type, np.array([0.0], dtype=np.float32))
            neuron_ids = population_ids[cell_type]
            type_n_neurons = len(neuron_ids)
            target_firing_rates[cell_type] = {
                "rates": rates,
                "neuron_ids": tf.convert_to_tensor(neuron_ids, dtype=tf.int32),
                "sorted_target_rates": tf.convert_to_tensor(
                    self._rates_dampening
                    * sample_firing_rates(rates, type_n_neurons, self._seed),
                    dtype=self._dtype,
                ),
            }

        return target_firing_rates

    def rates_per_sample_from_spikes(self, spikes, trim=True):
        """Return full-population rates with shape ``[batch, neurons]``."""
        spikes = spike_trimming(
            spikes,
            pre_delay=self._pre_delay,
            post_delay=self._post_delay,
            trim=trim,
        )
        if spikes.dtype != self._dtype:
            spikes = tf.cast(spikes, self._dtype)
        return tf.reduce_mean(spikes, axis=1)

    def rates_from_spikes(self, spikes, trim=True):
        """Return full-population rates averaged over samples and time."""
        rates_per_sample = self.rates_per_sample_from_spikes(spikes, trim=trim)
        return tf.reduce_mean(rates_per_sample, axis=0)

    def loss_from_rates(self, rates):
        """Evaluate the target from a full-population rate vector."""
        if rates.dtype != self._dtype:
            rates = tf.cast(rates, self._dtype)
        reg_loss = compute_spike_rate_target_loss(
            rates, self._target_rates, dtype=self._dtype
        )
        return reg_loss * self._rate_cost

    def __call__(self, spikes, trim=True):
        return self.loss_from_rates(self.rates_from_spikes(spikes, trim=trim))

# class SpikeRateDistributionRegularization:
#     def __init__(self, target_rates, rate_cost=0.5, dtype=tf.float32):
#         self._rate_cost = rate_cost
#         self._target_rates = target_rates
#         self._dtype = dtype

#     def __call__(self, spikes):
#         reg_loss = (
#             compute_spike_rate_distribution_loss(spikes, self._target_rates, dtype=self._dtype)
#             * self._rate_cost
#         )
#         reg_loss = tf.reduce_sum(reg_loss)

#         return reg_loss

class SynchronizationLoss(Layer):
    def __init__(self, network, stimulus_type='drifting_gratings', sync_cost=10., t_start=0., t_end=0.5, n_samples=50, neuropixels_data_dir='Synchronization_data',
                 data_dir='GLIF_network', dtype=tf.float32, core_mask=None, seed=42, **kwargs):
        super(SynchronizationLoss, self).__init__(dtype=dtype, **kwargs)

        # Handle legacy 'session' argument if passed via kwargs
        if 'session' in kwargs:
            _session = kwargs.get('session')
            if _session == 'spont':
                stimulus_type = 'spontaneous'
            elif _session == 'evoked':
                stimulus_type = 'drifting_gratings'

        if stimulus_type in ['spontaneous', 'gray']:
            session = 'spont'
        elif stimulus_type == 'drifting_gratings':
            session = 'evoked'
        else:
            raise ValueError(f"Unknown stimulus_type: {stimulus_type}. Choose among 'spontaneous', 'gray', or 'drifting_gratings'.")

        self._sync_cost = sync_cost
        self._t_start = t_start
        self._t_end = t_end
        self._t_start_seconds = int(t_start * 1000)
        self._t_end_seconds = int(t_end * 1000)
        self._data_dir = data_dir
        self._neuropixels_data_dir = neuropixels_data_dir
        self._dtype = dtype
        self._n_samples = n_samples
        self._base_seed = seed

        pop_names = other_v1_utils.pop_names(network, data_dir=self._data_dir)
        node_ei = np.array([pop_name[0] for pop_name in pop_names])
        excitatory_mask = node_ei == 'e'
        if core_mask is not None:
            core_mask = tf.get_static_value(core_mask)
            if core_mask is None:
                raise ValueError(
                    "SynchronizationLoss core_mask must be statically known."
                )
            core_mask = np.asarray(core_mask, dtype=bool)
            if core_mask.shape != excitatory_mask.shape:
                raise ValueError(
                    "SynchronizationLoss core_mask has shape "
                    f"{core_mask.shape}, expected {excitatory_mask.shape}."
                )
            excitatory_mask &= core_mask

        self._core_excitatory_mask = tf.constant(excitatory_mask, dtype=tf.bool)
        self.node_id_e = tf.range(
            np.count_nonzero(excitatory_mask), dtype=tf.int32
        )
        # Pre-define bin sizes (same as experimental data)
        bin_sizes = np.logspace(-3, 0, 20)
        # using the simulation length, limit bin_sizes to define at least 2 bins
        bin_sizes_mask = bin_sizes < (self._t_end - self._t_start) / 2
        bin_sizes = bin_sizes[bin_sizes_mask]
        self._bin_sizes_ms = tuple(max(1, int(round(v * 1000.0))) for v in bin_sizes)
        self._bin_sizes_ms_tf = tf.constant(self._bin_sizes_ms, dtype=tf.int32)
        self._epsilon_tf = tf.constant(1e-7, dtype=self._dtype)  # Small constant to avoid division by zero

        # Load the experimental data
        duration = str(int((t_end - t_start) * 1000))
        experimental_data_path = os.path.join(
            self._neuropixels_data_dir,
            'Fano_factor_v1',
            f'v1_fano_running_{duration}ms_{session}.npy',
        )
        # experimental_data_path = os.path.join(data_dir, f'all_fano_300ms_{session}.npy')
        assert os.path.exists(experimental_data_path), f'File not found: {experimental_data_path}'
        experimental_fanos = np.load(experimental_data_path, allow_pickle=True)
        experimental_fanos_mean = np.nanmean(experimental_fanos[:, bin_sizes_mask], axis=0)
        self.experimental_fanos_mean = tf.constant(experimental_fanos_mean, dtype=self._dtype)

    def pop_fano_tf(self, spikes):
        spikes = tf.expand_dims(spikes, axis=-1)
        fanos = tf.TensorArray(dtype=self._dtype, size=len(self._bin_sizes_ms))
        for i, bin_size in enumerate(self._bin_sizes_ms):
            # Use convolution for efficient binning
            kernel = tf.ones((bin_size, 1, 1), dtype=self._dtype)
            convolved = tf.nn.conv1d(spikes, kernel, stride=bin_size, padding="VALID")
            sp_counts = tf.squeeze(convolved, axis=-1)  # Shape: (60, new_width)
            # Compute mean and variance of spike counts
            mean_count = tf.reduce_mean(sp_counts, axis=1)
            var_count = tf.math.reduce_variance(sp_counts, axis=1)
            mean_count = tf.maximum(mean_count, self._epsilon_tf)
            # fanos.append(tf.reduce_mean(var_count / mean_count))
            fano_per_sample = var_count / mean_count  # => [n_samples]
            fano = tf.reduce_mean(fano_per_sample)
            fanos = fanos.write(i, fano)

        return fanos.stack()


    def __call__(self, spikes, trim=True):

        spikes = tf.boolean_mask(spikes, self._core_excitatory_mask, axis=2)

        if trim:
            spikes = spikes[:, self._t_start_seconds:self._t_end_seconds, :]
        duration_ms = tf.cast(tf.shape(spikes)[1], tf.int32)
        bin_limit_ms = duration_ms // 2
        bin_sizes_mask = self._bin_sizes_ms_tf < bin_limit_ms
        experimental_fanos_mean = tf.boolean_mask(self.experimental_fanos_mean, bin_sizes_mask)

        # if spikes.dtype != self._dtype:
        #     spikes = tf.cast(spikes, self._dtype)
        # choose random trials to sample from (usually we only have 1 trial to sample from)
        n_trials = tf.shape(spikes)[0]
        # increase the base seed to avoid the same random neurons to be selected in every instantiation of the class
        self._base_seed += 1
        sample_trials = tf.random.uniform([self._n_samples], minval=0, maxval=n_trials, dtype=tf.int32, seed=self._base_seed)
        # Generate sample counts with a normal distribution
        sample_size = 70
        sample_std = 30
        sample_counts = tf.cast(tf.random.normal([self._n_samples], mean=sample_size, stddev=sample_std, seed=self._base_seed), tf.int32)
        sample_counts = tf.clip_by_value(sample_counts, clip_value_min=15, clip_value_max=tf.shape(self.node_id_e)[0]) # clip the values to be between 15 and 14423
        # Randomize the neuron ids
        shuffled_e_ids = tf.random.shuffle(self.node_id_e, seed=self._base_seed)
        selected_spikes_sample = tf.TensorArray(spikes.dtype, size=self._n_samples)
        previous_id = tf.constant(0, dtype=tf.int32)
        for i in tf.range(self._n_samples):
            sample_num = sample_counts[i] # 40 #68
            sample_trial = sample_trials[i] # 0
            ## randomly choose sample_num ids from self.node_id_e with replacement
            ## sample_ids = tf.random.shuffle(self.node_id_e)[:sample_num]
            ## randomly choose sample_num ids from shuffled_ids without replacement
            if previous_id + sample_num > tf.size(shuffled_e_ids):
                # shuffled_e_ids = tf.random.shuffle(self.node_id_e, seed=self._base_seed)
                shuffled_e_ids = tf.random.shuffle(shuffled_e_ids, seed=self._base_seed)
                previous_id = tf.constant(0, dtype=tf.int32)
            sample_ids = shuffled_e_ids[previous_id:previous_id+sample_num]
            previous_id += sample_num

            selected_spikes = tf.reduce_sum(tf.gather(spikes[sample_trial], sample_ids, axis=1), axis=-1)
            selected_spikes_sample = selected_spikes_sample.write(i, selected_spikes)

        selected_spikes_sample = selected_spikes_sample.stack()
        if selected_spikes_sample.dtype != self._dtype:
            selected_spikes_sample = tf.cast(selected_spikes_sample, self._dtype)

        fanos_mean = self.pop_fano_tf(selected_spikes_sample)
        fanos_mean = tf.boolean_mask(fanos_mean, bin_sizes_mask)
        # # Calculate MSE between the experimental and calculated Fano factors
        def compute_mse():
            return tf.reduce_mean(tf.square(experimental_fanos_mean - fanos_mean))
        mse_loss = tf.cond(
            tf.size(experimental_fanos_mean) > 0,
            compute_mse,
            lambda: tf.constant(0.0, dtype=self._dtype),
        )
        # # Calculate the synchronization loss

        return self._sync_cost * mse_loss


class VoltageRegularization:
    def __init__(self, cell, voltage_cost=1e-5, dtype=tf.float32, core_mask=None, penalty_mode="range"):
        """
        Voltage regularization with two penalty modes.

        Args:
            cell: The cell model
            voltage_cost: Regularization coefficient
            dtype: TensorFlow data type
            core_mask: Boolean mask for selecting subset of neurons
            penalty_mode: Either "range" or "threshold"
                - "range": Penalizes voltages outside [0, 1] range
                - "threshold": Penalizes distance from threshold (1.0)
        """
        # Keep the scalar coefficient in fp32 to avoid fp16 overflow/underflow during backprop.
        self._voltage_cost = tf.constant(voltage_cost, dtype=tf.float32)
        self._cell = cell
        self._dtype = dtype
        self._penalty_mode = penalty_mode
        self._core_mask = core_mask

    @tf.function(jit_compile=True)
    def _safe_global_mean(self, penalty):
        """
        Compute a global mean without triggering fp16 MeanGrad overflow.

        Why this is needed:
        - tf.reduce_mean in fp16 builds a gradient scale factor 1/N.
        - If N > 65504, TensorFlow casts N to fp16 and overflows, causing RuntimeWarning.

        Strategy:
        - Keep elementwise-heavy math in fp16.
        - Reduce batch/time first in fp16 (small divisors).
        - Reduce the final neuron axis in fp32 (small intermediate tensor).
        """
        penalty = tf.reduce_mean(penalty, axis=0)  # divide by batch size (tipically  < 65504)
        penalty = tf.reduce_mean(penalty, axis=0)  # divide by sequence length (tipically  < 65504)
        return tf.reduce_mean(tf.cast(penalty, tf.float32), axis=0)  # divide by number of neurons (tipically > 65504, thus in fp32)

    @tf.function(jit_compile=True)
    def _compute_range_loss(self, voltages):
        """
        JIT-compiled range loss computation.

        Fuses all operations into a single kernel, avoiding intermediate tensor allocations.
        This is ~2-3x faster than the unfused version for large tensors.
        """
        # Equivalent single-branch form of max(0, v-1)^2 + max(0, -v)^2.
        # Keeping it as one branch can reduce intermediate tensor pressure.
        penalty = tf.square(tf.nn.relu(tf.abs(voltages - 0.5) - 0.5))
        return self._safe_global_mean(penalty)

    @tf.function(jit_compile=True)
    def _compute_threshold_loss(self, voltages):
        """JIT-compiled threshold loss computation."""
        penalty = tf.square(voltages - 1.0)
        return self._safe_global_mean(penalty)

    def __call__(self, voltages):

        if self._core_mask is not None:
            voltages = tf.boolean_mask(voltages, self._core_mask, axis=2)

        # No need to cast voltages to self._dtype for the loss computation, since the loss is computed in fp32 anyway. Just ensure it's in a floating point format.
        # if voltages.dtype != self._dtype:
        #     voltages = tf.cast(voltages, self._dtype)

        if self._penalty_mode == "range":
            voltage_loss = self._compute_range_loss(voltages)
        else:  # threshold mode
            voltage_loss = self._compute_threshold_loss(voltages)

        return voltage_loss * self._voltage_cost


class CustomMeanLayer(Layer):
    def call(self, inputs):
        spike_rates, mask = inputs
        masked_data = tf.boolean_mask(spike_rates, mask)
        return tf.reduce_mean(masked_data)


class OrientationSelectivityLoss:
    def __init__(self, network=None, osi_cost=1e-5, pre_delay=None, post_delay=None, dtype=tf.float32,
                 core_mask=None, method="crowd_osi", subtraction_ratio=1.0,
                 neuropixels_df="Neuropixels_data/v1_OSI_DSI_DF.csv", data_dir='',
                 rolling_decay=0.5, rolling_epsilon=1e-6,
                 rolling_target_sample_ess=80.0, rolling_batch_size=None,
                 rolling_max_decay=0.9999,
                 rolling_gradient_correction=True,
                 rolling_max_gradient_scale=20.0,
                 rolling_warmup=True):

        self._network = network
        self._osi_cost = osi_cost
        self._pre_delay = pre_delay
        self._post_delay = post_delay
        self._dtype = dtype
        self._core_mask = core_mask
        self._method = method
        self._subtraction_ratio = subtraction_ratio  # only for crowd_spikes method
        self._tf_pi = tf.constant(np.pi, dtype=dtype)
        self._neuropixels_df = neuropixels_df
        self.data_dir = data_dir
        self._rolling_gradient_correction = bool(rolling_gradient_correction)
        self._rolling_warmup = bool(rolling_warmup)
        self._rolling_target_sample_ess = tf.constant(
            float(rolling_target_sample_ess), dtype=self._dtype
        )
        self._rolling_config_batch_size = tf.constant(
            float(rolling_batch_size) if rolling_batch_size is not None else 1.0,
            dtype=self._dtype,
        )
        rolling_max_gradient_scale = float(rolling_max_gradient_scale)
        if rolling_max_gradient_scale <= 0.0:
            raise ValueError(
                "rolling_max_gradient_scale must be > 0, "
                f"got {rolling_max_gradient_scale}."
            )
        self._rolling_max_gradient_scale = tf.constant(
            rolling_max_gradient_scale, dtype=self._dtype
        )
        self._adaptative_scale_min = tf.constant(0.4, dtype=self._dtype)
        self._adaptative_scale_max = tf.constant(1.0, dtype=self._dtype)
        self._adaptative_shrink_k = tf.constant(20.0, dtype=self._dtype)
        if (
            self._core_mask is not None
            and self._method in (
                "crowd_spikes",
                "crowd_osi",
                "rolling_osi_emd",
                "adaptative_crowd_osi",
            )
        ):
            self.np_core_mask = self._core_mask.numpy()
            core_tuning_angles = network['tuning_angle'][self.np_core_mask]
            self._tuning_angles = tf.constant(core_tuning_angles, dtype=dtype)
        else:
            self._tuning_angles = tf.constant(network['tuning_angle'], dtype=dtype)

        if self._method == "neuropixels_fr":
            self._layer_info = other_v1_utils.get_layer_info(network)  # needed for neuropixels_fr method
            # the layer_info should be a dictionary that contains
            # the cell id of the corresponding layer.
            # the keys should be something like "EXC_L23" or "PV_L5"

        elif self._method == "crowd_osi":
            self._initialize_crowd_targets()
            self._rolling_epsilon = tf.constant(rolling_epsilon, dtype=self._dtype)

        elif self._method == "adaptative_crowd_osi":
            self._initialize_crowd_targets(include_experimental_counts=True)
            resolved_decay = self._resolve_rolling_decay(
                rolling_decay=rolling_decay,
                rolling_batch_size=rolling_batch_size,
                rolling_target_sample_ess=rolling_target_sample_ess,
                rolling_max_decay=rolling_max_decay,
            )
            self._rolling_decay = tf.constant(resolved_decay, dtype=self._dtype)
            self._rolling_one_minus_decay = tf.constant(1.0 - resolved_decay, dtype=self._dtype)
            self._rolling_epsilon = tf.constant(rolling_epsilon, dtype=self._dtype)
            self._initialize_rolling_state_variables(len(self._tuning_angles))

        elif self._method == "rolling_osi_emd":
            resolved_decay = self._resolve_rolling_decay(
                rolling_decay=rolling_decay,
                rolling_batch_size=rolling_batch_size,
                rolling_target_sample_ess=rolling_target_sample_ess,
                rolling_max_decay=rolling_max_decay,
            )
            self._rolling_decay = tf.constant(resolved_decay, dtype=self._dtype)
            self._rolling_one_minus_decay = tf.constant(1.0 - resolved_decay, dtype=self._dtype)
            self._rolling_epsilon = tf.constant(rolling_epsilon, dtype=self._dtype)
            self._initialize_rolling_osi_emd_targets()

    def _uses_rolling_state(self):
        return self._method in ("rolling_osi_emd", "adaptative_crowd_osi")

    def _resolve_rolling_decay(
        self,
        rolling_decay,
        rolling_batch_size,
        rolling_target_sample_ess,
        rolling_max_decay,
    ):
        rolling_decay = float(rolling_decay)
        if rolling_decay < 0.0:
            if rolling_batch_size is None:
                raise ValueError(
                    "rolling_batch_size must be provided when rolling_decay < 0 "
                    "(auto-decay mode)."
                )
            resolved_decay = compute_rolling_decay_from_sample_ess(
                batch_size=rolling_batch_size,
                target_sample_ess=rolling_target_sample_ess,
                max_decay=rolling_max_decay,
            )
            print(
                f"[{self._method}] Auto-decay enabled: "
                f"target_sample_ess={float(rolling_target_sample_ess):.3f}, "
                f"batch_size={int(rolling_batch_size)}, "
                f"resolved_decay={resolved_decay:.6f}"
            )
            return resolved_decay

        if not 0.0 <= rolling_decay < 1.0:
            raise ValueError(
                "rolling_decay must be in [0, 1), or < 0 to enable auto-decay. "
                f"Got {rolling_decay}."
            )
        return rolling_decay

    def calculate_delta_angle(self, stim_angle, tuning_angle):
        # angle unit is degrees.
        # this function calculates the difference between stim_angle and tuning_angle,
        # but it is fine to have the opposite direction.
        # so, delta angle is always between -90 and 90.
        # they are both vector, so dimension matche is needed.
        # stim_angle is a length of batch size
        # tuning_angle is a length of n_neurons

        # delta_angle = stim_angle - tuning_angle
        delta_angle = tf.expand_dims(stim_angle, axis=1) - tuning_angle
        delta_angle = tf.where(delta_angle > 90, delta_angle - 180, delta_angle)
        delta_angle = tf.where(delta_angle < -90, delta_angle + 180, delta_angle)
        # # do it twice to make sure everything is between -90 and 90.
        delta_angle = tf.where(delta_angle > 90, delta_angle - 180, delta_angle)
        delta_angle = tf.where(delta_angle < -90, delta_angle + 180, delta_angle)

        return delta_angle

    def get_neuropixels_osi_dsi(self):
        """
        Processes neuropixels data to obtain neurons average firing rates.

        Returns:
            dict: Dictionary containing rates and node_type_ids for each population query.
        """
        # Load data
        # neuropixels_data_path = f'Neuropixels_data/v1_OSI_DSI_DF.csv'
        neuropixels_data_path = self._neuropixels_df
        # if the default one is specified and the file doesn't exist, process the data
        if neuropixels_data_path == "Neuropixels_data/v1_OSI_DSI_DF.csv":
            if not os.path.exists(neuropixels_data_path):
                process_neuropixels_data(path=neuropixels_data_path)
        else:
            print(f"Using custom neuropixels data file for OSI/DSI loss: {neuropixels_data_path}")
        features_to_load = ['ecephys_unit_id', 'cell_type', 'OSI', 'DSI', "Ave_Rate(Hz)", "max_mean_rate(Hz)"]
        osi_dsi_df = pd.read_csv(neuropixels_data_path, index_col=0, sep=" ", usecols=features_to_load).dropna(how='all')

        nonresponding = osi_dsi_df["max_mean_rate(Hz)"] < 0.5
        osi_dsi_df.loc[nonresponding, ["OSI", "DSI"]] = np.nan
        osi_dsi_df = osi_dsi_df[osi_dsi_df["Ave_Rate(Hz)"] != 0]
        osi_dsi_df.dropna(inplace=True)
        osi_dsi_df["cell_type"] = osi_dsi_df["cell_type"].apply(neuropixels_cell_type_to_cell_type)
        grouped_np = osi_dsi_df.groupby("cell_type")
        osi_target = grouped_np['OSI'].mean()
        dsi_target = grouped_np['DSI'].mean()
        osi_values = {
            cell_type: np.sort(subdf['OSI'].dropna().to_numpy(dtype=np.float32))
            for cell_type, subdf in grouped_np
        }
        dsi_values = {
            cell_type: np.sort(subdf['DSI'].dropna().to_numpy(dtype=np.float32))
            for cell_type, subdf in grouped_np
        }

        cell_ids = get_population_neuron_ids(
            self._network,
            data_dir=self.data_dir,
            core_mask=self._core_mask,
            reindex_selected=True,
        )

        # osi_target = osi_df.groupby("cell_type")['OSI'].mean()
        # osi_target = osi_df.groupby("cell_type")['OSI'].median()
        # osi_df.groupby("cell_type")['OSI'].median()
        # convert to dict
        osi_dsi_exp_dict = {
            key: {
                "OSI": val,
                "DSI": dsi_target[key],
                "OSI_values": osi_values.get(key, np.empty((0,), dtype=np.float32)),
                "DSI_values": dsi_values.get(key, np.empty((0,), dtype=np.float32)),
                "ids": cell_ids.get(key, np.empty(0, dtype=np.int32)),
            }
            for key, val in osi_target.to_dict().items()
        }

        return osi_dsi_exp_dict

    def _initialize_crowd_targets(self, include_experimental_counts=False):
        self._target_osi_dsi = self.get_neuropixels_osi_dsi()
        self._min_rates_threshold = tf.constant(0.0005, dtype=self._dtype)

        n_nodes = len(self._tuning_angles)
        node_type_ids = np.zeros(n_nodes, dtype=np.int32)
        osi_target_values = []
        dsi_target_values = []
        cell_type_count = []
        experimental_cell_type_count = []

        for node_type_id, (_key, value) in enumerate(self._target_osi_dsi.items()):
            node_ids = value["ids"]
            osi_target_values.append(value["OSI"])
            dsi_target_values.append(value["DSI"])
            cell_type_count.append(len(node_ids))
            if include_experimental_counts:
                experimental_cell_type_count.append(len(value["OSI_values"]))
            # update the ndoe_type_ids tensor in positions node_ids with the node_type_id
            # self.node_type_ids = tf.tensor_scatter_nd_update(self.node_type_ids, indices=tf.expand_dims(node_ids, axis=1), updates=tf.fill(tf.shape(node_ids), node_type_id))
            node_type_ids[node_ids] = node_type_id

        self.osi_target_values = tf.constant(osi_target_values, dtype=self._dtype)
        self.dsi_target_values = tf.constant(dsi_target_values, dtype=self._dtype)
        self.cell_type_count = tf.constant(cell_type_count, dtype=self._dtype)
        self.node_type_ids = tf.constant(node_type_ids, dtype=tf.int32)
        self._n_node_types = len(self._target_osi_dsi)
        if include_experimental_counts:
            self.experimental_cell_type_count = tf.constant(
                experimental_cell_type_count, dtype=self._dtype
            )

    def _initialize_rolling_state_variables(self, n_nodes):
        zeros = tf.zeros((n_nodes,), dtype=self._dtype)
        self._rolling_ori_real = tf.Variable(zeros, trainable=False, name="rolling_osi_ori_real")
        self._rolling_ori_imag = tf.Variable(zeros, trainable=False, name="rolling_osi_ori_imag")
        self._rolling_dir_real = tf.Variable(zeros, trainable=False, name="rolling_osi_dir_real")
        self._rolling_dir_imag = tf.Variable(zeros, trainable=False, name="rolling_osi_dir_imag")
        self._rolling_denominator = tf.Variable(zeros, trainable=False, name="rolling_osi_denominator")
        self._rolling_weight_sum = tf.Variable(
            tf.constant(0.0, dtype=self._dtype),
            trainable=False,
            name="rolling_osi_weight_sum",
        )
        self._rolling_weight_sq_sum = tf.Variable(
            tf.constant(0.0, dtype=self._dtype),
            trainable=False,
            name="rolling_osi_weight_sq_sum",
        )

    def _initialize_rolling_osi_emd_targets(self):
        self._target_osi_dsi = self.get_neuropixels_osi_dsi()
        self._min_rates_threshold = tf.constant(0.0005, dtype=self._dtype)

        n_nodes = len(self._tuning_angles)
        group_indices = []
        osi_target_values = []
        dsi_target_values = []
        cell_type_count = []

        for cell_type in CELL_TYPE_ORDER:
            value = self._target_osi_dsi.get(cell_type)
            if value is None:
                continue

            node_ids = np.asarray(value["ids"], dtype=np.int32)
            exp_osi = np.asarray(value["OSI_values"], dtype=np.float32)
            exp_dsi = np.asarray(value["DSI_values"], dtype=np.float32)
            if node_ids.size == 0 or exp_osi.size == 0 or exp_dsi.size == 0:
                continue

            group_indices.append(node_ids)
            osi_target_values.append(resample_sorted_distribution(exp_osi, node_ids.size))
            dsi_target_values.append(resample_sorted_distribution(exp_dsi, node_ids.size))
            cell_type_count.append(float(node_ids.size))

        if not group_indices:
            raise ValueError(
                "rolling_osi_emd requires at least one cell type with both model neurons "
                "and Neuropixels OSI/DSI samples."
            )

        row_splits = np.zeros(len(group_indices) + 1, dtype=np.int32)
        row_splits[1:] = np.cumsum([group.size for group in group_indices], dtype=np.int32)
        flat_indices = np.concatenate(group_indices, axis=0).astype(np.int32, copy=False)
        flat_osi_targets = np.concatenate(osi_target_values, axis=0).astype(np.float32, copy=False)
        flat_dsi_targets = np.concatenate(dsi_target_values, axis=0).astype(np.float32, copy=False)

        row_splits_tf = tf.convert_to_tensor(row_splits, dtype=tf.int32)
        self._emd_group_indices = tf.RaggedTensor.from_row_splits(
            tf.convert_to_tensor(flat_indices, dtype=tf.int32),
            row_splits_tf,
            validate=False,
        )
        self._osi_target_distributions = tf.RaggedTensor.from_row_splits(
            tf.convert_to_tensor(flat_osi_targets, dtype=self._dtype),
            row_splits_tf,
            validate=False,
        )
        self._dsi_target_distributions = tf.RaggedTensor.from_row_splits(
            tf.convert_to_tensor(flat_dsi_targets, dtype=self._dtype),
            row_splits_tf,
            validate=False,
        )
        self.cell_type_count = tf.constant(cell_type_count, dtype=self._dtype)
        self._n_node_types = tf.constant(len(group_indices), dtype=tf.int32)

        self._initialize_rolling_state_variables(n_nodes)

    def get_rolling_state(self):
        if not self._uses_rolling_state():
            return {}
        return {
            "ori_real": self._rolling_ori_real.numpy(),
            "ori_imag": self._rolling_ori_imag.numpy(),
            "dir_real": self._rolling_dir_real.numpy(),
            "dir_imag": self._rolling_dir_imag.numpy(),
            "denominator": self._rolling_denominator.numpy(),
            "weight_sum": self._rolling_weight_sum.numpy(),
            "weight_sq_sum": self._rolling_weight_sq_sum.numpy(),
        }

    def set_rolling_state(self, state):
        if not self._uses_rolling_state():
            if state:
                raise ValueError("Cannot restore rolling state for a non-rolling OSI loss.")
            return False

        if not state:
            return False

        state_specs = (
            ("ori_real", self._rolling_ori_real),
            ("ori_imag", self._rolling_ori_imag),
            ("dir_real", self._rolling_dir_real),
            ("dir_imag", self._rolling_dir_imag),
            ("denominator", self._rolling_denominator),
        )
        for key, variable in state_specs:
            if key not in state:
                raise ValueError(f"Rolling state is missing key '{key}'.")
            value = np.asarray(state[key], dtype=self._dtype.as_numpy_dtype)
            if tuple(value.shape) != tuple(variable.shape):
                raise ValueError(
                    f"Rolling state '{key}' has shape {value.shape}, "
                    f"expected {tuple(variable.shape)}."
                )
            variable.assign(value)
        optional_state_specs = (
            ("weight_sum", self._rolling_weight_sum),
            ("weight_sq_sum", self._rolling_weight_sq_sum),
        )
        for key, variable in optional_state_specs:
            if key not in state:
                continue
            value = np.asarray(state[key], dtype=self._dtype.as_numpy_dtype)
            if tuple(value.shape) != tuple(variable.shape):
                raise ValueError(
                    f"Rolling state '{key}' has shape {value.shape}, "
                    f"expected {tuple(variable.shape)}."
                )
            variable.assign(value)

        if "weight_sum" not in state or "weight_sq_sum" not in state:
            steady_ess_steps = (
                (1.0 + self._rolling_decay)
                / tf.maximum(1.0 - self._rolling_decay, self._rolling_epsilon)
            )
            steady_weight_sq_sum = 1.0 / tf.maximum(
                steady_ess_steps * self._rolling_config_batch_size,
                self._rolling_epsilon,
            )
            self._rolling_weight_sum.assign(tf.constant(1.0, dtype=self._dtype))
            self._rolling_weight_sq_sum.assign(steady_weight_sq_sum)
        return True

    def vonmises_model_fr(self, structure, population):
        from scipy.stats import vonmises
        paramdic = self._von_mises_params
        _params = paramdic[structure][population]
        if len(_params) == 4:
            mu, kappa, a, b = _params
        vonmises_pdf = vonmises(kappa, loc=mu).pdf

        angles = np.deg2rad(np.arange(-85, 86, 10)) * 2  # *2 needed to make it proper model
        model_fr = a + b * vonmises_pdf(angles)

        return model_fr

    def neuropixels_fr_loss(self, spikes, angle):
        # if the trget fr is not set, construct them
        if not hasattr(self, "_target_frs"):

            # self._von_mises_params = np.load("GLIF_network/param_dict_orientation.npy")
            # pickle instead
            with open(f"{self.data_dir}/param_dict_orientation.pkl", 'rb') as f:
                self._von_mises_params = pkl.load(f)
            # get the model values with 10 degree increments
            structure = "VISp"
            self._target_frs = {}
            for key in self._layer_info.keys():
                self._target_frs[key] = self.vonmises_model_fr(structure, key)
                # TODO: convert it to tensor if needed.

        # assuming 1 ms bins
        spike_rates = tf.reduce_mean(spikes, axis=[0, 1]) / spikes.shape[1] * 1000
        angle_bins = tf.constant(np.arange(-90, 91, 10), dtype=self._dtype)
        nbins = angle_bins.shape[0] - 1
        # now, process each layer
        # losses = tf.TensorArray(tf.float32, size=len(self._layer_info))
        losses = []
        delta_angle = self.calculate_delta_angle(angle, self._tuning_angles)
        custom_mean_layer = CustomMeanLayer()

        for key, value in self._layer_info.items():
            # first, calculate delta_angle

            # rates = tf.TensorArray(tf.float32, size=nbins)
            rates_list = []
            for i in range(nbins):
                mask = (delta_angle >= angle_bins[i]) & (delta_angle < angle_bins[i+1])
                # take the intersection with core mask
                mask = tf.logical_and(mask, self._core_mask)
                mask = tf.logical_and(mask, value)
                # mask = mask.flatten()
                # doesn't work.
                mask = tf.reshape(mask, [-1])
                mean_val = custom_mean_layer([spike_rates, mask])
                # rates_ = rates.write(i, mean_val)
                rates_list.append(mean_val)
                # rates = rates.write(i, tf.reduce_mean(tf.boolean_mask(spike_rates, mask)))

            # calculate the loss
            # rates = rates.stack()
            rates = tf.stack(rates_list)
            loss = tf.reduce_mean(tf.square(rates - self._target_frs[key]))
            # if key == "EXC_L6":
                # print the results!
                # tf.print("Layer6: ", rates)
                # tf.print("target: ", self._target_frs[key])
            # losses = losses.write(i, loss)
            losses.append(loss)

        # final_loss = tf.reduce_sum(losses.stack()) * self._osi_cost
        final_loss = tf.reduce_mean(tf.stack(losses)) * self._osi_cost
        return final_loss

    def crowd_spikes_loss(self, spikes, angle):
        # I need to access the tuning angle. of all the neurons.
        angle = tf.cast(angle, self._dtype)

        if self._core_mask is not None:
            spikes = tf.boolean_mask(spikes, self._core_mask, axis=2)

        delta_angle = self.calculate_delta_angle(angle, self._tuning_angles)
        # sum spikes in _z, and multiply with delta_angle.
        mean_spikes = tf.reduce_mean(spikes, axis=[1])
        mean_angle = mean_spikes * delta_angle
        # Here, the expected value with random firing to subtract
        # (this prevents the osi loss to drive the firing rates to go to zero.)
        expected_sum_angle = tf.reduce_mean(mean_spikes) * 45

        angle_loss = tf.reduce_mean(tf.abs(mean_angle)) - expected_sum_angle * self._subtraction_ratio

        return angle_loss * self._osi_cost

    @tf.function(jit_compile=True)
    def _compute_crowd_moment_core(self, rates, radians_delta_angle, batch_size, node_type_ids, n_node_types):
        """Return phase-anchored crowd Fourier moments per cell type."""
        cos_2x = tf.math.cos(2.0 * radians_delta_angle)
        sin_2x = tf.math.sin(2.0 * radians_delta_angle)
        cos_x = tf.math.cos(radians_delta_angle)
        sin_x = tf.math.sin(radians_delta_angle)

        batch_offsets = tf.range(batch_size, dtype=node_type_ids.dtype) * n_node_types
        segment_ids = node_type_ids[tf.newaxis, :] + batch_offsets[:, tf.newaxis]
        segment_ids_flat = tf.reshape(segment_ids, [-1])
        num_segments = batch_size * n_node_types

        denominator = tf.math.unsorted_segment_mean(
            tf.reshape(rates, [-1]), segment_ids_flat, num_segments=num_segments
        )
        denominator = tf.reshape(denominator, [batch_size, n_node_types])
        denominator = tf.maximum(denominator, self._min_rates_threshold)

        def _normalized_segment_mean(weighted_values):
            numerator = tf.math.unsorted_segment_mean(
                tf.reshape(weighted_values, [-1]),
                segment_ids_flat,
                num_segments=num_segments,
            )
            numerator = tf.reshape(numerator, [batch_size, n_node_types])
            return numerator / denominator

        osi_real = _normalized_segment_mean(rates * cos_2x)
        osi_imag = _normalized_segment_mean(rates * sin_2x)
        dsi_real = _normalized_segment_mean(rates * cos_x)
        dsi_imag = _normalized_segment_mean(rates * sin_x)

        return (
            tf.reduce_mean(osi_real, axis=0),
            tf.reduce_mean(osi_imag, axis=0),
            tf.reduce_mean(dsi_real, axis=0),
            tf.reduce_mean(dsi_imag, axis=0),
        )

    def _smoothed_vector_magnitude(self, real, imag):
        eps_sq = tf.square(self._rolling_epsilon)
        magnitude = tf.sqrt(tf.square(real) + tf.square(imag) + eps_sq) - self._rolling_epsilon
        return tf.maximum(magnitude, tf.zeros((), dtype=self._dtype))

    def _compute_rolling_moment_values(self, rates, radians_delta_angle):
        ori_angle = 2.0 * radians_delta_angle
        dir_angle = radians_delta_angle

        batch_size = tf.cast(tf.shape(rates)[0], dtype=self._dtype)
        ori_batch_real = tf.reduce_mean(rates * tf.math.cos(ori_angle), axis=0)
        ori_batch_imag = tf.reduce_mean(rates * tf.math.sin(ori_angle), axis=0)
        dir_batch_real = tf.reduce_mean(rates * tf.math.cos(dir_angle), axis=0)
        dir_batch_imag = tf.reduce_mean(rates * tf.math.sin(dir_angle), axis=0)
        den_batch = tf.reduce_mean(rates, axis=0)

        new_ori_real = self._rolling_decay * self._rolling_ori_real + self._rolling_one_minus_decay * ori_batch_real
        new_ori_imag = self._rolling_decay * self._rolling_ori_imag + self._rolling_one_minus_decay * ori_batch_imag
        new_dir_real = self._rolling_decay * self._rolling_dir_real + self._rolling_one_minus_decay * dir_batch_real
        new_dir_imag = self._rolling_decay * self._rolling_dir_imag + self._rolling_one_minus_decay * dir_batch_imag
        new_denominator = (
            self._rolling_decay * self._rolling_denominator
            + self._rolling_one_minus_decay * den_batch
        )
        new_weight_sum = (
            self._rolling_decay * self._rolling_weight_sum
            + self._rolling_one_minus_decay
        )
        new_weight_sq_sum = (
            tf.square(self._rolling_decay) * self._rolling_weight_sq_sum
            + tf.square(self._rolling_one_minus_decay)
            / tf.maximum(batch_size, self._rolling_epsilon)
        )

        return (
            new_ori_real,
            new_ori_imag,
            new_dir_real,
            new_dir_imag,
            new_denominator,
            new_weight_sum,
            new_weight_sq_sum,
        )

    @tf.function(jit_compile=False)
    def _distribution_emd_loss(self, values, target_distributions):
        emd_losses = tf.TensorArray(self._dtype, size=self._n_node_types)
        for i in tf.range(self._n_node_types):
            current_values = tf.gather(values, self._emd_group_indices[i])
            target_values = target_distributions[i]
            emd = tf.reduce_mean(tf.abs(tf.sort(current_values) - target_values))
            emd_losses = emd_losses.write(i, emd)
        return emd_losses.stack()

    @tf.function(jit_compile=False)
    def _distribution_zero_l1_loss(self, values):
        losses = tf.TensorArray(self._dtype, size=self._n_node_types)
        for i in tf.range(self._n_node_types):
            current_values = tf.gather(values, self._emd_group_indices[i])
            losses = losses.write(i, tf.reduce_mean(tf.square(current_values)))
        return losses.stack()

    def _apply_rolling_gradient_correction(self, value):
        if not self._rolling_gradient_correction:
            return value

        scale = tf.minimum(
            1.0 / tf.maximum(self._rolling_one_minus_decay, self._rolling_epsilon),
            self._rolling_max_gradient_scale,
        )
        stopped_value = tf.stop_gradient(value)
        return stopped_value + scale * (value - stopped_value)

    def _rolling_effective_sample_count(self, weight_sum, weight_sq_sum):
        return tf.square(weight_sum) / tf.maximum(weight_sq_sum, self._rolling_epsilon)

    def _rolling_loss_warmup_scale(self, weight_sum, weight_sq_sum):
        if not self._rolling_warmup:
            return tf.constant(1.0, dtype=self._dtype)

        effective_samples = self._rolling_effective_sample_count(
            weight_sum, weight_sq_sum
        )
        scale = effective_samples / tf.maximum(
            self._rolling_target_sample_ess, self._rolling_epsilon
        )
        return tf.stop_gradient(
            tf.clip_by_value(scale, 0.0, 1.0)
        )

    def _update_rolling_selectivity_estimates(self, rates, radians_delta_angle, update_state=True):
        (
            new_ori_real,
            new_ori_imag,
            new_dir_real,
            new_dir_imag,
            new_denominator,
            new_weight_sum,
            new_weight_sq_sum,
        ) = self._compute_rolling_moment_values(
            rates, radians_delta_angle
        )

        if update_state:
            self._rolling_ori_real.assign(tf.stop_gradient(new_ori_real))
            self._rolling_ori_imag.assign(tf.stop_gradient(new_ori_imag))
            self._rolling_dir_real.assign(tf.stop_gradient(new_dir_real))
            self._rolling_dir_imag.assign(tf.stop_gradient(new_dir_imag))
            self._rolling_denominator.assign(tf.stop_gradient(new_denominator))
            self._rolling_weight_sum.assign(tf.stop_gradient(new_weight_sum))
            self._rolling_weight_sq_sum.assign(tf.stop_gradient(new_weight_sq_sum))

        gradient_ori_real = self._apply_rolling_gradient_correction(new_ori_real)
        gradient_ori_imag = self._apply_rolling_gradient_correction(new_ori_imag)
        gradient_dir_real = self._apply_rolling_gradient_correction(new_dir_real)
        gradient_dir_imag = self._apply_rolling_gradient_correction(new_dir_imag)
        gradient_denominator = self._apply_rolling_gradient_correction(new_denominator)

        # Use a smoothed vector norm to avoid NaN gradients at exactly zero activity.
        # In TF 2.15, grad(sqrt(a^2+b^2)) at a=b=0 can be non-finite.
        osi_magnitude = self._smoothed_vector_magnitude(
            gradient_ori_real, gradient_ori_imag
        )
        dsi_magnitude = self._smoothed_vector_magnitude(
            gradient_dir_real, gradient_dir_imag
        )

        safe_denominator = tf.maximum(gradient_denominator, self._min_rates_threshold)
        osi_real_estimates = gradient_ori_real / safe_denominator
        osi_imag_estimates = gradient_ori_imag / safe_denominator
        dsi_real_estimates = gradient_dir_real / safe_denominator
        dsi_imag_estimates = gradient_dir_imag / safe_denominator
        osi_estimates = osi_magnitude / safe_denominator
        dsi_estimates = dsi_magnitude / safe_denominator
        warmup_scale = self._rolling_loss_warmup_scale(
            new_weight_sum, new_weight_sq_sum
        )

        return (
            osi_estimates,
            dsi_estimates,
            osi_real_estimates,
            osi_imag_estimates,
            dsi_real_estimates,
            dsi_imag_estimates,
            warmup_scale,
        )

    def _adaptative_rolling_moments(self, rates, radians_delta_angle, update_state=True):
        if not update_state:
            return (
                self._rolling_ori_real,
                self._rolling_ori_imag,
                self._rolling_dir_real,
                self._rolling_dir_imag,
                self._rolling_denominator,
            )

        (
            new_ori_real,
            new_ori_imag,
            new_dir_real,
            new_dir_imag,
            new_denominator,
            new_weight_sum,
            new_weight_sq_sum,
        ) = self._compute_rolling_moment_values(
            rates, radians_delta_angle
        )
        self._rolling_ori_real.assign(tf.stop_gradient(new_ori_real))
        self._rolling_ori_imag.assign(tf.stop_gradient(new_ori_imag))
        self._rolling_dir_real.assign(tf.stop_gradient(new_dir_real))
        self._rolling_dir_imag.assign(tf.stop_gradient(new_dir_imag))
        self._rolling_denominator.assign(tf.stop_gradient(new_denominator))
        self._rolling_weight_sum.assign(tf.stop_gradient(new_weight_sum))
        self._rolling_weight_sq_sum.assign(tf.stop_gradient(new_weight_sq_sum))
        return new_ori_real, new_ori_imag, new_dir_real, new_dir_imag, new_denominator

    def _segment_mean_by_cell_type(self, values):
        return tf.math.unsorted_segment_mean(
            values,
            self.node_type_ids,
            num_segments=self._n_node_types,
        )

    def _adaptative_scale_from_moments(self, real, imag, denominator):
        safe_denominator = tf.maximum(denominator, self._min_rates_threshold)
        single_neuron_selectivity = (
            self._smoothed_vector_magnitude(real, imag) / safe_denominator
        )
        mean_single_selectivity = self._segment_mean_by_cell_type(
            single_neuron_selectivity
        )

        crowd_real = self._segment_mean_by_cell_type(real)
        crowd_imag = self._segment_mean_by_cell_type(imag)
        crowd_denominator = tf.maximum(
            self._segment_mean_by_cell_type(denominator),
            self._min_rates_threshold,
        )
        crowd_selectivity = (
            self._smoothed_vector_magnitude(crowd_real, crowd_imag)
            / crowd_denominator
        )

        one = tf.ones_like(mean_single_selectivity)
        raw_scale = tf.where(
            mean_single_selectivity > self._rolling_epsilon,
            crowd_selectivity / tf.maximum(mean_single_selectivity, self._rolling_epsilon),
            one,
        )
        clipped_scale = tf.clip_by_value(
            raw_scale, self._adaptative_scale_min, self._adaptative_scale_max
        )

        model_weights = tf.maximum(self.cell_type_count, self._rolling_epsilon)
        global_scale = (
            tf.reduce_sum(clipped_scale * model_weights)
            / tf.reduce_sum(model_weights)
        )
        shrink_weight = (
            self.experimental_cell_type_count
            / (self.experimental_cell_type_count + self._adaptative_shrink_k)
        )
        shrunk_scale = (
            shrink_weight * clipped_scale
            + (1.0 - shrink_weight) * global_scale
        )
        return tf.stop_gradient(
            tf.clip_by_value(
                shrunk_scale,
                self._adaptative_scale_min,
                self._adaptative_scale_max,
            )
        )

    def _adaptative_target_scales(self, rates, radians_delta_angle, update_state=True):
        (
            ori_real,
            ori_imag,
            dir_real,
            dir_imag,
            denominator,
        ) = self._adaptative_rolling_moments(
            rates, radians_delta_angle, update_state=update_state
        )
        osi_scale = self._adaptative_scale_from_moments(
            ori_real, ori_imag, denominator
        )
        dsi_scale = self._adaptative_scale_from_moments(
            dir_real, dir_imag, denominator
        )
        return osi_scale, dsi_scale

    def rates_per_sample_from_spikes(self, spikes, trim=True):
        """Return full-population rates with shape ``[batch, neurons]``."""
        spikes = spike_trimming(
            spikes,
            pre_delay=self._pre_delay,
            post_delay=self._post_delay,
            trim=trim,
        )
        if spikes.dtype != self._dtype:
            spikes = tf.cast(spikes, self._dtype)
        return tf.reduce_mean(spikes, axis=1)

    def _prepare_rates(self, rates, normalizer):
        if rates.dtype != self._dtype:
            rates = tf.cast(rates, self._dtype)

        if self._core_mask is not None:
            rates = tf.boolean_mask(rates, self._core_mask, axis=1)

        if normalizer is not None:
            if normalizer.dtype != self._dtype:
                normalizer = tf.cast(normalizer, self._dtype)
            if self._core_mask is not None:
                normalizer = tf.boolean_mask(normalizer, self._core_mask, axis=0)
            normalizer = tf.maximum(normalizer, self._min_rates_threshold)
            rates = rates / normalizer

        return rates

    def _radians_delta_angle(self, angle):
        angle = tf.cast(tf.reshape(angle, [-1]), self._dtype)
        delta_angle = angle[:, tf.newaxis] - self._tuning_angles[tf.newaxis, :]
        return delta_angle * (self._tf_pi / 180.0)

    def crowd_osi_loss_from_rates(self, rates, angle, normalizer=None):
        rates = self._prepare_rates(rates, normalizer)
        radians_delta_angle = self._radians_delta_angle(angle)
        batch_size = tf.shape(rates)[0]
        osi_real_type, osi_imag_type, dsi_real_type, dsi_imag_type = self._compute_crowd_moment_core(
            rates, radians_delta_angle, batch_size,
            self.node_type_ids, self._n_node_types
        )

        osi_loss_type = (
            tf.math.square(osi_real_type - self.osi_target_values)
            + tf.math.square(osi_imag_type)
        )
        dsi_loss_type = (
            tf.math.square(dsi_real_type - self.dsi_target_values)
            + tf.math.square(dsi_imag_type)
        )

        numerator = tf.reduce_sum((osi_loss_type + dsi_loss_type) * self.cell_type_count)
        denominator = tf.reduce_sum(self.cell_type_count)

        total_loss = (numerator / denominator) * self._osi_cost

        return total_loss

    def adaptative_crowd_osi_loss_from_rates(
        self, rates, angle, normalizer=None, update_state=True
    ):
        rates = self._prepare_rates(rates, normalizer)
        radians_delta_angle = self._radians_delta_angle(angle)
        batch_size = tf.shape(rates)[0]
        osi_real_type, osi_imag_type, dsi_real_type, dsi_imag_type = self._compute_crowd_moment_core(
            rates,
            radians_delta_angle,
            batch_size,
            self.node_type_ids,
            self._n_node_types,
        )
        osi_scale, dsi_scale = self._adaptative_target_scales(
            rates, radians_delta_angle, update_state=update_state
        )

        osi_loss_type = tf.math.square(
            osi_real_type - osi_scale * self.osi_target_values
        ) + tf.math.square(osi_imag_type)
        dsi_loss_type = tf.math.square(
            dsi_real_type - dsi_scale * self.dsi_target_values
        ) + tf.math.square(dsi_imag_type)

        numerator = tf.reduce_sum((osi_loss_type + dsi_loss_type) * self.cell_type_count)
        denominator = tf.reduce_sum(self.cell_type_count)

        return (numerator / denominator) * self._osi_cost

    def rolling_osi_emd_loss_from_rates(
        self, rates, angle, normalizer=None, update_state=True
    ):
        rates = self._prepare_rates(rates, normalizer)
        radians_delta_angle = self._radians_delta_angle(angle)
        (
            _osi_magnitude_estimates,
            _dsi_magnitude_estimates,
            osi_real_estimates,
            osi_imag_estimates,
            dsi_real_estimates,
            dsi_imag_estimates,
            warmup_scale,
        ) = self._update_rolling_selectivity_estimates(
            rates, radians_delta_angle, update_state=update_state
        )
        osi_loss_type = self._distribution_emd_loss(
            osi_real_estimates, self._osi_target_distributions
        ) + self._distribution_zero_l1_loss(
            osi_imag_estimates
        )
        dsi_loss_type = self._distribution_emd_loss(
            dsi_real_estimates, self._dsi_target_distributions
        ) + self._distribution_zero_l1_loss(
            dsi_imag_estimates
        )

        numerator = tf.reduce_sum((osi_loss_type + dsi_loss_type) * self.cell_type_count)
        denominator = tf.reduce_sum(self.cell_type_count)

        return warmup_scale * (numerator / denominator) * self._osi_cost

    def loss_from_rates(
        self, rates, angle, normalizer=None, update_state=True
    ):
        """Evaluate a rate-based OSI/DSI target from reusable sample rates."""
        if self._method == "crowd_osi":
            return self.crowd_osi_loss_from_rates(
                rates, angle, normalizer=normalizer
            )
        if self._method == "adaptative_crowd_osi":
            return self.adaptative_crowd_osi_loss_from_rates(
                rates,
                angle,
                normalizer=normalizer,
                update_state=update_state,
            )
        if self._method == "rolling_osi_emd":
            return self.rolling_osi_emd_loss_from_rates(
                rates,
                angle,
                normalizer=normalizer,
                update_state=update_state,
            )
        raise ValueError(
            f"OSI/DSI method {self._method!r} requires spike sequences."
        )

    def crowd_osi_loss(self, spikes, angle, normalizer=None):
        rates = self.rates_per_sample_from_spikes(spikes, trim=False)
        return self.crowd_osi_loss_from_rates(
            rates, angle, normalizer=normalizer
        )

    def adaptative_crowd_osi_loss(
        self, spikes, angle, normalizer=None, update_state=True
    ):
        rates = self.rates_per_sample_from_spikes(spikes, trim=False)
        return self.adaptative_crowd_osi_loss_from_rates(
            rates,
            angle,
            normalizer=normalizer,
            update_state=update_state,
        )

    def rolling_osi_emd_loss(
        self, spikes, angle, trim=None, normalizer=None, update_state=True
    ):
        del trim  # Kept for compatibility with the legacy positional argument.
        rates = self.rates_per_sample_from_spikes(spikes, trim=False)
        return self.rolling_osi_emd_loss_from_rates(
            rates,
            angle,
            normalizer=normalizer,
            update_state=update_state,
        )

    def __call__(
        self, spikes, angle, trim, normalizer=None, update_state=True
    ):
        if self._method in (
            "crowd_osi",
            "adaptative_crowd_osi",
            "rolling_osi_emd",
        ):
            rates = self.rates_per_sample_from_spikes(spikes, trim=trim)
            return self.loss_from_rates(
                rates,
                angle,
                normalizer=normalizer,
                update_state=update_state,
            )

        spikes = spike_trimming(spikes, pre_delay=self._pre_delay, post_delay=self._post_delay, trim=trim)

        if spikes.dtype != self._dtype:
            spikes = tf.cast(spikes, self._dtype)

        if self._method == "crowd_spikes":
            return self.crowd_spikes_loss(spikes, angle)
        if self._method == "neuropixels_fr":
            return self.neuropixels_fr_loss(spikes, angle)
        raise ValueError(f"Unknown OSI/DSI loss method: {self._method}")
