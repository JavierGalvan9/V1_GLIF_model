import numpy as np
import tensorflow as tf
import os 
import pickle as pkl
# import subprocess
from . import other_v1_utils


# Define a custom gradient for the spike function.
# Diverse functions can be used to define the gradient.
# Here we provide variations of this functions depending on
# the gradient type and the precision of the input tensor.

def gauss_pseudo(v_scaled, sigma, amplitude):
    return tf.math.exp(-tf.square(v_scaled) / tf.square(sigma)) * amplitude

def pseudo_derivative(v_scaled, dampening_factor):
    return dampening_factor * tf.maximum(1 - tf.abs(v_scaled), 0)

# def slayer_pseudo(v_scaled, sigma, amplitude):
#     return tf.math.exp(-sigma * tf.abs(v_scaled)) * amplitude

# @tf.custom_gradient
# def spike_slayer(v_scaled, sigma, amplitude):
#     z_ = tf.greater(v_scaled, 0.0)
#     z_ = tf.cast(z_, tf.float32)

#     def grad(dy):
#         de_dz = dy
#         dz_dv_scaled = slayer_pseudo(v_scaled, sigma, amplitude)

#         de_dv_scaled = de_dz * dz_dv_scaled

#         return [de_dv_scaled, tf.zeros_like(sigma), tf.zeros_like(amplitude)]

#     return tf.identity(z_, name="spike_slayer"), grad

@tf.custom_gradient
def spike_gauss(v_scaled, sigma, amplitude):
    z_ = tf.greater(v_scaled, 0.0)
    z_ = tf.cast(z_, tf.float32)

    def grad(dy):
        de_dz = dy
        dz_dv_scaled = gauss_pseudo(v_scaled, sigma, amplitude)
        de_dv_scaled = de_dz * dz_dv_scaled

        return [de_dv_scaled, tf.zeros_like(sigma), tf.zeros_like(amplitude)]

    return tf.identity(z_, name="spike_gauss"), grad

@tf.custom_gradient
def spike_gauss_16(v_scaled, sigma, amplitude):
    z_ = tf.greater(v_scaled, 0.0)
    z_ = tf.cast(z_, tf.float16)

    def grad(dy):
        de_dz = dy
        dz_dv_scaled = gauss_pseudo(v_scaled, sigma, amplitude)

        de_dv_scaled = de_dz * dz_dv_scaled

        return [de_dv_scaled, tf.zeros_like(sigma), tf.zeros_like(amplitude)]

    return tf.identity(z_, name="spike_gauss"), grad


@tf.custom_gradient
def spike_gauss_b16(v_scaled, sigma, amplitude):
    z_ = tf.greater(v_scaled, 0.)
    z_ = tf.cast(z_, tf.bfloat16)

    def grad(dy):
        de_dz = dy
        dz_dv_scaled = gauss_pseudo(v_scaled, sigma, amplitude)

        de_dv_scaled = de_dz * dz_dv_scaled

        return [de_dv_scaled, tf.zeros_like(sigma), tf.zeros_like(amplitude)]

    return tf.identity(z_, name='spike_gauss'), grad

@tf.custom_gradient
def spike_function(v_scaled, dampening_factor):
    z_ = tf.greater(v_scaled, 0.0)
    z_ = tf.cast(z_, tf.float32)

    def grad(dy):
        de_dz = dy
        dz_dv_scaled = pseudo_derivative(v_scaled, dampening_factor)

        de_dv_scaled = de_dz * dz_dv_scaled

        return [de_dv_scaled, tf.zeros_like(dampening_factor)]

    return tf.identity(z_, name="spike_function"), grad

@tf.custom_gradient
def spike_function_16(v_scaled, dampening_factor):
    z_ = tf.greater(v_scaled, 0.0)
    z_ = tf.cast(z_, tf.float16)

    def grad(dy):
        de_dz = dy
        dz_dv_scaled = pseudo_derivative(v_scaled, dampening_factor)

        de_dv_scaled = de_dz * dz_dv_scaled

        return [de_dv_scaled, tf.zeros_like(dampening_factor)]

    return tf.identity(z_, name="spike_function"), grad

@tf.custom_gradient
def spike_function_b16(v_scaled, dampening_factor):
    z_ = tf.greater(v_scaled, 0.0)
    z_ = tf.cast(z_, tf.bfloat16)

    def grad(dy):
        de_dz = dy
        dz_dv_scaled = pseudo_derivative(v_scaled, dampening_factor)

        de_dv_scaled = de_dz * dz_dv_scaled

        return [de_dv_scaled, tf.zeros_like(dampening_factor)]

    return tf.identity(z_, name="spike_function"), grad


@tf.custom_gradient
def calculate_synaptic_currents(rec_z_buf, synapse_indices, weight_values, dense_shape, 
                                synaptic_basis_weights, syn_ids, pre_ind_table):
    # Get the batch size and number of neurons
    batch_size = tf.cast(tf.shape(rec_z_buf)[0], dtype=tf.int64)
    n_post_neurons = tf.cast(dense_shape[0], dtype=tf.int64)
    n_syn_basis = tf.cast(tf.shape(synaptic_basis_weights)[1], dtype=tf.int64)  # Number of receptor types
    # Find the indices of non-zero inputs in x_t
    # non_zero_indices: [num_non_zero_spikes, 2], columns are [batch_index, pre_neuron_index]
    non_zero_indices = tf.where(rec_z_buf > 0)
    batch_indices = non_zero_indices[:, 0]         
    pre_neuron_indices = non_zero_indices[:, 1]
    # Retrieve relevant connections and weights for these pre_neurons
    new_indices, new_weights, new_syn_ids, post_in_degree, all_synaptic_inds = get_new_inds_table(
        synapse_indices, weight_values, syn_ids, pre_neuron_indices, pre_ind_table
    )
    # Repeat batch_indices to match total_num_connections
    batch_indices_per_connection = tf.repeat(batch_indices, post_in_degree)
    post_neuron_indices = new_indices[:, 0]
    # We will sum over all connections to get currents for each neuron and each basis
    num_segments = batch_size * n_post_neurons
    segment_ids = batch_indices_per_connection * n_post_neurons + post_neuron_indices
    # Gather the factor sets for the synapses involved
    new_weights = tf.expand_dims(new_weights, axis=1)  # [total_num_connections, 1]
    new_weights = new_weights * tf.gather(synaptic_basis_weights, new_syn_ids, axis=0)
    # if per_type_training:
    #     per_type_weights = tf.expand_dims(tf.gather(recurrent_per_type_weight_values, 
    #                                                 tf.gather(connection_type_ids, all_synaptic_inds)), axis=1)
    #     new_weights = new_weights * per_type_weights
    # Calculate the total recurrent current received by each neuron per basis dimension
    i_rec_flat = tf.math.unsorted_segment_sum(
        new_weights,
        segment_ids,
        num_segments=num_segments
    )

    def grad(dy):
        # Reshape gradient dy to match batch x neuron dimensions
        # dy_reshaped = tf.reshape(dy, [batch_size, n_post_neurons, n_syn_basis])
        # Define a function to process each receptor type
        def per_receptor_grad(r_id):
            # Extract the gradient for this receptor type (shape: [batch_size, n_post_neurons])
            dy_r = tf.reshape(dy[:, r_id], [batch_size, n_post_neurons])
            # dy_r = dy_reshaped[:, :, r_id]
            # Compute gradient w.r.t rec_z_buf for this receptor type
            recurrent_weights_factors = tf.gather(synaptic_basis_weights[:, r_id], syn_ids, axis=0)
            weights_syn_receptors = weight_values * recurrent_weights_factors
            sparse_w_rec = tf.sparse.SparseTensor(synapse_indices, weights_syn_receptors, dense_shape)
            de_dv_rid = tf.sparse.sparse_dense_matmul(dy_r, sparse_w_rec, adjoint_a=False)
            
            return de_dv_rid  # shape: [batch_size, n_pre_neurons]
        # Use tf.map_fn to apply per_receptor_grad to each receptor index
        # tf.map_fn will return a tensor of shape [n_syn_basis, batch_size, n_pre_neurons]
        de_dv_all = tf.map_fn(per_receptor_grad, tf.range(n_syn_basis), 
                            dtype=dy.dtype,
                            parallel_iterations=1)
        # Sum over all receptors
        # de_dv_all: [n_syn_basis, batch_size, n_pre_neurons]
        de_dv = tf.reduce_sum(de_dv_all, axis=0)  # [batch_size, n_pre_neurons]
        de_dv = tf.cast(de_dv, dtype=rec_z_buf.dtype)

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
            
        # Gradient w.r.t weight_values
        dnew_weights = tf.gather(dy, segment_ids)  # Match connections
        dbasis_scaled_grad = dnew_weights * tf.gather(synaptic_basis_weights, new_syn_ids, axis=0)
        de_dweight_values_connection = tf.reduce_sum(dbasis_scaled_grad, axis=1)       
        # Instead of tensor_scatter_nd_add, use unsorted_segment_sum:
        de_dweight_values = tf.math.unsorted_segment_sum(
            data=de_dweight_values_connection,
            segment_ids=all_synaptic_inds,
            num_segments=tf.shape(weight_values)[0]
        )

        return [
            de_dv,              # Gradient w.r.t rec_z_buf
            None,               # synapse_indices (constant)
            de_dweight_values,    # Gradient w.r.t weight_values
            None,                 # dense_shape[0] (constant)
            None,                 # dense_shape[1] (constant)
            None,                 # synaptic_basis_weights (constant)
            None,                 # syn_ids (constant)
            None                  # pre_ind_table (constant)
        ]

    return i_rec_flat, grad

# @tf.custom_gradient
# def calculate_synaptic_currents(rec_z_buf, synapse_indices, weight_values, dense_shape, 
#                                 synaptic_basis_weights, syn_ids, pre_ind_table):
#     # Get the batch size and number of neurons
#     batch_size = tf.cast(tf.shape(rec_z_buf)[0], dtype=tf.int64)
#     n_post_neurons = tf.cast(dense_shape[0], dtype=tf.int64)
#     n_pre_neurons = tf.cast(dense_shape[1], dtype=tf.int64)
#     n_syn_basis = tf.cast(tf.shape(synaptic_basis_weights)[1], dtype=tf.int64)  # Number of receptor types
#     # n_syn_basis = int(synaptic_basis_weights.shape[1])
#     # n_syn_basis = 5
#     # Find the indices of non-zero inputs in x_t
#     # non_zero_indices: [num_non_zero_spikes, 2], columns are [batch_index, pre_neuron_index]
#     non_zero_indices = tf.where(rec_z_buf > 0)
#     batch_indices = non_zero_indices[:, 0]         
#     pre_neuron_indices = non_zero_indices[:, 1]
#     # Retrieve relevant connections and weights for these pre_neurons
#     new_indices, new_weights, new_syn_ids, post_in_degree, all_synaptic_inds = get_new_inds_table(
#         synapse_indices, weight_values, syn_ids, pre_neuron_indices, pre_ind_table
#     )
#     # Repeat batch_indices to match total_num_connections
#     batch_indices_per_connection = tf.repeat(batch_indices, post_in_degree)
#     post_neuron_indices = new_indices[:, 0]
#     # We will sum over all connections to get currents for each neuron and each basis
#     num_segments = batch_size * n_post_neurons
#     segment_ids = batch_indices_per_connection * n_post_neurons + post_neuron_indices
#     # Gather the factor sets for the synapses involved
#     new_weights = tf.expand_dims(new_weights, axis=1)  # [total_num_connections, 1]
#     weights_factors = tf.gather(synaptic_basis_weights, new_syn_ids, axis=0)
#     new_weights = new_weights * weights_factors
#     # new_weights = new_weights * tf.gather(synaptic_basis_weights, new_syn_ids, axis=0)
#     # if per_type_training:
#     #     per_type_weights = tf.expand_dims(tf.gather(recurrent_per_type_weight_values, 
#     #                                                 tf.gather(connection_type_ids, all_synaptic_inds)), axis=1)
#     #     new_weights = new_weights * per_type_weights
#     # Calculate the total recurrent current received by each neuron per basis dimension
#     i_rec_flat = tf.math.unsorted_segment_sum(
#         new_weights,
#         segment_ids,
#         num_segments=num_segments
#     )

#     # i_rec = tf.reshape(i_rec_flat, [batch_size, n_post_neurons, -1])
#     # i_rec = tf.cast(i_rec, tf.float16)

#     def grad(dy):
#         # Reshape gradient dy to match batch x neuron dimensions
#         # dy = tf.cast(dy, dtype=rec_z_buf.dtype)
#         # tf.print('Hey: ', dy.shape)
#         dy_reshaped = tf.reshape(dy, [batch_size, n_post_neurons, n_syn_basis])
#         # dy_flattened = tf.reshape(dy, [batch_size * n_post_neurons, n_syn_basis])
#         # print(dy.shape)
#         dy_t = tf.transpose(dy_reshaped, [1, 0, 2])  # [n_post_neurons, batch_size, n_syn_basis]
#         # N = tf.shape(synapse_indices)[0]
#         post_idx = synapse_indices[:, 0]  # shape [N]
#         pre_idx = synapse_indices[:, 1]   # shape [N]
#         dy_post_all = tf.gather(dy_t, post_idx, axis=0)
#         # Define a function to process a single receptor dimension
#         def per_receptor_grad(r_id):
#             # Extract receptor slice: [N, batch_size]
#             dy_post_all_r = dy_post_all[:, :, r_id]

#             # Get the factors for this receptor: [N]
#             recurrent_weights_factors_r = tf.gather(synaptic_basis_weights[:, r_id], syn_ids, axis=0)
#             val_all_r = weight_values * recurrent_weights_factors_r

#             # Compute edge contributions: [N, batch_size]
#             edge_contrib_all = dy_post_all_r * val_all_r[:, None]

#             # Segment sum over edges: [n_pre_neurons, batch_size]
#             de_dv_r_transposed = tf.math.unsorted_segment_sum(
#                 edge_contrib_all,
#                 pre_idx,
#                 n_pre_neurons
#             )
#             return de_dv_r_transposed  # shape: [n_pre_neurons, batch_size]

#         # Apply per_receptor_grad to each receptor dimension
#         # Result: [n_syn_basis, n_pre_neurons, batch_size]
#         de_dv_all = tf.map_fn(per_receptor_grad, tf.range(n_syn_basis), dtype=tf.float32)

#         # Sum over the receptor dimension: [n_pre_neurons, batch_size]
#         de_dv_transposed = tf.reduce_sum(de_dv_all, axis=0)

#         # Cast and transpose to get [batch_size, n_pre_neurons]
#         de_dv = tf.cast(de_dv_transposed, dtype=rec_z_buf.dtype)
#         de_dv = tf.transpose(de_dv)

#         # # print(dy_post_all.shape)
#         # recurrent_weights_factors = tf.gather(synaptic_basis_weights, syn_ids, axis=0)
#         # val_all = weight_values[:, None] * recurrent_weights_factors
#         # # print(val_all.shape)
#         # edge_contrib_all = dy_post_all * val_all[:, None, :]
#         # # print(edge_contrib_all.shape)
#         # # edge_contrib_all_transposed = tf.transpose(edge_contrib_all, [1, 0, 2])
#         # # print(edge_contrib_all_transposed.shape)
#         # de_dv_all = tf.math.unsorted_segment_sum(edge_contrib_all, pre_idx, n_pre_neurons)
#         # # print(de_dv_all.shape)
#         # de_dv = tf.reduce_sum(de_dv_all, axis=2)  # [batch_size, n_pre_neurons]
#         # de_dv = tf.cast(de_dv, dtype=rec_z_buf.dtype)
#         # de_dv = tf.transpose(de_dv)

#         # Gradient w.r.t weight_values
#         dnew_weights = tf.gather(dy, segment_ids)  # [total_num_connections, n_syn_basis]
#         # dbasis_scaled_grad = dnew_weights * tf.gather(synaptic_basis_weights, new_syn_ids, axis=0)
#         dbasis_scaled_grad = dnew_weights * weights_factors
#         de_dweight_values_connection = tf.reduce_sum(dbasis_scaled_grad, axis=1)
#         # Instead of tensor_scatter_nd_add, use unsorted_segment_sum:
#         de_dweight_values = tf.math.unsorted_segment_sum(
#             data=de_dweight_values_connection,
#             segment_ids=all_synaptic_inds,
#             num_segments=tf.shape(weight_values)[0]
#         )

#         return [
#             de_dv,              # Gradient w.r.t rec_z_buf
#             None,               # synapse_indices (constant)
#             de_dweight_values,    # Gradient w.r.t weight_values
#             None,                 # dense_shape[0] (constant)
#             None,                 # dense_shape[1] (constant)
#             None,                 # synaptic_basis_weights (constant)
#             None,                 # syn_ids (constant)
#             None                  # pre_ind_table (constant)
#         ]

#     return i_rec_flat, grad

# @tf.custom_gradient
# def calculate_synaptic_currents(rec_z_buf, synapse_indices, weight_values, dense_shape, 
#                                 synaptic_basis_weights, syn_ids, pre_ind_table):
#     # Forward pass setup
#     batch_size = tf.cast(tf.shape(rec_z_buf)[0], dtype=tf.int64)
#     n_post_neurons = tf.cast(dense_shape[0], tf.int64)
#     n_pre_neurons = tf.cast(dense_shape[1], tf.int64)
#     n_syn_basis = tf.cast(tf.shape(synaptic_basis_weights)[1], tf.int64)

#     # Identify non-zero spikes
#     non_zero_indices = tf.where(rec_z_buf > 0)
#     batch_indices = non_zero_indices[:, 0]
#     pre_neuron_indices = non_zero_indices[:, 1]

#     # Retrieve synaptic connections
#     # get_new_inds_table returns:
#     # new_indices: [total_num_connections, 2] with (post, pre)
#     # new_weights: [total_num_connections]
#     # new_syn_ids: [total_num_connections]
#     # post_in_degree: number of post connections per spike
#     # all_synaptic_inds: the indices in weight_values for each connection
#     new_indices, new_weights, new_syn_ids, post_in_degree, all_synaptic_inds = get_new_inds_table(
#         synapse_indices, weight_values, syn_ids, pre_neuron_indices, pre_ind_table
#     )

#     # Segment IDs for forward accumulation
#     batch_indices_per_connection = tf.repeat(batch_indices, post_in_degree)
#     post_neuron_indices = new_indices[:, 0]
#     num_segments = batch_size * n_post_neurons
#     segment_ids = batch_indices_per_connection * n_post_neurons + post_neuron_indices

#     # Apply basis weights for forward pass
#     new_weights = tf.expand_dims(new_weights, axis=1)  # [total_num_connections, 1]
#     new_weights = new_weights * tf.gather(synaptic_basis_weights, new_syn_ids, axis=0)
#     # i_rec_flat: [batch_size * n_post_neurons, n_syn_basis]
#     i_rec_flat = tf.math.unsorted_segment_sum(new_weights, segment_ids, num_segments=num_segments)

#     def grad(dy):
#         # dy: [batch_size, n_post_neurons, n_syn_basis]
#         # Reshape dy to [batch_size, n_post_neurons, n_syn_basis]
#         dy = tf.reshape(dy, [batch_size, n_post_neurons, n_syn_basis])
#         # ------------------------
#         # Gradient w.r.t rec_z_buf (de_dv)
#         # ------------------------
        
#         # Flatten dy over post_neurons and n_syn_basis
#         # dy_stacked: [batch_size, n_post_neurons * n_syn_basis]
#         # dy_stacked = tf.reshape(dy_reshaped, [batch_size, n_post_neurons * n_syn_basis])
#         dy_stacked = tf.reshape(dy, [batch_size, -1])
#         # print(dy.shape, dy_stacked.shape, dy_reshaped.shape)
#         # Build a large sparse matrix for all receptors:
#         # For each connection i, we have (post, pre, syn_id)
#         # We'll create n_syn_basis entries per connection:
#         # row = post * n_syn_basis + r_id
#         # col = pre
#         # value = weight_values[i] * synaptic_basis_weights[syn_ids[i], r_id]
#         num_connections = tf.cast(tf.shape(synapse_indices)[0], tf.int64)
#         r_ids = tf.range(n_syn_basis, dtype=tf.int64)
#         repeated_r_ids = tf.tile(r_ids, [num_connections])
#         # Repeat each connection n_syn_basis times to handle all receptors
#         expanded_post = tf.repeat(synapse_indices[:, 0], n_syn_basis)   # [num_connections*n_syn_basis]
#         expanded_pre  = tf.repeat(synapse_indices[:, 1], n_syn_basis)   # [num_connections*n_syn_basis]
#         expanded_syn_ids = tf.cast(tf.repeat(syn_ids, n_syn_basis), dtype=tf.int64)     # syn_id per each receptor repetition
#         # expanded_conn_ids = tf.repeat(all_synaptic_inds, n_syn_basis)

#         # Compute values for sparse_w_big
#         # Actually, we want to index the correct receptor in a vectorized manner:
#         # We used tf.tile(r_ids) above; we know r_id repeats regularly.
#         # Let's extract per connection receptor indices:
#         # shape: [total_num_connections * n_syn_basis]
        
#         # gather corresponding factor from basis_factors per row:
#         # synaptic_basis_weights: [n_synapses, n_syn_basis]
#         # expanded_syn_ids and repeated_r_ids give the correct element:
#         flat_factors = tf.gather_nd(synaptic_basis_weights, tf.stack([expanded_syn_ids, repeated_r_ids], axis=1))
#         # weight_values per connection:
#         # expanded_weights = tf.gather(weight_values, expanded_conn_ids)
#         expanded_weights = tf.repeat(weight_values, n_syn_basis)
#         sparse_values = expanded_weights * flat_factors
#         # Build the big sparse tensor: [n_post_neurons * n_syn_basis, n_pre_neurons]
#         expanded_rows = expanded_post * n_syn_basis + repeated_r_ids
#         expanded_cols = expanded_pre
#         sparse_w_big = tf.sparse.SparseTensor(
#             indices=tf.stack([expanded_rows, expanded_cols], axis=1),
#             values=sparse_values,
#             dense_shape=[n_post_neurons * n_syn_basis, n_pre_neurons]
#         )
#         # sparse_w_big = tf.sparse.reorder(sparse_w_big)
#         # Perform sparse-dense matmul:
#         # sparse_w_big^T: [n_pre_neurons, n_post_neurons * n_syn_basis]
#         # dy_stacked^T: [n_post_neurons * n_syn_basis, batch_size]
#         de_dv = tf.sparse.sparse_dense_matmul(dy_stacked, sparse_w_big, adjoint_a=False)
#         # Cast to original type
#         de_dv = tf.cast(de_dv, dtype=rec_z_buf.dtype)

#         # ------------------------
#         # Gradient w.r.t weight_values (de_dweight_values)
#         # ------------------------
#         # For each connection (segment_id), dy contributed is in dnew_weights:
#         dy_flattened = tf.reshape(dy, [batch_size * n_post_neurons, n_syn_basis])
#         dnew_weights = tf.gather(dy_flattened, segment_ids)  # [total_num_connections, n_syn_basis]
#         dbasis_scaled_grad = dnew_weights * tf.gather(synaptic_basis_weights, new_syn_ids, axis=0)
#         # sum over receptor dimension:
#         de_dweight_values_connection = tf.reduce_sum(dbasis_scaled_grad, axis=1)
#         de_dweight_values = tf.tensor_scatter_nd_add(
#             tf.zeros_like(weight_values),
#             indices=tf.expand_dims(all_synaptic_inds, axis=1),
#             updates=de_dweight_values_connection
#         )

#         return [
#             de_dv,               # Gradient w.r.t rec_z_buf
#             None,                # synapse_indices is constant
#             de_dweight_values,   # Gradient w.r.t weight_values
#             None, None,          # dense_shape is constant
#             None,                # synaptic_basis_weights is constant
#             None,                # syn_ids is constant
#             None                 # pre_ind_table is constant
#         ]

#     return i_rec_flat, grad

# @tf.custom_gradient
# def calculate_synaptic_currents(rec_z_buf, synapse_indices, weight_values, dense_shape, 
#                                 synaptic_basis_weights, syn_ids, pre_ind_table):
#     # Get the batch size and number of neurons
#     batch_size = tf.cast(tf.shape(rec_z_buf)[0], dtype=tf.int64)
#     n_post_neurons = tf.cast(dense_shape[0], dtype=tf.int64)
#     n_syn_basis = tf.cast(tf.shape(synaptic_basis_weights)[1], dtype=tf.int64)

#     # Find non-zero spikes in rec_z_buf
#     # non_zero_indices: [num_non_zero_spikes, 2], columns: [batch_index, pre_neuron_index]
#     non_zero_indices = tf.where(rec_z_buf > 0)
#     batch_indices = non_zero_indices[:, 0]
#     pre_neuron_indices = non_zero_indices[:, 1]

#     # Retrieve synaptic connections for these pre_neurons
#     new_indices, new_weights, new_syn_ids, post_in_degree, all_synaptic_inds = get_new_inds_table(
#         synapse_indices, weight_values, syn_ids, pre_neuron_indices, pre_ind_table
#     )
#     # Repeat batch_indices per connection
#     batch_indices_per_connection = tf.repeat(batch_indices, post_in_degree)
#     post_neuron_indices = new_indices[:, 0]

#     # Segment IDs for forward pass accumulation
#     num_segments = batch_size * n_post_neurons
#     segment_ids = batch_indices_per_connection * n_post_neurons + post_neuron_indices

#     # Expand and apply synaptic_basis_weights for forward pass
#     new_weights = tf.expand_dims(new_weights, axis=1)  # [total_num_connections, 1]
#     new_weights = new_weights * tf.gather(synaptic_basis_weights, new_syn_ids, axis=0)
#     # i_rec_flat: [batch_size * n_post_neurons, n_syn_basis]
#     i_rec_flat = tf.math.unsorted_segment_sum(new_weights, segment_ids, num_segments=num_segments)
#     i_rec = tf.reshape(i_rec_flat, [batch_size, n_post_neurons, -1])
#     i_rec = tf.cast(i_rec, tf.float16)

#     def grad(dy):
#         # dy: [batch_size * n_post_neurons, n_syn_basis]
#         # Reshape dy back to [batch_size, n_post_neurons, n_syn_basis]
#         dy_reshaped = tf.reshape(dy, [batch_size, n_post_neurons, n_syn_basis])
        
#         # -------------------------
#         # Compute de_dv (Gradient w.r.t rec_z_buf)
#         # -------------------------
        
#         # We know each connection c corresponds to:
#         # (batch_indices_per_connection[c], post_neuron_indices[c], new_syn_ids[c])
#         # We can gather the dy values per connection:
#         # dy_for_connection: shape [total_num_connections, n_syn_basis]
#         dy_for_connection = tf.gather_nd(
#             dy_reshaped,
#             tf.stack([batch_indices_per_connection, post_neuron_indices], axis=1)
#         )
#         # syn_factors: [total_num_connections, n_syn_basis]
#         conn_syn_weights = tf.gather(synaptic_basis_weights, new_syn_ids, axis=0)
#         # Combine dy with syn_factors:
#         # element-wise multiply and sum over the receptor dimension:
#         # resulting shape: [total_num_connections]
#         connection_val = tf.reduce_sum(tf.cast(dy_for_connection, tf.float32) * conn_syn_weights, axis=1)
        
#         # Now incorporate weight_values to get the per-connection contribution:
#         # Remember that for backward w.r.t rec_z_buf we need the sum over post connections,
#         # and we know each connection maps pre->post, we have:
#         # segment_ids_pre for summation over pre_neurons and batch dimension:
#         n_pre_neurons = tf.cast(tf.shape(rec_z_buf)[1], dtype=tf.int64)
#         segment_ids_pre = batch_indices_per_connection * n_pre_neurons + new_indices[:, 1]
        
#         # Incorporate original weights:
#         connection_val = connection_val * tf.gather(weight_values, all_synaptic_inds)
        
#         # Sum over all connections to get [batch_size * n_pre_neurons]
#         de_dv_flat = tf.math.unsorted_segment_sum(connection_val, segment_ids_pre, num_segments=batch_size * n_pre_neurons)
#         de_dv = tf.reshape(de_dv_flat, [batch_size, n_pre_neurons])
        
#         # Cast to match original dtype
#         de_dv = tf.cast(de_dv, dtype=rec_z_buf.dtype)

#         # -------------------------
#         # Compute de_dweight_values (Gradient w.r.t weight_values)
#         # -------------------------
        
#         # dnew_weights: gradient flowed back through new_weights
#         # dy matches connections through segment_ids,
#         # so we gather dy by segment_ids:
#         dy_flattened = tf.reshape(dy_reshaped, [batch_size * n_post_neurons, n_syn_basis])
#         dnew_weights = tf.cast(tf.gather(dy_flattened, segment_ids), tf.float32)
#         # dnew_weights: [total_num_connections, n_syn_basis]

#         # dbasis_scaled_grad = dnew_weights * syn_basis for each connection
#         dbasis_scaled_grad = dnew_weights * tf.gather(synaptic_basis_weights, new_syn_ids, axis=0)
#         # sum over receptor dimension to get back to weight_values dimension
#         de_dweight_values_connection = tf.reduce_sum(dbasis_scaled_grad, axis=1)

#         # Scatter back to weight_values
#         de_dweight_values = tf.tensor_scatter_nd_add(
#             tf.zeros_like(weight_values),
#             indices=tf.expand_dims(all_synaptic_inds, axis=1),
#             updates=de_dweight_values_connection
#         )
        
#         return [
#             de_dv,               # Gradient w.r.t rec_z_buf
#             None,                # synapse_indices is constant
#             de_dweight_values,   # Gradient w.r.t weight_values
#             None,                # dense_shape is constant
#             None,                # dense_shape is constant
#             None,                # synaptic_basis_weights is constant (no training)
#             None,                # syn_ids is constant
#             None                 # pre_ind_table is constant
#         ]

#     return i_rec, grad

# SYN_CHUNK_SIZE = 100000  # Adjust based on memory constraints
# @tf.custom_gradient
# def calculate_synaptic_currents(rec_z_buf, synapse_indices, weight_values, dense_shape, 
#                                 synaptic_basis_weights, syn_ids, pre_ind_table):
#     batch_size = tf.cast(tf.shape(rec_z_buf)[0], tf.int64)
#     n_post_neurons = tf.cast(dense_shape[0], tf.int64)
#     n_pre_neurons = tf.cast(dense_shape[1], tf.int64)
#     n_syn_basis = tf.cast(tf.shape(synaptic_basis_weights)[1], tf.int64)
#     # We know n_syn_basis = 5 in your case, but let's keep it general.

#     # Identify non-zero spikes for forward pass
#     non_zero_indices = tf.where(rec_z_buf > 0)
#     batch_indices = non_zero_indices[:, 0]
#     pre_neuron_indices = non_zero_indices[:, 1]

#     # Retrieve "active" synapses (for forward pass)
#     new_indices, new_weights, new_syn_ids, post_in_degree, all_synaptic_inds = get_new_inds_table(
#         synapse_indices, weight_values, syn_ids, pre_neuron_indices, pre_ind_table
#     )

#     # Forward pass accumulation over active synapses
#     batch_indices_per_connection = tf.repeat(batch_indices, post_in_degree)
#     post_neuron_indices = new_indices[:, 0]
#     segment_ids = batch_indices_per_connection * n_post_neurons + post_neuron_indices
#     num_segments = batch_size * n_post_neurons
#     new_weights = tf.expand_dims(new_weights, axis=1)  # [num_connections, 1]
#     new_weights = new_weights * tf.gather(synaptic_basis_weights, new_syn_ids, axis=0)
#     i_rec_flat = tf.math.unsorted_segment_sum(new_weights, segment_ids, num_segments=num_segments)

#     def grad(dy):
#         # dy: [batch_size*n_post_neurons, n_syn_basis]
#         dy_reshaped = tf.reshape(dy, [batch_size, n_post_neurons, n_syn_basis])
#         # Flatten dy along post and receptor dims: [batch_size, n_post_neurons*n_syn_basis]
#         dy_stacked = tf.reshape(dy_reshaped, [batch_size, -1])  # [batch_size, n_post_neurons*5]

#         total_synapses = tf.shape(synapse_indices)[0]
#         de_dv_accum = tf.zeros([batch_size, n_pre_neurons], dtype=rec_z_buf.dtype)

#         # Process all synapses in chunks
#         def syn_loop_cond(syn_start, de_dv_accum):
#             return syn_start < total_synapses

#         def syn_loop_body(syn_start, de_dv_accum):
#             syn_end = tf.minimum(syn_start + SYN_CHUNK_SIZE, total_synapses)
#             curr_syn_count = syn_end - syn_start

#             # Extract synapse chunk
#             syn_chunk = synapse_indices[syn_start:syn_end]   # [curr_syn_count, 2]
#             w_chunk = weight_values[syn_start:syn_end]       # [curr_syn_count]
#             sid_chunk = syn_ids[syn_start:syn_end]           # [curr_syn_count]

#             # For each synapse and each of the 5 receptors, we create entries:
#             # We have curr_syn_count synapses and n_syn_basis receptors
#             # expanded_post: [curr_syn_count * n_syn_basis]
#             expanded_post = tf.repeat(syn_chunk[:, 0], n_syn_basis)
#             expanded_pre = tf.repeat(syn_chunk[:, 1], n_syn_basis)
#             expanded_syn_ids = tf.cast(tf.repeat(sid_chunk, n_syn_basis), tf.int64)

#             # r_ids: 0,1,2,...,n_syn_basis-1 repeated for all synapses
#             r_ids = tf.range(n_syn_basis, dtype=tf.int64)
#             expanded_r_ids = tf.tile(r_ids, [curr_syn_count])

#             # Gather factors for each (synapse, receptor)
#             syn_factors = tf.gather_nd(
#                 synaptic_basis_weights,
#                 tf.stack([expanded_syn_ids, expanded_r_ids], axis=1)
#             )  # shape: [curr_syn_count*n_syn_basis]

#             expanded_w = tf.repeat(w_chunk, n_syn_basis)  # [curr_syn_count*n_syn_basis]
#             sparse_values = expanded_w * syn_factors

#             # Build sparse tensor:
#             # rows = post*n_syn_basis + r_id
#             expanded_rows = expanded_post * n_syn_basis + expanded_r_ids
#             expanded_cols = expanded_pre

#             sparse_w_chunk = tf.sparse.SparseTensor(
#                 indices=tf.stack([expanded_rows, expanded_cols], axis=1),
#                 values=sparse_values,
#                 dense_shape=[n_post_neurons * n_syn_basis, n_pre_neurons]
#             )

#             # Multiply dy_stacked with sparse_w_chunk
#             # dy_stacked: [batch_size, n_post_neurons*n_syn_basis]
#             # sparse_w_chunk: [n_post_neurons*n_syn_basis, n_pre_neurons]
#             de_dv_chunk = tf.sparse.sparse_dense_matmul(dy_stacked, sparse_w_chunk)
#             de_dv_chunk = tf.cast(de_dv_chunk, dtype=rec_z_buf.dtype)

#             # Accumulate result
#             de_dv_accum += de_dv_chunk

#             return syn_end, de_dv_accum

#         _, de_dv = tf.while_loop(
#             syn_loop_cond, syn_loop_body,
#             [tf.constant(0, dtype=tf.int32), de_dv_accum],
#             parallel_iterations=1,
#             back_prop=True
#         )

#         # -------------------------
#         # Compute de_dweight_values (Gradient w.r.t weight_values)
#         # -------------------------
        
#         # dnew_weights: gradient flowed back through new_weights
#         # dy matches connections through segment_ids,
#         # so we gather dy by segment_ids:
#         dy_flattened = tf.reshape(dy_reshaped, [batch_size * n_post_neurons, n_syn_basis])
#         dnew_weights = tf.cast(tf.gather(dy_flattened, segment_ids), tf.float32)
#         # dnew_weights: [total_num_connections, n_syn_basis]

#         # dbasis_scaled_grad = dnew_weights * syn_basis for each connection
#         dbasis_scaled_grad = dnew_weights * tf.gather(synaptic_basis_weights, new_syn_ids, axis=0)
#         # sum over receptor dimension to get back to weight_values dimension
#         de_dweight_values_connection = tf.reduce_sum(dbasis_scaled_grad, axis=1)

#         # Scatter back to weight_values
#         de_dweight_values = tf.tensor_scatter_nd_add(
#             tf.zeros_like(weight_values),
#             indices=tf.expand_dims(all_synaptic_inds, axis=1),
#             updates=de_dweight_values_connection
#         )

#         return [
#             de_dv,
#             None,           # synapse_indices
#             de_dweight_values,  # weight_values gradient
#             None, None,     # dense_shape constants
#             None,           # synaptic_basis_weights constant
#             None,           # syn_ids constant
#             None            # pre_ind_table constant
#         ]

#     return i_rec_flat, grad

# @tf.custom_gradient
# def calculate_synaptic_currents(rec_z_buf, synapse_indices, weight_values, dense_shape, 
#                                 synaptic_basis_weights, syn_ids, pre_ind_table):
#     # Get the batch size and number of neurons
#     batch_size = tf.cast(tf.shape(rec_z_buf)[0], dtype=tf.int64)
#     n_post_neurons = dense_shape[0]
#     n_syn_basis = tf.shape(synaptic_basis_weights)[1]  # Number of receptor types
#     # Find the indices of non-zero inputs in x_t
#     # non_zero_indices: [num_non_zero_spikes, 2], columns are [batch_index, pre_neuron_index]
#     non_zero_indices = tf.where(rec_z_buf > 0)
#     batch_indices = non_zero_indices[:, 0]         
#     pre_neuron_indices = non_zero_indices[:, 1]
#     # Retrieve relevant connections and weights for these pre_neurons
#     new_indices, new_weights, new_syn_ids, post_in_degree, all_synaptic_inds = get_new_inds_table(
#         synapse_indices, weight_values, syn_ids, pre_neuron_indices, pre_ind_table
#     )
#     # Repeat batch_indices to match total_num_connections
#     batch_indices_per_connection = tf.repeat(batch_indices, post_in_degree)
#     post_neuron_indices = new_indices[:, 0]
#     # We will sum over all connections to get currents for each neuron and each basis
#     num_segments = batch_size * n_post_neurons
#     segment_ids = batch_indices_per_connection * n_post_neurons + post_neuron_indices
#     # Gather the factor sets for the synapses involved
#     new_weights = tf.expand_dims(new_weights, axis=1)  # [total_num_connections, 1]
#     new_weights = new_weights * tf.gather(synaptic_basis_weights, new_syn_ids, axis=0)
#     # if per_type_training:
#     #     per_type_weights = tf.expand_dims(tf.gather(recurrent_per_type_weight_values, 
#     #                                                 tf.gather(connection_type_ids, all_synaptic_inds)), axis=1)
#     #     new_weights = new_weights * per_type_weights
#     # Calculate the total recurrent current received by each neuron per basis dimension
#     i_rec_flat = tf.math.unsorted_segment_sum(
#         new_weights,
#         segment_ids,
#         num_segments=num_segments
#     )

#     def grad(dy):
#         # Reshape gradient dy to match batch x neuron dimensions
#         dy_reshaped = tf.reshape(dy, [batch_size, n_post_neurons, -1])
#         syn_factors_all = tf.gather(synaptic_basis_weights, syn_ids, axis=0)
#         # Initialize gradients
#         de_dv = tf.zeros_like(rec_z_buf, dtype=weight_values.dtype)
#         # Loop through receptor types to calculate gradients
#         for r_id in tf.range(n_syn_basis):
#             # Extract the gradient for this receptor type
#             dy_r = dy_reshaped[:, :, r_id]  # Shape: [batch_size, n_post_neurons]
#             # Compute gradient w.r.t rec_z_buf (de_dv)
#             # recurrent_weights_factors = tf.gather(synaptic_basis_weights[:, r_id], syn_ids, axis=0)
#             # weights_syn_receptors = weight_values * recurrent_weights_factors
#             weights_syn_receptors = weight_values * syn_factors_all[:, r_id]

#             sparse_w_rec = tf.sparse.SparseTensor(synapse_indices, weights_syn_receptors, dense_shape)
#             de_dv_rid = tf.sparse.sparse_dense_matmul(dy_r, sparse_w_rec, adjoint_a=False)
#             de_dv += de_dv_rid
#             # de_dv += tf.cast(de_dv_rid, dtype=rec_z_buf.dtype)
#         de_dv = tf.cast(de_dv, dtype=rec_z_buf.dtype)
        
#         # Gradient w.r.t weight_values
#         dnew_weights = tf.gather(dy, segment_ids)  # Match connections
#         dbasis_scaled_grad = dnew_weights * tf.gather(synaptic_basis_weights, new_syn_ids, axis=0)
#         de_dweight_values = tf.tensor_scatter_nd_add(
#             tf.zeros_like(weight_values),
#             indices=tf.expand_dims(all_synaptic_inds, axis=1),
#             updates=tf.reduce_sum(dbasis_scaled_grad, axis=1)
#         )
#         return [
#             de_dv,              # Gradient w.r.t rec_z_buf
#             None,               # synapse_indices (constant)
#             de_dweight_values,    # Gradient w.r.t weight_values
#             None,                 # dense_shape[0] (constant)
#             None,                 # dense_shape[1] (constant)
#             None,                 # synaptic_basis_weights (constant)
#             None,                 # syn_ids (constant)
#             None                  # pre_ind_table (constant)
#         ]

#     return i_rec_flat, grad

def exp_convolve(tensor, decay=0.8, reverse=False, initializer=None, axis=0):
    rank = len(tensor.get_shape())
    perm = np.arange(rank)
    perm[0], perm[axis] = perm[axis], perm[0]
    tensor = tf.transpose(tensor, perm)

    if initializer is None:
        initializer = tf.zeros_like(tensor[0])

    def scan_fun(_acc, _t):
        return _acc * decay + _t

    filtered = tf.scan(scan_fun, tensor, reverse=reverse,
                       initializer=initializer)

    filtered = tf.transpose(filtered, perm)
    return filtered

def make_pre_ind_table(indices, n_source_neurons=197613):
    """
    This function creates a table that maps presynaptic indices to 
    the indices of the recurrent_indices tensor using a RaggedTensor.
    This approach ensures that every presynaptic neuron, even those with no
    postsynaptic connections, has an entry in the RaggedTensor.
    """
    pre_inds = indices[:, 1]
    n_syn = pre_inds.shape[0]
    # Since pre_inds may not be sorted, we sort them along with synapse_indices
    sorted_pre_inds, sorted_synapse_indices = tf.math.top_k(-pre_inds, k=n_syn)
    # Count occurrences (out-degrees) for each presynaptic neuron using bincount
    sorted_pre_inds = tf.cast(-sorted_pre_inds, dtype=tf.int32)  # Undo the negation to get the sorted pre_inds and cast to reduce memory usage
    # Count occurrences (out-degrees) for each presynaptic neuron using bincount
    counts = tf.math.bincount(sorted_pre_inds, minlength=n_source_neurons)
    # Create row_splits that covers all presynaptic neurons (0 to n_source_neurons)
    row_splits = tf.concat([[0], tf.cumsum(counts)], axis=0)
    # Create the RaggedTensor with empty rows for missing neurons
    rt = tf.RaggedTensor.from_row_splits(sorted_synapse_indices, row_splits)

    return rt

def get_new_inds_table(indices, weights, syn_ids, non_zero_cols, pre_ind_table):
    """Optimized function that prepares new sparse indices tensor."""
    # Gather the rows corresponding to the non_zero_cols
    selected_rows = tf.gather(pre_ind_table, non_zero_cols)
    # Flatten the selected rows to get all_inds
    all_synapse_inds = selected_rows.flat_values
    # get the number of postsynaptic neurons 
    post_in_degree = selected_rows.row_lengths()
    # Gather from indices, weights and syn_ids using all_inds
    new_indices = tf.gather(indices, all_synapse_inds)
    new_weights = tf.gather(weights, all_synapse_inds)
    new_syn_ids = tf.gather(syn_ids, all_synapse_inds)

    return new_indices, new_weights, new_syn_ids, post_in_degree, all_synapse_inds


# class BackgroundNoiseLayer(tf.keras.layers.Layer):
#     """
#     This class calculates the input currents from the BKG noise by processing all timesteps at once."
#     For that reason is unfeasible if the user wants to train the LGN -> V1 weights.
#     Each call takes 0.03 seconds for 600 ms of simulation.

#     Returns:
#         _type_: input_currents (self.compute_dtype)
#     """
#     def __init__(self, indices, weights, dense_shape,  
#                  weights_factors, batch_size, seq_len,
#                  bkg_firing_rate=250, n_bkg_units=100, 
#                  **kwargs):
#         super().__init__(**kwargs)
#         self._bkg_weights = weights
#         self._bkg_indices = indices
#         self._dense_shape = dense_shape
#         self._bkg_input_weights_factors = weights_factors
#         self._batch_size = batch_size
#         self._seq_len = seq_len
#         self._n_syn_basis = weights_factors.shape[1]
#         # self._lr_scale = lr_scale
#         self._bkg_firing_rate = bkg_firing_rate
#         self._n_bkg_units = n_bkg_units

#     def calculate_bkg_i_in(self, inputs):
#         # This function performs the tensor multiplication to calculate the recurrent currents at each timestep
#         i_in = tf.TensorArray(dtype=self.compute_dtype, size=self._n_syn_basis)
#         for r_id in range(self._n_syn_basis):
#             weights_syn_receptors = self._bkg_weights * self._bkg_input_weights_factors[:, r_id]
#             sparse_w_in = tf.sparse.SparseTensor(
#                 self._bkg_indices,
#                 weights_syn_receptors, 
#                 self._dense_shape,
#             )
#             i_receptor = tf.sparse.sparse_dense_matmul(
#                                                         sparse_w_in,
#                                                         inputs,
#                                                         adjoint_b=True
#                                                     )
#             # Optionally cast the output back to float16
#             if i_receptor.dtype != self.compute_dtype:
#                 i_receptor = tf.cast(i_receptor, dtype=self.compute_dtype)

#             # Append i_receptor to the TensorArray
#             i_in = i_in.write(r_id, i_receptor)

#         # Stack the TensorArray into a single tensor
#         i_in = i_in.stack()

#         return i_in

#     @tf.function
#     def call(self, inp): # inp only provides the shape
#         # Generate the background spikes
#         seq_len = tf.shape(inp)[1]
#         rest_of_brain = tf.random.poisson(shape=(self._batch_size * seq_len, self._n_bkg_units), 
#                                         lam=self._bkg_firing_rate * .001, 
#                                         dtype=self.variable_dtype) # this implementation is slower
        
#         # rest_of_brain = tf.cast(tf.random.uniform(
#         #         (self._batch_size, seq_len, self._n_bkg_units)) < self._bkg_firing_rate * .001, 
#         #         tf.float32) # (1, 600, 100)
#         # rest_of_brain = tf.reshape(rest_of_brain, (self._batch_size * seq_len, self._n_bkg_units)) # (batch_size*sequence_length, input_dim)
       
#         # Create a TensorArray to save the results for every receptor type
#         noise_input = self.calculate_bkg_i_in(rest_of_brain) # (5, 65871, 600)

#         noise_input = tf.transpose(noise_input, (2, 1, 0)) # (600, 65871, 5) 
#         # Reshape properly the input current
#         noise_input = tf.reshape(noise_input, (self._batch_size, seq_len, -1)) # (1, 600, 250000) # (1, 3000, 333170)
#         # noise_input = tf.expand_dims(noise_input, axis=0) # (1, 600, 65871, 5)

#         return noise_input
    
# class BKGInputLayerCell(tf.keras.layers.Layer):
#     def __init__(self, indices, weights, dense_shape, syn_ids, synaptic_basis_weights,
#                  **kwargs):
#         super().__init__(**kwargs)
#         self._indices = indices
#         self._input_weights = weights
#         self._input_syn_ids = syn_ids
#         self._synaptic_basis_weights = synaptic_basis_weights
#         # self._n_syn_basis = weights_factors.shape[1]  # Number of receptors
#         self._dense_shape = dense_shape
#         # Precompute the synapses table
#         self.pre_ind_table = make_pre_ind_table(indices, n_source_neurons=dense_shape[1])

#     @property
#     def state_size(self):
#         # No states are maintained in this cell
#         return []

#     # def build(self, input_shape):
#     #     # If you have any trainable variables, initialize them here
#     #     pass

#     # @tf.function
#     def call(self, inputs_t, states):
#         # inputs_t: Shape [batch_size, input_dim]
#         batch_size = tf.shape(inputs_t)[0]
#         n_post_neurons = self._dense_shape[0]
#         # Find the indices of non-zero spikes in rec_z_buf
#         # non_zero_indices: [num_non_zero_spikes, 2], columns are [batch_index, pre_neuron_index]
#         non_zero_indices = tf.where(inputs_t > 0)
#         batch_indices = non_zero_indices[:, 0]         # Shape: [num_non_zero_spikes]
#         pre_neuron_indices = non_zero_indices[:, 1]    # Shape: [num_non_zero_spikes]
#         new_indices, new_weights, new_syn_ids, post_in_degree, _ = get_new_inds_table(self._indices, self._input_weights, self._input_syn_ids, pre_neuron_indices, self.pre_ind_table)
#         # Expand batch_indices to match the length of inds_flat
#         # row_lengths = selected_rows.row_lengths()  # Shape: [num_non_zero_spikes]
#         batch_indices_per_connection = tf.repeat(batch_indices, post_in_degree)
#         # batch_indices_per_connection: Shape: [total_num_connections]
#         post_neuron_indices = new_indices[:, 0]  # Indices of post-synaptic neurons
#         # Compute segment_ids for unsorted_segment_sum
#         # We need to combine batch_indices and post_neuron_indices to create unique segment IDs
#         segment_ids = batch_indices_per_connection * n_post_neurons + post_neuron_indices
#         num_segments = batch_size * n_post_neurons

#         # Get the number of presynaptic spikes
#         n_pre_spikes = tf.cast(tf.gather(inputs_t[0, :], new_indices[:, 1]), dtype=self.variable_dtype)
        
#         # Get the weights for each active synapse
#         new_weights = tf.expand_dims(new_weights * n_pre_spikes, axis=1)
#         new_weights = new_weights * tf.gather(self._synaptic_basis_weights, new_syn_ids, axis=0)
#         # sorted_data = tf.gather(self._input_weights, inds, axis=0) * tf.gather(self._input_weights_factors, inds, axis=0)
#         # Calculate the total LGN input current received by each neuron
#         i_in = tf.math.unsorted_segment_sum(
#             new_weights,
#             segment_ids,
#             num_segments=num_segments
#         )
#         # Optionally cast the output back to float16
#         if i_in.dtype != self.compute_dtype:
#             i_in = tf.cast(i_in, dtype=self.compute_dtype)

#         # Add batch dimension
#         # i_in = tf.expand_dims(i_in, axis=0)  # Shape: [1, n_post_neurons, n_syn_basis]
#         i_in = tf.reshape(i_in, [batch_size, -1])
#         # Since no states are maintained, return empty state
#         return i_in, []

# class BKGInputLayer(tf.keras.layers.Layer):
#     """
#     Calculates input currents from the LGN by processing one timestep at a time using a custom RNN cell.
#     """
#     def __init__(self, indices, weights, dense_shape, syn_ids, synaptic_basis_weights,
#                  batch_size, bkg_firing_rate=250, n_bkg_units=100,
#                  **kwargs):
#         super().__init__(**kwargs)
#         self.input_cell = BKGInputLayerCell(
#             indices, weights, dense_shape, syn_ids, synaptic_basis_weights,
#             **kwargs
#         )
#         self._batch_size = batch_size
#         self._bkg_firing_rate = bkg_firing_rate
#         self._n_bkg_units = n_bkg_units
#         # Create the input RNN layer with the custom cell to recursively process all the inputs by timesteps
#         self.input_rnn = tf.keras.layers.RNN(self.input_cell, return_sequences=True, return_state=False, name='noise_rsnn')

#     @tf.function
#     def call(self, inputs, **kwargs):
#         seq_len = tf.shape(inputs)[1]
#         rest_of_brain = tf.random.poisson(shape=(self._batch_size, seq_len, self._n_bkg_units), 
#                                     lam=self._bkg_firing_rate * .001, 
#                                     dtype=self.variable_dtype) # this implementation is slower
#         # inputs: Shape [batch_size, seq_len, input_dim]
#         input_current = self.input_rnn(rest_of_brain, **kwargs)  # Outputs: [batch_size, seq_len, n_postsynaptic_neurons]

#         return input_current
    
# class LGNInputLayerCell(tf.keras.layers.Layer):
#     def __init__(self, indices, weights, dense_shape, syn_ids, synaptic_basis_weights,
#                  **kwargs):
#         super().__init__(**kwargs)
#         self._indices = indices
#         self._input_weights = weights
#         self._input_syn_ids = syn_ids
#         self._synaptic_basis_weights = synaptic_basis_weights
#         # self._n_syn_basis = weights_factors.shape[1]  # Number of receptors
#         self._dense_shape = dense_shape
#         # Precompute the synapses table
#         self.pre_ind_table = make_pre_ind_table(indices, n_source_neurons=dense_shape[1])

#     @property
#     def state_size(self):
#         # No states are maintained in this cell
#         return []

#     # def build(self, input_shape):
#     #     # If you have any trainable variables, initialize them here
#     #     pass

#     # @tf.function
#     def call(self, inputs_t, states):
#         # inputs_t: Shape [batch_size, input_dim]
#         batch_size = tf.shape(inputs_t)[0]
#         n_post_neurons = self._dense_shape[0]
#         # Find the indices of non-zero spikes in rec_z_buf
#         # non_zero_indices: [num_non_zero_spikes, 2], columns are [batch_index, pre_neuron_index]
#         non_zero_indices = tf.where(inputs_t > 0)
#         batch_indices = non_zero_indices[:, 0]         # Shape: [num_non_zero_spikes]
#         pre_neuron_indices = non_zero_indices[:, 1]    # Shape: [num_non_zero_spikes]
#         new_indices, new_weights, new_syn_ids, post_in_degree, _ = get_new_inds_table(self._indices, self._input_weights, self._input_syn_ids, pre_neuron_indices, self.pre_ind_table)
#         # Expand batch_indices to match the length of inds_flat
#         # row_lengths = selected_rows.row_lengths()  # Shape: [num_non_zero_spikes]
#         batch_indices_per_connection = tf.repeat(batch_indices, post_in_degree)
#         # batch_indices_per_connection: Shape: [total_num_connections]
#         post_neuron_indices = new_indices[:, 0]  # Indices of post-synaptic neurons
#         # Compute segment_ids for unsorted_segment_sum
#         # We need to combine batch_indices and post_neuron_indices to create unique segment IDs
#         segment_ids = batch_indices_per_connection * n_post_neurons + post_neuron_indices
#         num_segments = batch_size * n_post_neurons

#         # Get the weights for each active synapse
#         new_weights = tf.expand_dims(new_weights, axis=1)
#         new_weights = new_weights * tf.gather(self._synaptic_basis_weights, new_syn_ids, axis=0)
#         # sorted_data = tf.gather(self._input_weights, inds, axis=0) * tf.gather(self._input_weights_factors, inds, axis=0)
#         # Calculate the total LGN input current received by each neuron
#         i_in = tf.math.unsorted_segment_sum(
#             new_weights,
#             segment_ids,
#             num_segments=num_segments
#         )

#         # Optionally cast the output back to float16
#         if i_in.dtype != self.compute_dtype:
#             i_in = tf.cast(i_in, dtype=self.compute_dtype)

#         # Add batch dimension
#         # i_in = tf.expand_dims(i_in, axis=0)  # Shape: [1, n_post_neurons, n_syn_basis]
#         i_in = tf.reshape(i_in, [batch_size, -1])
#         # Since no states are maintained, return empty state
#         return i_in, []

# class LGNInputLayer(tf.keras.layers.Layer):
#     """
#     Calculates input currents from the LGN by processing one timestep at a time using a custom RNN cell.
#     """
#     def __init__(self, indices, weights, dense_shape, syn_ids, synaptic_basis_weights,
#                  **kwargs):
#         super().__init__(**kwargs)
#         self.input_cell = LGNInputLayerCell(
#             indices, weights, dense_shape, syn_ids, synaptic_basis_weights,
#             **kwargs
#         )
#         # Create the input RNN layer with the custom cell to recursively process all the inputs by timesteps
#         self.input_rnn = tf.keras.layers.RNN(self.input_cell, return_sequences=True, return_state=False, name='lgn_rsnn')

#     @tf.function
#     def call(self, inputs, **kwargs):
#         # inputs: Shape [batch_size, seq_len, input_dim]
#         input_current = self.input_rnn(inputs, **kwargs)  # Outputs: [batch_size, seq_len, n_postsynaptic_neurons]

#         return input_current


class SignedConstraint(tf.keras.constraints.Constraint):
    def __init__(self, positive):
        self._positive = positive

    def __call__(self, w):
        condition = tf.greater(self._positive, 0)  # yields bool
        sign_corrected_w = tf.where(condition, tf.nn.relu(w), -tf.nn.relu(-w))
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
        input_weight_scale=1.0,
        recurrent_weight_scale=1.0,
        lr_scale=1.0,
        spike_gradient=False,
        max_delay=5,
        batch_size=1,
        bkg_firing_rate=250,
        pseudo_gauss=False,
        train_recurrent=True,
        train_recurrent_per_type=True,
        train_input=False,
        train_noise=True,
        hard_reset=False,
        current_input=False
    ):
        super().__init__()
        _params = dict(network["node_params"])
        # Rescale the voltages to have them near 0, as we wanted the effective step size
        # for the weights to be normalized when learning (weights are scaled similarly)
        voltage_scale = _params["V_th"] - _params["E_L"]
        voltage_offset = _params["E_L"]
        _params["V_th"] = (_params["V_th"] - voltage_offset) / voltage_scale
        _params["E_L"] = (_params["E_L"] - voltage_offset) / voltage_scale
        _params["V_reset"] = (_params["V_reset"] - voltage_offset) / voltage_scale
        _params["asc_amps"] = (_params["asc_amps"] / voltage_scale[..., None])  # _params['asc_amps'] has shape (111, 2)
        # Define the other model variables
        self._node_type_ids = np.array(network["node_type_ids"])
        self._dt = tf.constant(dt, self.compute_dtype)
        self._recurrent_dampening = tf.constant(recurrent_dampening_factor, self.compute_dtype)
        self._dampening_factor = tf.constant(dampening_factor, self.compute_dtype)
        self._pseudo_gauss = pseudo_gauss
        self._lr_scale = tf.constant(lr_scale, dtype=self.compute_dtype)
        self._spike_gradient = spike_gradient
        self._hard_reset = hard_reset
        self._current_input = current_input
        self._n_neurons = int(network["n_nodes"])
        self._bkg_firing_rate = bkg_firing_rate
        self._gauss_std = tf.constant(gauss_std, self.compute_dtype)
        # Determine the membrane time decay constant
        tau = _params["C_m"] / _params["g"]
        membrane_decay = np.exp(-dt / tau)
        current_factor = 1 / _params["C_m"] * (1 - membrane_decay) * tau

        # Determine the synaptic dynamic parameters for each of the 5 basis receptors.
        path='synaptic_data/tau_basis.npy' # [0.7579732  1.33243834 2.34228851 4.11750046 7.23813909]
        tau_syns = np.load(path)
        self._n_syn_basis = tau_syns.size
        syn_decay = np.exp(-dt / tau_syns)
        syn_decay = tf.constant(syn_decay, dtype=self.compute_dtype)
        syn_decay = tf.tile(syn_decay, [self._n_neurons])
        self.syn_decay = tf.expand_dims(syn_decay, axis=0) # expand the dimension for processing different receptor types
        psc_initial = np.e / tau_syns
        psc_initial = tf.constant(psc_initial, dtype=self.compute_dtype)
        psc_initial = tf.tile(psc_initial, [self._n_neurons])
        self.psc_initial = tf.expand_dims(psc_initial, axis=0) # expand the dimension for processing different receptor types

        # Find the maximum delay in the network
        self.max_delay = int(np.round(np.min([np.max(network["synapses"]["delays"]), max_delay])))
        self.batch_size = batch_size
        
        def _gather(prop):
            return tf.gather(prop, self._node_type_ids)
    
        def _f(_v, trainable=False):
            return tf.Variable(tf.cast(_gather(_v), self.compute_dtype), trainable=trainable)

        def inv_sigmoid(_x):
            return tf.math.log(_x / (1 - _x))

        def custom_val(_v, trainable=False):
            _v = tf.Variable(
                tf.cast(inv_sigmoid(_gather(_v)), self.compute_dtype),
                trainable=trainable,
            )
            def _g():
                return tf.nn.sigmoid(_v.read_value())

            return _v, _g

        # Gather the neuron parameters for every neuron
        self.t_ref = _f(_params["t_ref"])  # refractory time
        self.v_reset = _f(_params["V_reset"])
        self.asc_amps = _f(_params["asc_amps"], trainable=False)
        _k = tf.cast(_params['k'], self.compute_dtype)
        # inverse sigmoid of the adaptation rate constant (1/ms)
        param_k, param_k_read = custom_val(_k, trainable=False)
        k = param_k_read()
        self.exp_dt_k = tf.exp(-self._dt * k)
        self.v_th = _f(_params["V_th"])
        self.v_gap = self.v_reset - self.v_th
        e_l = _f(_params["E_L"])
        self.normalizer = self.v_th - e_l
        param_g = _f(_params["g"])
        self.gathered_g = param_g * e_l
        self.decay = _f(membrane_decay)
        self.current_factor = _f(current_factor)
        self.voltage_scale = _f(voltage_scale)
        self.voltage_offset = _f(voltage_offset)

        # Find the synaptic basis representation for each synaptic type
        # path = os.path.join('GLIF_network', 'syn_id_to_syn_weights_dict.pkl')
        path = os.path.join(network["data_dir"], 'tf_data', 'syn_id_to_syn_weights_dict.pkl')
        with open(path, "rb") as f:
            syn_id_to_syn_weights_dict = pkl.load(f)
        synaptic_basis_weights = np.array(list(syn_id_to_syn_weights_dict.values()))
        self.synaptic_basis_weights = tf.constant(synaptic_basis_weights, dtype=self.variable_dtype)

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
        # Define the Tensorflow variables
        self.recurrent_indices = tf.Variable(indices, dtype=tf.int64, trainable=False)
        self.pre_ind_table = make_pre_ind_table(indices, n_source_neurons=self.recurrent_dense_shape[1])
        # add dimension for the weights factors - TensorShape([23525415, 1])
        # weights = tf.expand_dims(weights, axis=1) 
        # Set the sign of the connections (exc or inh)
        # recurrent_weight_positive = tf.Variable(
        #     weights >= 0.0, name="recurrent_weights_sign", trainable=False)
        recurrent_weight_positive = tf.constant(weights >= 0, dtype=tf.int8)

        # if training the recurrent connection per type, turn off recurrent training
        # of individual connections
        if train_recurrent:
            if train_recurrent_per_type:
                individual_training = False
                per_type_training = True
            else:
                individual_training = True
                per_type_training = False
        else:
            individual_training = False
            per_type_training = False

        # Scale the weights
        self.recurrent_weight_values = tf.Variable(
            weights * recurrent_weight_scale / lr_scale, 
            name="sparse_recurrent_weights",
            constraint=SignedConstraint(recurrent_weight_positive),
            trainable=individual_training,
            dtype=self.variable_dtype
        ) # shape = (n_synapses,)

        # prepare per_type variable, if required
        if per_type_training:
            self.per_type_training = True
            self.connection_type_ids = other_v1_utils.connection_type_ids(network)
            max_id = np.max(self.connection_type_ids) + 1
            # prepare a variable and gather with type ids.
            self.recurrent_per_type_weight_values = tf.Variable(
                tf.ones(max_id),
                name="recurrent_per_type_weights",
                constraint=ClipConstraint(0.2, 5.0),
                trainable=True,
                dtype=self.variable_dtype
            ) # shape = (n_connection_types (21 * 21))
            # multiply this to the weights (this needs to be done in the loop)
        else:
            self.per_type_training = False
            
        self.syn_ids = tf.constant(syn_ids, dtype=tf.int64)
        # self.recurrent_weights_factors = tf.gather(self.synaptic_basis_weights, self.syn_ids, axis=0) # TensorShape([23525415, 5])
        print(f"    > # Recurrent synapses: {len(indices)}")

        del indices, weights, dense_shape, delays, syn_ids

        ### LGN input connectivity ###
        self.input_dim = lgn_input["n_inputs"]
        self.lgn_input_dense_shape = (self._n_neurons, self.input_dim,)
        input_indices = np.array(lgn_input["indices"])
        input_weights = np.array(lgn_input["weights"])
        input_syn_ids = np.array(lgn_input["syn_ids"])
        input_delays = np.array(lgn_input["delays"])
        # Scale down the input weights
        input_weights = (input_weights/ voltage_scale[self._node_type_ids[input_indices[:, 0]]])
        input_delays = np.round(np.clip(input_delays, dt, self.max_delay)/dt).astype(np.int32)
        # Introduce the delays in the postsynaptic neuron indices
        # input_indices[:, 1] = input_indices[:, 1] + self._n_neurons * (input_delays - 1)
        self.input_indices = tf.Variable(input_indices, trainable=False, dtype=tf.int64)

        # Define the Tensorflow variables
        # input_weights = tf.expand_dims(input_weights, axis=1) # add dimension for the weights factors - TensorShape([23525415, 1])
        # input_weight_positive = tf.Variable(
        #     input_weights >= 0.0, name="input_weights_sign", trainable=False)
        input_weight_positive = tf.constant(input_weights >= 0, dtype=tf.int8)

        self.input_weight_values = tf.Variable(
            input_weights * input_weight_scale / lr_scale,
            name="sparse_input_weights",
            constraint=SignedConstraint(input_weight_positive),
            trainable=train_input,
            dtype=self.variable_dtype
        )
        self.input_syn_ids = tf.constant(input_syn_ids, dtype=tf.int64)
        if not self._current_input:
            self.pre_input_ind_table = make_pre_ind_table(input_indices, n_source_neurons=self.input_dim)

        print(f"    > # LGN input synapses {len(input_indices)}")
        del input_indices, input_weights, input_syn_ids, input_delays

        ### BKG input connectivity ###
        self.bkg_input_dense_shape = (self._n_neurons, bkg_input["n_inputs"],)
        bkg_input_indices = np.array(bkg_input['indices'])
        bkg_input_weights = np.array(bkg_input['weights'])
        bkg_input_syn_ids = np.array(bkg_input['syn_ids'])
        bkg_input_delays = np.array(bkg_input['delays'])

        bkg_input_weights = (bkg_input_weights/voltage_scale[self._node_type_ids[bkg_input_indices[:, 0]]])
        bkg_input_delays = np.round(np.clip(bkg_input_delays, dt, self.max_delay)/dt).astype(np.int32)
        # Introduce the delays in the postsynaptic neuron indices
        # bkg_input_indices[:, 1] = bkg_input_indices[:, 1] + self._n_neurons * (bkg_input_delays - 1)
        self.bkg_input_indices = tf.Variable(bkg_input_indices, trainable=False, dtype=tf.int64)
        self.pre_bkg_ind_table = make_pre_ind_table(bkg_input_indices, n_source_neurons=bkg_input["n_inputs"])

        # Define Tensorflow variables
        # bkg_input_weight_positive = tf.Variable(
        #     bkg_input_weights >= 0.0, name="bkg_input_weights_sign", trainable=False)
        bkg_input_weight_positive = tf.constant(bkg_input_weights >= 0, dtype=tf.int8)
        self.bkg_input_weights = tf.Variable(
            bkg_input_weights * input_weight_scale / lr_scale, 
            name="rest_of_brain_weights", 
            constraint=SignedConstraint(bkg_input_weight_positive),
            trainable=train_noise,
            dtype=self.variable_dtype
        )

        self.bkg_input_syn_ids = tf.constant(bkg_input_syn_ids, dtype=tf.int64)
        # self.bkg_input_weights_factors = tf.gather(self.synaptic_basis_weights, bkg_input_syn_ids, axis=0)
        
        print(f"    > # BKG input synapses {len(bkg_input_indices)}")
        del bkg_input_indices, bkg_input_weights, bkg_input_syn_ids, bkg_input_delays

    def calculate_input_current_from_spikes(self, x_t):
        # x_t: Shape [batch_size, input_dim]
        # batch_size = tf.shape(x_t)[0]
        n_post_neurons = self.lgn_input_dense_shape[0]
        # Find the indices of non-zero inputs
        non_zero_indices = tf.where(x_t > 0)
        batch_indices = non_zero_indices[:, 0]
        pre_neuron_indices = non_zero_indices[:, 1]
        # Get the indices into self.recurrent_indices for each pre_neuron_index
        # self.pre_ind_table is a RaggedTensor or a list of lists mapping pre_neuron_index to indices in recurrent_indices
        new_indices, new_weights, new_syn_ids, post_in_degree, _ = get_new_inds_table(self.input_indices, self.input_weight_values, self.input_syn_ids, pre_neuron_indices, self.pre_input_ind_table)
        # Expand batch_indices to match the length of inds_flat
        # row_lengths = selected_rows.row_lengths()  # Shape: [num_non_zero_spikes]
        batch_indices_per_connection = tf.repeat(batch_indices, post_in_degree)
        # Get post-synaptic neuron indices
        post_neuron_indices = new_indices[:, 0]        
        # Compute segment IDs
        segment_ids = batch_indices_per_connection * n_post_neurons + post_neuron_indices
        num_segments = self.batch_size * n_post_neurons  
        # Get the weights for each active synapse
        new_weights = tf.expand_dims(new_weights, axis=1)
        new_weights = new_weights * tf.gather(self.synaptic_basis_weights, new_syn_ids, axis=0)
        # Calculate input currents
        i_in_flat = tf.math.unsorted_segment_sum(
            new_weights,
            segment_ids,
            num_segments=num_segments
        )
        # Cast i_rec to the compute dtype if necessary
        if i_in_flat.dtype != self.compute_dtype:
            i_in_flat = tf.cast(i_in_flat, dtype=self.compute_dtype)

        # Add batch dimension
        # i_in_flat = tf.reshape(i_in_flat, [batch_size, -1])

        return i_in_flat
    
    def calculate_input_current_from_firing_probabilities(self, x_t):
        """
        Calculate the input current to the LGN neurons from the input layer.
        """
        # batch_size = tf.shape(x_t)[0]
        # This function performs the tensor multiplication to calculate the recurrent currents at each timestep
        i_in = tf.TensorArray(dtype=self.compute_dtype, size=self._n_syn_basis)
        for r_id in range(self._n_syn_basis):
            input_weights_factors = tf.gather(self.synaptic_basis_weights[:, r_id], self.input_syn_ids, axis=0)
            weights_syn_receptors = self.input_weight_values * input_weights_factors
            sparse_w_in = tf.sparse.SparseTensor(
                self.input_indices,
                weights_syn_receptors,
                self.lgn_input_dense_shape
            )
            i_receptor = tf.sparse.sparse_dense_matmul(
                                                sparse_w_in, 
                                                tf.cast(x_t, dtype=self.variable_dtype), 
                                                adjoint_b=True
                                                )
            # Optionally cast the output back to float16
            if i_receptor.dtype != self.compute_dtype:
                i_receptor = tf.cast(i_receptor, dtype=self.compute_dtype)
            # Append i_receptor to the TensorArray
            i_in = i_in.write(r_id, i_receptor)
        # Stack the TensorArray into a single tensor
        i_in = i_in.stack()
        # flat the output
        i_in = tf.transpose(i_in)
        i_in_flat = tf.reshape(i_in, [self.batch_size * self._n_neurons, self._n_syn_basis])

        return i_in_flat
    
    # def calculate_noise_current(self, rest_of_brain):
    #     # This function performs the tensor multiplication to calculate the recurrent currents at each timestep
    #     i_in = tf.TensorArray(dtype=self.compute_dtype, size=self._n_syn_basis)
    #     for r_id in tf.range(self._n_syn_basis):
    #         bkg_input_weights_factors = tf.gather(self.synaptic_basis_weights[:, r_id], self.bkg_input_syn_ids, axis=0)
    #         weights_syn_receptors = self.bkg_input_weights * bkg_input_weights_factors
    #         sparse_w_in = tf.sparse.SparseTensor(
    #             self.bkg_input_indices,
    #             weights_syn_receptors, 
    #             self.bkg_input_dense_shape,
    #         )
    #         i_receptor = tf.sparse.sparse_dense_matmul(
    #                                                     sparse_w_in,
    #                                                     rest_of_brain,
    #                                                     adjoint_b=True
    #                                                 )
    #         # Optionally cast the output back to float16
    #         if i_receptor.dtype != self.compute_dtype:
    #             i_receptor = tf.cast(i_receptor, dtype=self.compute_dtype)
    #         # Append i_receptor to the TensorArray
    #         i_in = i_in.write(r_id, i_receptor)
    #     # Stack the TensorArray into a single tensor
    #     i_in = i_in.stack()
    #     noise_input = tf.transpose(i_in, (2, 1, 0)) # (batch_size, 65871, 5) 

    #     return noise_input
    
    def calculate_noise_current(self, rest_of_brain):
        # x_t: Shape [batch_size, input_dim]
        batch_size = tf.shape(rest_of_brain)[0]
        n_post_neurons = self.bkg_input_dense_shape[0]
        # Find the indices of non-zero inputs
        non_zero_indices = tf.where(rest_of_brain > 0)
        batch_indices = non_zero_indices[:, 0]
        pre_neuron_indices = non_zero_indices[:, 1]
        # Get the indices into self.recurrent_indices for each pre_neuron_index
        # self.pre_ind_table is a RaggedTensor or a list of lists mapping pre_neuron_index to indices in recurrent_indices
        new_indices, new_weights, new_syn_ids, post_in_degree, _ = get_new_inds_table(self.bkg_input_indices, self.bkg_input_weights, self.bkg_input_syn_ids, pre_neuron_indices, self.pre_bkg_ind_table)
        # Expand batch_indices to match the length of inds_flat
        # row_lengths = selected_rows.row_lengths()  # Shape: [num_non_zero_spikes]
        batch_indices_per_connection = tf.repeat(batch_indices, post_in_degree)
        # Get post-synaptic neuron indices
        post_neuron_indices = new_indices[:, 0]        
        # Compute segment IDs
        segment_ids = batch_indices_per_connection * n_post_neurons + post_neuron_indices
        num_segments = batch_size * n_post_neurons  
        # Get the number of presynaptic spikes
        # n_pre_spikes = tf.cast(tf.gather(rest_of_brain[0, :], new_indices[:, 1]), dtype=self.variable_dtype)
        # Get the number of presynaptic spikes
        presynaptic_indices = tf.stack([batch_indices_per_connection, new_indices[:, 1]], axis=1)
        n_pre_spikes = tf.cast(tf.gather_nd(rest_of_brain, presynaptic_indices), dtype=self.variable_dtype)
        # Get the weights for each active synapse
        new_weights = tf.expand_dims(new_weights * n_pre_spikes, axis=1)
        new_weights = new_weights * tf.gather(self.synaptic_basis_weights, new_syn_ids, axis=0)
        # Calculate input currents
        i_in_flat = tf.math.unsorted_segment_sum(
            new_weights,
            segment_ids,
            num_segments=num_segments
        )
        # Cast i_rec to the compute dtype if necessary
        if i_in_flat.dtype != self.compute_dtype:
            i_in_flat = tf.cast(i_in_flat, dtype=self.compute_dtype)

        return i_in_flat
    
    def calculate_i_rec_with_custom_grad(self, rec_z_buf):          
        i_rec_flat = calculate_synaptic_currents(rec_z_buf, self.recurrent_indices, self.recurrent_weight_values, self.recurrent_dense_shape, 
                                                 self.synaptic_basis_weights, self.syn_ids, self.pre_ind_table)
        # # Cast i_rec to the compute dtype if necessary
        if i_rec_flat.dtype != self.compute_dtype:
            i_rec_flat = tf.cast(i_rec_flat, dtype=self.compute_dtype)

        return i_rec_flat
    
    # def calculate_i_rec(self, rec_z_buf):
    #     # This is a new faster implementation that uses the pre_ind_table as a raggedTensor and exploits
    #     # the sparseness of the rec_z_buf.
    #     # it identifies the non_zero rows of rec_z_buf and only computes the
    #     # contributions for those rows.     
    #     batch_size = tf.shape(rec_z_buf)[0]
    #     n_post_neurons = self.recurrent_dense_shape[0]
    #     # Find the indices of non-zero spikes in rec_z_buf
    #     # non_zero_indices: [num_non_zero_spikes, 2], columns are [batch_index, pre_neuron_index]
    #     non_zero_indices = tf.where(rec_z_buf > 0)
    #     batch_indices = non_zero_indices[:, 0]         # Shape: [num_non_zero_spikes]
    #     pre_neuron_indices = non_zero_indices[:, 1]    # Shape: [num_non_zero_spikes]
    #     # Get the indices into self.recurrent_indices for each pre_neuron_index
    #     # self.pre_ind_table is a RaggedTensor or a list of lists mapping pre_neuron_index to indices in recurrent_indices
    #     new_indices, new_weights, new_syn_ids, post_in_degree, all_synapse_inds = get_new_inds_table(self.recurrent_indices, self.recurrent_weight_values, self.syn_ids, pre_neuron_indices, self.pre_ind_table)
    #     # Expand batch_indices to match the length of inds_flat
    #     # row_lengths = selected_rows.row_lengths()  # Shape: [num_non_zero_spikes]
    #     batch_indices_per_connection = tf.repeat(batch_indices, post_in_degree)
    #     # batch_indices_per_connection: Shape: [total_num_connections]
    #     post_neuron_indices = new_indices[:, 0]  # Indices of post-synaptic neurons
    #     # Compute segment_ids for unsorted_segment_sum
    #     # We need to combine batch_indices and post_neuron_indices to create unique segment IDs
    #     segment_ids = batch_indices_per_connection * n_post_neurons + post_neuron_indices
    #     num_segments = batch_size * n_post_neurons
    #     # Get the weights for each active synapse
    #     new_weights = tf.expand_dims(new_weights, axis=1)
    #     new_weights = new_weights * tf.gather(self.synaptic_basis_weights, new_syn_ids, axis=0)
    #     # Weight the synaptic currents by the per-type weights
    #     if self.per_type_training:
    #         per_type_weights = tf.expand_dims(tf.gather(self.recurrent_per_type_weight_values, 
    #                                                     tf.gather(self.connection_type_ids, all_synapse_inds)), axis=1)
    #         new_weights = new_weights * per_type_weights
    #     # Calculate the total recurrent current received by each neuron
    #     i_rec_flat = tf.math.unsorted_segment_sum(
    #         new_weights,
    #         segment_ids,
    #         num_segments=num_segments
    #     )

    #     if i_rec_flat.dtype != self.compute_dtype:
    #         i_rec_flat = tf.cast(i_rec_flat, dtype=self.compute_dtype)
    #     # Add batch dimension
    #     # i_rec = tf.expand_dims(i_rec, axis=0)
    #     # i_rec_flat = tf.reshape(i_rec_flat, [1, -1])
    #     # i_rec_flat = tf.reshape(i_rec_flat, [batch_size, -1])
            
    #     return i_rec_flat        
    
    def update_psc(self, psc, psc_rise, rec_inputs):
        new_psc_rise = psc_rise * self.syn_decay + rec_inputs * self.psc_initial
        new_psc = psc * self.syn_decay + self._dt * self.syn_decay * psc_rise
        return new_psc, new_psc_rise

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
        )
        return state_size
    
    def zero_state(self, batch_size, dtype=tf.float32):
        # The neurons membrane voltage start the simulation at their reset value
        z0_buf = tf.zeros((batch_size, self._n_neurons * self.max_delay), dtype)
        v0 = tf.ones((batch_size, self._n_neurons), dtype) * tf.cast(
            self.v_th * 0.0 + 1.0 * self.v_reset, dtype)
        r0 = tf.zeros((batch_size, self._n_neurons), dtype)
        asc = tf.zeros((batch_size, self._n_neurons * 2), dtype)
        psc_rise0 = tf.zeros((batch_size, self._n_neurons * self._n_syn_basis), dtype)
        psc0 = tf.zeros((batch_size, self._n_neurons * self._n_syn_basis), dtype)

        return z0_buf, v0, r0, asc, psc_rise0, psc0

    # @tf.function
    def call(self, inputs, state, constants=None):

        # # Get all the model inputs
        # external_current = inputs[:, :self._n_neurons*self._n_syn_basis] # external inputs shape (1, 399804)
        # bkg_noise = inputs[:, self._n_neurons*self._n_syn_basis:-self._n_neurons]
        lgn_input = inputs[:, :self.input_dim]
        # bkg_input = inputs[:, self.input_dim:-self._n_neurons]
        state_input = inputs[:, -self._n_neurons:] # dummy zeros
        # batch_size = tf.shape(lgn_input)[0]

        bkg_input = tf.random.poisson(shape=(self.batch_size, self.bkg_input_dense_shape[1]), 
                                    lam=self._bkg_firing_rate*.001, 
                                    dtype=self.variable_dtype) # this implementation is slower

        if self._spike_gradient:
            state_input = tf.zeros((1,), dtype=self.compute_dtype)
        else:
            state_input = tf.zeros((4,), dtype=self.compute_dtype)
                
        # Extract the network variables from the state
        z_buf, v, r, asc, psc_rise, psc = state
        # Get previous spikes
        prev_z = z_buf[:, :self._n_neurons]  # Shape: [batch_size, n_neurons]
        # Define the spikes buffer
        dampened_z_buf = z_buf * self._recurrent_dampening  # dampened version of z_buf 
        # Now we use tf.stop_gradient to prevent the term (z_buf - dampened_z_buf) to be trained
        rec_z_buf = (tf.stop_gradient(z_buf - dampened_z_buf) + dampened_z_buf)  
        # Calculate the recurrent postsynaptic currents
        # i_rec = self.calculate_i_rec(rec_z_buf)
        i_rec = self.calculate_i_rec_with_custom_grad(rec_z_buf)
        # # Add all the current sources
        # rec_inputs = i_rec + external_current + bkg_noise
        # Calculate the postsynaptic current from the external input
        if self._current_input:
            external_current = self.calculate_input_current_from_firing_probabilities(lgn_input)
        else:
            external_current = self.calculate_input_current_from_spikes(lgn_input)

        i_noise = self.calculate_noise_current(bkg_input)
        # tf.print('Hey!')
        # tf.print(i_rec.shape, external_current.shape, i_noise.shape)
        # tf.print(tf.reduce_sum(tf.reshape(i_noise, [batch_size, self._n_neurons * self._n_syn_basis]), axis=1))
        # tf.print(tf.reduce_sum(tf.reshape(external_current, [batch_size, self._n_neurons * self._n_syn_basis]), axis=1))
        # tf.print(tf.reduce_sum(tf.reshape(i_rec, [batch_size, self._n_neurons * self._n_syn_basis]), axis=1))
        # Add all the current sources
        rec_inputs = i_rec + external_current + i_noise
        # Reshape i_rec_flat back to [batch_size, num_neurons]
        rec_inputs = tf.reshape(rec_inputs, [self.batch_size, self._n_neurons * self._n_syn_basis])
        # Scale with the learning rate
        rec_inputs = rec_inputs * self._lr_scale
        if constants is not None and not self._spike_gradient:
            rec_inputs = rec_inputs + state_input * self._lr_scale

        # Calculate the new psc variables
        new_psc, new_psc_rise = self.update_psc(psc, psc_rise, rec_inputs)        
        # Calculate the ASC
        asc = tf.reshape(asc, (self.batch_size, self._n_neurons, 2))
        new_asc = self.exp_dt_k * asc + tf.expand_dims(prev_z, axis=-1) * self.asc_amps
        new_asc = tf.reshape(new_asc, (self.batch_size, self._n_neurons * 2))
        # Calculate the postsynaptic current 
        input_current = tf.reshape(psc, (self.batch_size, self._n_neurons, self._n_syn_basis))
        input_current = tf.reduce_sum(input_current, -1) # sum over receptors
        if constants is not None and self._spike_gradient:
            input_current += state_input

        # Add all the postsynaptic current sources
        c1 = input_current + tf.reduce_sum(asc, axis=-1) + self.gathered_g

        # Calculate the new voltage values
        decayed_v = self.decay * v
        reset_current = prev_z * self.v_gap
        new_v = decayed_v + self.current_factor * c1 + reset_current

        # Update the voltage according to the LIF equation and the refractory period
        # New r is a variable that accounts for the refractory period in which a neuron cannot spike
        new_r = tf.nn.relu(r + prev_z * self.t_ref - self._dt)  # =max(r + prev_z * self.t_ref - self._dt, 0)
        if self._hard_reset:
            # Here we keep the voltage at the reset value during the refractory period
            new_v = tf.where(new_r > 0.0, self.v_reset, new_v)
            # Here we make a hard reset and let the voltage freely evolve but we do not let the
            # neuron spike during the refractory period

        # Generate the network spikes
        v_sc = (new_v - self.v_th) / self.normalizer
        if self._pseudo_gauss:
            if self.compute_dtype == tf.bfloat16:
                new_z = spike_gauss_b16(v_sc, self._gauss_std, self._dampening_factor)
            elif self.compute_dtype == tf.float16:
                new_z = spike_gauss_16(v_sc, self._gauss_std, self._dampening_factor)
            else:
                new_z = spike_gauss(v_sc, self._gauss_std, self._dampening_factor)
        else:
            if self.compute_dtype == tf.bfloat16:
                new_z = spike_function_b16(v_sc, self._dampening_factor)
            elif self.compute_dtype == tf.float16:
                new_z = spike_function_16(v_sc, self._dampening_factor)
            else:
                new_z = spike_function(v_sc, self._dampening_factor)

        # Generate the new spikes if the refractory period is concluded
        new_z = tf.where(tf.greater(new_r, 0.0), tf.zeros_like(new_z), new_z)
        # Add current spikes to the buffer
        new_z_buf = tf.concat([new_z, z_buf[:, :-self._n_neurons]], axis=1)  # Shift buffer

        # Define the model outputs and the new state of the network
        outputs = (
            new_z,
            new_v * self.voltage_scale + self.voltage_offset,
            # (input_current + tf.reduce_sum(asc, axis=-1)) * self.voltage_scale,
        )

        new_state = (
            new_z_buf,
            new_v,
            new_r,
            new_asc,
            new_psc_rise,
            new_psc,
        )

        return outputs, new_state


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
    max_delay=5,
    batch_size=None,
    pseudo_gauss=False,
    hard_reset=False,
    current_input=False
):

    # Create the input layer of the model
    x = tf.keras.layers.Input(shape=(None, n_input,))
    neurons = network["n_nodes"]

    # Create an input layer for the initial state of the RNN
    state_input_holder = tf.keras.layers.Input(shape=(None, neurons))
    state_input = tf.cast(tf.identity(state_input_holder), dtype)  

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
        max_delay=max_delay,
        pseudo_gauss=pseudo_gauss,
        batch_size=batch_size, 
        train_recurrent=train_recurrent,
        train_recurrent_per_type=train_recurrent_per_type,
        train_input=train_input,
        train_noise=train_noise,
        hard_reset=hard_reset,
        current_input=current_input
    )

    # initialize the RNN state to zero using the zero_state() method of the V1Column class.
    zero_state = cell.zero_state(batch_size, dtype)

    if use_state_input:
        # The shape of each input tensor matches the shape of the corresponding
        # tensor in the zero_state tuple, except for the batch dimension. The batch
        # dimension is left unspecified, allowing the tensor to be fed variable-sized
        # batches of data.
        initial_state_holder = tf.nest.map_structure(lambda _x: tf.keras.layers.Input(shape=_x.shape[1:], dtype=_x.dtype), zero_state)
        # The code then copies the input tensors into the rnn_initial_state variable
        # using tf.nest.map_structure(). This creates a nested structure of tensors with
        # the same shape as the original zero_state structure.
        rnn_initial_state = tf.nest.map_structure(tf.identity, initial_state_holder)
        # In both cases, the code creates a constants tensor using tf.zeros_like() or
        # tf.zeros(). This tensor is used to provide constant input to the RNN during
        # computation. The shape of the constants tensor matches the batch_size.
        constants = tf.zeros_like(rnn_initial_state[0][:, 0], dtype)
    else:
        rnn_initial_state = zero_state
        constants = tf.zeros((batch_size,))

    # # Create the LGN input layer of the model
    # rnn_inputs = LGNInputLayer(
    #     cell.input_indices,
    #     cell.input_weight_values,
    #     cell.lgn_input_dense_shape,
    #     cell.input_syn_ids,
    #     cell.synaptic_basis_weights,
    #     # lr_scale=lr_scale,
    #     name="input_layer",
    # )(x)

    # # Create the BKG input layer of the model
    # bkg_inputs = BackgroundNoiseLayer(
    #     cell.bkg_input_indices,
    #     cell.bkg_input_weights,
    #     cell.bkg_input_dense_shape,
    #     cell.bkg_input_weights_factors, 
    #     batch_size, 
    #     seq_len,
    #     # lr_scale=lr_scale,
    #     name="noise_layer",
    # )(x) # the input is provided just because in a Keras custom layer, the call method should accept input

    # Generate the background spikes
    # seq_len = tf.shape(x)[1]
    # bkg_inputs = BKGInputLayer(
    #     cell.bkg_input_indices,
    #     cell.bkg_input_weights,
    #     cell.bkg_input_dense_shape,
    #     cell.bkg_input_syn_ids, 
    #     cell.synaptic_basis_weights,
    #     batch_size, 
    #     name="noise_layer",
    #     )(x)

    # Concatenate the input layer with the initial state of the RNN
    # full_inputs = tf.concat((rnn_inputs, bkg_inputs, state_input), -1) # (None, 600, 5*n_neurons+n_neurons)
    # full_inputs = tf.concat((tf.cast(x, dtype), bkg_inputs, state_input), -1)
    full_inputs = tf.concat((tf.cast(x, dtype), state_input), -1)
    
    # Create the RNN layer of the model using the V1Column cell
    # The RNN layer returns the output of the RNN layer and the final state of the RNN
    # layer. The output of the RNN layer is a tensor of shape (batch_size, seq_len,
    # neurons). The final state of the RNN layer is a tuple of tensors, each of shape
    # (batch_size, neurons).
    rnn = tf.keras.layers.RNN(cell, return_sequences=True, return_state=return_state, name="rsnn")

    # Apply the rnn layer to the full_inputs tensor
    rsnn_out = rnn(full_inputs, initial_state=rnn_initial_state, constants=constants)

    # Check if the return_state argument is True or False and assign the output of the
    # RNN layer to the hidden variable accordingly.
    if return_state:
        hidden = rsnn_out[0]
        # new_state = out[1:]
    else:
        hidden = rsnn_out

    spikes = hidden[0]
    voltage = hidden[1]

    # computes the mean of the spikes tensor along the second and third dimensions
    # (which represent time and neurons),
    rate = tf.reduce_mean(spikes)

    # The neuron output option selects only the output neurons from the spikes tensor
    if neuron_output:
        # The output_spikes tensor is computed by taking a linear combination
        # of the current spikes and the previous spikes, with the coefficients
        # determined by the dampening_factor. This serves to reduce the volatility
        # of the output spikes, making them more stable.
        output_spikes = 1 / dampening_factor * spikes + (1 - 1 / dampening_factor) * tf.stop_gradient(spikes)
        # The output tensor is then computed by selecting the spikes from the
        # output neurons and scaling them by a learned factor. The scale factor
        # is computed using a softplus activation function applied to the output
        # of a dense layer, and the threshold is computed by passing the output
        # spikes through another dense layer.
        output = tf.gather(output_spikes, network["readout_neuron_ids"], axis=2)
        output = tf.reduce_mean(output, -1)
        scale = 1 + tf.nn.softplus(tf.keras.layers.Dense(1)(tf.zeros_like(output[:1, :1])))
        thresh = tf.keras.layers.Dense(1)(tf.zeros_like(output))
        output = tf.stack([thresh[..., 0], output[..., -1]], -1) * scale
    # If neuron_output is False, then the output tensor is simply the result of
    # passing the spikes tensor through a dense layer with n_output units.
    else:
        output = tf.keras.layers.Dense(n_output, name="projection", trainable=False)(spikes)
        # output = tf.keras.layers.Dense(n_output, name="projection", trainable=True)(spikes)
        
    # Finally, a prediction layer is created
    # output = tf.keras.layers.Lambda(lambda _a: _a, name="prediction")(output)

    # If return_sequences is True, then the mean_output tensor is computed by
    # averaging over sequences of length down_sample in the output tensor.
    # Otherwise, mean_output is simply the mean of the last cue_duration time steps
    # of the output tensor.
    # if return_sequences:
    #     mean_output = tf.reshape(output, (-1, int(seq_len / down_sample), down_sample, n_output))
    #     mean_output = tf.reduce_mean(mean_output, 2)
    #     mean_output = tf.nn.softmax(mean_output, axis=-1)
    # else:
    #     mean_output = tf.reduce_mean(output[:, -cue_duration:], 1)
    #     mean_output = tf.nn.softmax(mean_output)

    if use_state_input:
        # many_input_model = tf.keras.Model(
        #     inputs=[x, state_input_holder, initial_state_holder], 
        #     outputs=mean_output
        # )
        many_input_model = tf.keras.Model(
            inputs=[x, state_input_holder,initial_state_holder],
            outputs=[output])
    else:
        # many_input_model = tf.keras.Model(
        #     inputs=[x, state_input_holder], 
        #     outputs=mean_output
        # )
        many_input_model = tf.keras.Model(
            inputs=[x, state_input_holder],
            outputs=[output])

    if add_metric:
        # add the firing rate of the neurons as a metric to the model
        many_input_model.add_metric(rate, name="rate")

    return many_input_model


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
        max_delay=5,
        batch_size=1,
        pseudo_gauss=False,
        hard_reset=True,
    )
