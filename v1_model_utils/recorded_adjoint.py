"""Exact BPTT of a V1Column rollout without replaying its forward pass.

The segmented runner keeps chunk-boundary states and, for the gradient, re-runs
each chunk's forward under a tape. The cell's backward operators, however, read
only three things per timestep from the forward pass: the float32 membrane after
the step, the blocked (refractory) mask and the spike history. The recorded
adjoint keeps exactly those - the membrane by reference, the binary ones packed
31 to an int32 word - and runs the timesteps in reverse calling the operators
the cell's custom gradients call, with the same arguments, routing the
cotangents as autodiff does. No forward operator runs in the backward pass.

Nor is the dense [batch, time, neurons] spike sequence materialized: the
training losses need only per-neuron spike counts and a few sampled population
traces (SpikeStatistics), which the forward loop streams and whose cotangents
the reverse loop turns back into each timestep's spike gradient.
"""

from typing import NamedTuple

import tensorflow as tf

from . import models
from .cuda_csr_external.wrapper import (
    external_backward,
    external_pair_weights,
    incoming_weight_backward,
    resource_external_backward,
)
from .cuda_csr_recurrent.wrapper import (
    MAX_SPIKE_SLOTS,
    overwrite_entries,
    recurrent_backward,
    resource_recurrent_backward,
    spike_activity_masks,
    spike_slots,
)
from .cuda_frame_store import wrapper as frame_store
from .cuda_glif_state.wrapper import glif_constants, glif_step_backward, record_packed_masks


class SampledTraces(NamedTuple):
    """Population traces of the batch rows ``rows`` over the timesteps ``window``.

    ``sample_ids`` ([rows, per_trial * max_count] neuron ids) and
    ``neuron_mask`` ([rows, per_trial, max_count] in the spike dtype) are a
    SynchronizationLoss sample plan; the trace of row b's sample p sums the
    masked neurons of that sample, as ``_gather_population_traces`` does.
    """

    rows: tuple
    window: tuple
    sample_ids: tf.Tensor
    neuron_mask: tf.Tensor


class SpikeStatistics(NamedTuple):
    """What a recorded rollout returns instead of the spike sequence: the spike
    counts of every (sample, neuron) over ``count_window`` and the ``traces``."""

    count_window: tuple
    traces: tuple = ()


_WORD_BITS = models.SegmentedRecomputeRunner._PACKED_SPIKES_PER_WORD


class _Sampling(NamedTuple):
    """The SampledTraces of a SpikeStatistics as one flat list of entries: entry
    k adds ``mask[k]`` times spike ``ids[k]`` of the flattened [batch * width]
    spikes to trace ``traces[k]`` (trial-major over the groups, in order).
    Each trace's entries are contiguous; ``entries`` is their common count
    when every group has the same ``max_count`` (else 0).
    Indices are int64: int32 tensors live in host memory, and the step loops
    would copy them to the device every step."""

    ids: tf.Tensor
    mask: tf.Tensor
    traces: tf.Tensor
    n_traces: int
    entries: int

    @classmethod
    def of(cls, statistics, width):
        ids, masks, traces, offset = [], [], [], 0
        for group in statistics.traces:
            rows, per_trial, max_count = group.neuron_mask.shape
            sample_rows = tf.range(group.rows[0], group.rows[1], dtype=tf.int64)
            ids.append(tf.reshape(
                sample_rows[:, None] * width + tf.cast(group.sample_ids, tf.int64), [-1]
            ))
            masks.append(tf.reshape(group.neuron_mask, [-1]))
            traces.append(tf.repeat(
                tf.range(offset, offset + rows * per_trial, dtype=tf.int64), max_count
            ))
            offset += rows * per_trial
        counts = {group.neuron_mask.shape[2] for group in statistics.traces}
        return cls(tf.concat(ids, 0), tf.concat(masks, 0), tf.concat(traces, 0), offset,
                   counts.pop() if len(counts) == 1 else 0)


@tf.function(jit_compile=True)
def _record_step(spikes, counts, counted, sampling):
    """Add the step's spikes to the counts and sum the sampled traces (exact:
    binary terms, integer sums)."""
    traces = ()
    if sampling is not None:
        sampled = tf.gather(tf.reshape(spikes, [-1]), sampling.ids) * sampling.mask
        # Contiguous equal-length traces: a row reduction, not a scatter.
        traces = ((
            tf.reduce_sum(tf.reshape(sampled, [sampling.n_traces, sampling.entries]), axis=1)
            if sampling.entries
            else tf.math.unsorted_segment_sum(sampled, sampling.traces, sampling.n_traces)
        ),)
    return (counts + tf.cast(spikes, counts.dtype) * counted,) + traces


@tf.function(jit_compile=True)
def _pack_step(spikes, refractory):
    """The step's packed spikes and blocked mask, for a cell whose GLIF op did
    not write them (record_packed_masks)."""
    return (
        models._pack_spike_words(spikes, _WORD_BITS),
        models._pack_spike_words(refractory > 0, _WORD_BITS),
    )


@tf.function(jit_compile=True)
def _unpack_frames(packed_spikes, packed_blocked, width, dtype):
    """Unpack a spike frame and (unless None) a blocked mask."""
    return (
        models._unpack_spike_words(packed_spikes, width, dtype, _WORD_BITS),
        None if packed_blocked is None
        else models._unpack_spike_words(packed_blocked, width, tf.bool, _WORD_BITS),
    )


@tf.function(jit_compile=True)
def _add_traces(count_gradient, trace_gradient, sampling, runs, first):
    """The step's spike cotangent: the traces' cotangents scattered onto the
    counts' (whose zeros the caller made positive), as the sums the dense
    gradient forms by scattering into zeros and adding (x + 0 = x for every x
    but -0). Entries that sample one neuron (``runs``, see _duplicate_runs) are
    summed first, as that scatter sums them, and added once, by their run's
    ``first`` entry; the others add +0."""
    updates = tf.gather(trace_gradient, sampling.traces) * sampling.mask
    summed = tf.math.unsorted_segment_sum(updates, runs, updates.shape[0])
    updates = tf.where(first, tf.gather(summed, runs), tf.zeros_like(updates))
    scattered = tf.tensor_scatter_nd_add(
        tf.reshape(count_gradient, [-1]), sampling.ids[:, None], updates
    )
    return tf.reshape(scattered, count_gradient.shape)


@tf.function(jit_compile=True)
def _traced_values(count_gradient, trace_gradient, sampling, runs):
    """What _add_traces leaves at the sampled entries: each count cotangent
    plus its run's summed traces (the run's other entries add +0, which
    changes no sum here: the counts' cotangent holds no -0)."""
    updates = tf.gather(trace_gradient, sampling.traces) * sampling.mask
    summed = tf.math.unsorted_segment_sum(updates, runs, updates.shape[0])
    return tf.gather(tf.reshape(count_gradient, [-1]), sampling.ids) + tf.gather(summed, runs)


def _duplicate_runs(ids):
    """Each entry's run (the rank of its id among the distinct ids) and whether
    it is the first entry of its run."""
    order = tf.argsort(ids, stable=True)
    ordered = tf.gather(ids, order)
    starts = tf.concat([[True], ordered[1:] != ordered[:-1]], 0)
    runs = tf.cumsum(tf.cast(starts, tf.int64)) - 1
    order = tf.cast(order, tf.int64)[:, None]
    return (tf.scatter_nd(order, runs, tf.shape(ids, tf.int64)),
            tf.scatter_nd(order, tf.cast(starts, tf.int64), tf.shape(ids, tf.int64)) > 0)


def _scaled_abs_max(bound, scale):
    """The abs-max bound (float bits of the largest finite magnitude, as the GLIF
    backward emits it) of ``values * scale``, from that of ``values``.

    The product is formed by the same compute-dtype multiply as the scaled
    values, and its rounding is monotone and symmetric, so the largest finite
    ``|values * scale|`` is ``|bound * scale|`` whenever that is finite. If it
    overflows, the recurrent backward measures the bound itself.
    """
    magnitude = tf.cast(tf.bitcast(bound, tf.float32), scale.dtype) * scale
    return tf.bitcast(tf.cast(tf.abs(magnitude), tf.float32), tf.uint32)


def _empty_list(dtype, shape):
    return tf.raw_ops.EmptyTensorList(
        element_shape=shape, max_num_elements=-1, element_dtype=dtype
    )


def _on_replica_device(value):
    """``value`` as an op placed on the current (replica) device.

    A same-dtype ``tf.cast`` or ``convert_to_tensor`` of a variable returns its
    bare read, which carries no device. Fed to a ``tf.while_loop``, its lowered
    Enter node is then left unplaced and lands on GPU:0, so under
    MirroredStrategy every other replica copies the tensor across GPUs on
    every iteration (for the recurrent weights 336 MB per step). Loop
    invariants read from variables go through here.
    """
    return tf.identity(value)


def _push(handle, value):
    return tf.raw_ops.TensorListPushBack(input_handle=handle, tensor=value)


def _pop(handle, like):
    return tf.raw_ops.TensorListPopBack(
        input_handle=handle, element_shape=like.shape, element_dtype=like.dtype
    )


# How the runner keeps its float32 membrane record (see the runner).
VOLTAGE_OFFLOAD_MODES = ("none", "raw", "compressed")


class RecordedAdjointRunner(models.SegmentedRecomputeRunner):
    """A SegmentedRecomputeRunner whose rollouts, given SpikeStatistics, return
    those statistics instead of the spike sequence and differentiate from
    minimal forward records instead of replaying chunks.

    With ``voltage_offload`` "raw" or "compressed" (graph mode, CUDA frame
    store available) the float32 membrane record, its earliest frames first,
    goes to pinned host memory as the forward runs and comes back ahead of the
    reverse loop (cuda_frame_store). The latest ``_PREFETCH`` frames, which the
    reverse loop needs first, stay on the GPU, and so do the frames the copy
    engine could not take before the forward ended (the store admits copies as
    earlier ones finish), so the offloaded share adapts to the PCIe bandwidth.
    "raw" copies are exact, so the gradients are those of the GPU-resident
    record. "compressed" encodes offloaded frames with TQ-pack (lossy, on the
    grid 2**-16: absolute error <= 2**-17 ~= 7.6e-6 per stored value; the live
    forward voltage stays exact) only as far as the link could not otherwise
    take them before the forward ends; the others go out exact, and the
    frames the copy engine cannot take stay on the GPU encoded.

    ``__call__(inputs, initial_state, statistics)`` returns the counts (in the
    spike dtype, like the dense sequence they sum), one [rows * per_trial,
    window] trace tensor per SampledTraces and the final state. Without
    statistics it is the segmented runner unchanged. The weight gradients are
    summed in the segmented runner's order (per chunk from the last step, then
    over the chunks from the last one), so both runners return the same bits
    for the same forward trajectory.
    """

    # Membrane frames the reverse loop fetches ahead of the one it takes (the
    # latest of them are never offloaded), and frames per copy group: one
    # compute-stream event per group each way.
    _PREFETCH = 8
    _COPY_GROUP = 4

    def __init__(self, core_model, cell, sequence_length, chunk_size,
                 voltage_offload="none", **kwargs):
        super().__init__(core_model, sequence_length, chunk_size, **kwargs)
        if voltage_offload not in VOLTAGE_OFFLOAD_MODES:
            raise ValueError(f"voltage_offload must be one of {VOLTAGE_OFFLOAD_MODES}.")
        self.cell = cell
        self.voltage_offload = voltage_offload
        self._weight_roles = {
            "recurrent": cell.recurrent_weight_values.value,
            "bkg": cell.bkg_input_weights.value,
            "lgn": cell.input_weight_values.value,
        }
        self.unsupported = self._unsupported(cell)
        self.records = not self.unsupported

    def _unsupported(self, cell):
        """The conditions of this adjoint the cell's rollout fails (empty when
        every derivative it has is one this adjoint takes)."""
        trainable = {id(variable.value) for variable in cell.trainable_variables}
        conditions = {
            "cuda state backend": cell._state_backend == "cuda",
            "chained current sources": cell._chain_current_sources,
            f"1 < max_delay <= {MAX_SPIKE_SLOTS}": 1 < cell.max_delay <= MAX_SPIKE_SLOTS,
            "no voltage sequences": not cell._return_voltage_sequences,
            "no output_neuron_ids": cell._output_neuron_ids is None,
            "spike dtype = compute dtype": cell._output_spike_dtype == cell.compute_dtype,
            "no LGN/BKG activity gradients": not (cell._compute_lgn_activity_gradient
                                                  or cell._compute_bkg_activity_gradient),
            "differentiate_inputs off": not self.differentiate_inputs,
            "only rec/BKG/LGN trainable":
                trainable <= {id(value) for value in self._weight_roles.values()},
            "accumulator present": (self._accumulated_variable is not None
                                    or not cell.recurrent_weight_values.trainable),
        }
        return tuple(name for name, met in conditions.items() if not met)

    def __call__(self, inputs, initial_state, statistics=None):
        if statistics is None:
            return super().__call__(inputs, initial_state)
        if not self.records:
            raise ValueError(
                "This cell's rollout has derivatives the recorded adjoint lacks: "
                + ", ".join(self.unsupported)
            )
        time_check = tf.debugging.assert_equal(
            tf.shape(inputs)[1],
            self.sequence_length,
            message="Recorded input length does not match sequence_length.",
        )

        @tf.custom_gradient
        def recorded_rollout(inputs, *initial_state):
            with tf.control_dependencies([time_check]):
                outputs, final_state, records = self._forward_records(
                    inputs, initial_state, statistics
                )

            def grad(*output_gradients, variables=None):
                n_outputs = len(outputs)
                state_gradients, variable_gradients = self._adjoint(
                    inputs, initial_state, final_state, records, statistics,
                    output_gradients[:n_outputs], output_gradients[n_outputs:],
                    tuple(variables or ()),
                )
                argument_gradients = (None,) + state_gradients
                if variables:
                    return argument_gradients, list(variable_gradients)
                return argument_gradients

            return outputs + final_state, grad

        return recorded_rollout(inputs, *tuple(initial_state))

    # Forward -------------------------------------------------------------

    def _forward_records(self, inputs, initial_state, statistics):
        cell, length = self.cell, self.sequence_length
        states = cell.to_loop_state(initial_state)
        history = states[0].slots
        spike_dtype = history[0].dtype
        batch, width = history[0].shape
        words = -(-width // _WORD_BITS)
        # The packed spike frames start with the initial history, oldest first,
        # so z[k] is frame k + max_delay.
        frames = _empty_list(tf.int32, [batch, words])
        for slot in reversed(history):
            frames = _push(frames, models._pack_spike_words(slot, _WORD_BITS))
        store = self._voltage_store(batch, width)
        voltages = tf.zeros([0]) if store else _empty_list(tf.float32, [batch, width])
        # Frames from resident_from on stay on the GPU.
        resident_from = length - self._PREFETCH
        blocked = _empty_list(tf.int32, [batch, words])
        sampling = _Sampling.of(statistics, width) if statistics.traces else None
        traces = (tf.TensorArray(spike_dtype, size=length),) if sampling else ()
        # Counts are integers up to the window length: float16 spikes count in
        # float16, exact up to 2048, at half float32's traffic (and the counts'
        # cotangent stays float16, as the sequence's is). Whether each step is
        # counted is a device constant: a host scalar would be copied every step.
        count_start, count_stop = statistics.count_window
        count_dtype = (
            tf.float16 if spike_dtype == tf.float16 and count_stop - count_start <= 2048
            else tf.float32
        )
        counts = tf.zeros([batch, width], count_dtype)
        counted = tf.constant(
            [float(count_start <= time < count_stop) for time in range(length)], count_dtype
        )

        # The LGN weights are constant over the rollout: their per-pair sums for
        # the coalesced current sweep are computed once, here, not every step.
        pair_weights = (
            external_pair_weights(
                tf.cast(cell.input_weight_values, cell.variable_dtype), cell.input_csr
            )
            if cell._synaptic_current_backend == "cuda" and cell.compute_dtype == "float16"
            else None
        )

        def step(time, states, counts, frames, voltages, blocked, traces):
            # The statistics read the exposed spikes, as the losses read the
            # sequence, so their gradient arrives as the exposed spikes' one.
            # `call`, not Keras' `__call__`: called directly (outside the RNN
            # layer) a mixed-precision layer's __call__ would autocast the
            # weights to float16, which the RNN layer's rollout does not.
            # The GLIF op writes the packed spikes and blocked mask itself.
            with record_packed_masks(_WORD_BITS) as record:
                spikes, new_states = cell.call(inputs[:, time], states)
            new_states = tf.nest.pack_sequence_as(states, tf.nest.flatten(new_states))
            voltage, refractory = new_states[1], new_states[2]
            packed_spikes, packed_blocked = (
                (record.spike_words, record.blocked_words) if record.spike_words is not None
                else _pack_step(spikes, refractory)
            )
            counts, *values = _record_step(spikes, counts, counted[time], sampling)
            traces = tuple(array.write(time, value) for array, value in zip(traces, values))
            voltages = (
                frame_store.offload_frame(
                    voltage, time, store, length, resident_from, self._COPY_GROUP,
                    codec="tq" if self.voltage_offload == "compressed" else "raw",
                ) if store
                else _push(voltages, voltage)
            )
            return (time + 1, new_states, counts, _push(frames, packed_spikes),
                    voltages, _push(blocked, packed_blocked), traces)

        object.__setattr__(cell, "_lgn_pair_weights", pair_weights)
        try:
            _, states, counts, frames, voltages, blocked, traces = tf.while_loop(
                lambda time, *_: time < length,
                step,
                (tf.constant(0), states, counts, frames, voltages, blocked, traces),
                parallel_iterations=32,
            )
        finally:
            object.__delattr__(cell, "_lgn_pair_weights")
        outputs = (tf.cast(counts, spike_dtype),)
        if traces:
            stacked = traces[0].stack()
            offset = 0
            for group in statistics.traces:
                rows = (group.rows[1] - group.rows[0]) * group.neuron_mask.shape[1]
                start, stop = group.window
                outputs += (tf.transpose(stacked[start:stop, offset:offset + rows]),)
                offset += rows
        final_state = tuple(cell.from_loop_state(states))
        # Every step's BKG counts for the BKG weight gradient, [length, batch,
        # n_bkg], generated as the replay regenerates them (same seed and
        # step, bitwise) but in one launch beside the forward loop.
        bkg_activity = None
        if cell.bkg_input_weights.trainable:
            bkg_activity = tf.stop_gradient(models.sample_bkg_activity(
                cell, batch, initial_state[6], num_steps=length
            ))
        return outputs, final_state, (frames, voltages, blocked, bkg_activity, store)

    def _voltage_store(self, batch, width):
        """The host store key of this rollout's membrane record, or None to
        keep the record on the GPU. Keys count the rollouts of the outermost
        graph being traced (so retraces reuse their stores) and name the
        frame shape; the op keeps one store per key and GPU."""
        if (self.voltage_offload == "none" or self._PREFETCH >= self.sequence_length
                or tf.executing_eagerly() or not frame_store.available()):
            return None
        graph = tf.compat.v1.get_default_graph()
        while getattr(graph, "outer_graph", None) is not None:
            graph = graph.outer_graph
        count = getattr(graph, "_v1_voltage_stores", 0)
        graph._v1_voltage_stores = count + 1
        return f"voltages_{count}_{batch}x{width}"

    # Backward ------------------------------------------------------------

    def _spike_cotangents(self, statistic_gradients, statistics, like):
        """``(carry, at)``: ``at(time, carry)`` returns the spike cotangent the
        dense sequence gradient holds at ``time`` (the counts' cotangent inside
        the count window plus, inside a trace window, the traces' cotangents)
        and the carry for the next call. When the carry is the returned
        cotangent itself, the next call rewrites only the sampled entries of
        it, in place once the caller has released it."""
        length = self.sequence_length
        count_gradient = statistic_gradients[0]
        if count_gradient is None:
            count_gradient = tf.zeros_like(like)
        if statistics.traces:
            # The dense gradient adds the traces' (zero-padded) gradient at
            # every step, which turns a -0 into a +0.
            count_gradient = tf.where(
                tf.equal(count_gradient, 0), tf.zeros_like(count_gradient), count_gradient
            )
        count_start, count_stop = statistics.count_window

        def count_gradient_at(time):
            if (count_start, count_stop) == (0, length):
                return count_gradient
            return tf.where(
                (time >= count_start) & (time < count_stop),
                count_gradient, tf.zeros_like(count_gradient),
            )

        no_carry = tf.zeros([0], like.dtype)
        if not statistics.traces:
            return no_carry, lambda time, carry: (count_gradient_at(time), carry)
        padded = []
        for group, gradient in zip(statistics.traces, statistic_gradients[1:]):
            rows, per_trial, _ = group.neuron_mask.shape
            start, stop = group.window
            if gradient is None:
                gradient = tf.zeros([rows * per_trial, stop - start], like.dtype)
            padded.append(tf.pad(gradient, [[0, 0], [start, length - stop]]))
        trace_gradients = tf.transpose(tf.concat(padded, axis=0))
        sampling = _Sampling.of(statistics, like.shape[1])
        runs, first = _duplicate_runs(sampling.ids)

        # Outside its window a trace's cotangent row is zero: adding those
        # +-0 leaves the (zero-free of -0) counts' cotangent unchanged.
        # Always scattering avoids a per-step branch, whose predicate the
        # device-side switch would copy over from the host every step.
        if (count_start, count_stop) != (0, length):
            return no_carry, lambda time, carry: (_add_traces(
                count_gradient_at(time), trace_gradients[time], sampling, runs, first
            ), carry)

        def spike_cotangent(time, carry):
            # Off the sampled entries every step's cotangent is the counts'
            # one: overwrite only those entries of the previous step's
            # cotangent, in place when it holds the only reference (else
            # the op copies it first, as _add_traces must copy the counts').
            cotangent = overwrite_entries(carry, sampling.ids, _traced_values(
                count_gradient, trace_gradients[time], sampling, runs
            ))
            return cotangent, cotangent

        return count_gradient, spike_cotangent

    def _adjoint(self, inputs, initial_state, final_state, records, statistics,
                 statistic_gradients, state_gradients, variables):
        cell, length, delay = self.cell, self.sequence_length, self.cell.max_delay
        frames, voltages, blocked, bkg_activity, store = records
        final_slots = tf.split(final_state[0], delay, axis=1)
        spike_dtype = final_slots[0].dtype
        width = final_slots[0].shape[1]
        roles = {}
        for variable in variables:
            role = next((name for name, value in self._weight_roles.items()
                         if variable is value), None)
            if role is None:
                raise ValueError(f"The recorded adjoint has no gradient for {variable.name}.")
            roles[role] = variable
        summed = tuple(role for role in ("bkg", "lgn") if role in roles)

        # The cell's inputs to its backward operators, read once.
        constants = tuple(_on_replica_device(value) for value in glif_constants(cell))
        basis = _on_replica_device(tf.cast(cell.synaptic_basis_weights, tf.float32))
        recurrent_weights = _on_replica_device(
            tf.cast(cell.recurrent_weight_values, cell.variable_dtype)
        )
        dampening = _on_replica_device(tf.cast(cell._recurrent_dampening, spike_dtype))
        if cell._track_voltage_penalty:
            penalty = cell._voltage_penalty_mode
            inverse = tf.math.reciprocal(tf.cast(cell._n_neurons, tf.float32))
            penalty_gradient = state_gradients[-1]
            if penalty_gradient is None:
                penalty_gradient = tf.zeros_like(final_state[-1])
        else:
            penalty, inverse = "none", tf.zeros([], tf.float32)
            penalty_gradient = tf.zeros([0], tf.float32)
        no_voltage = tf.zeros([0], spike_dtype)
        no_addend = tf.zeros([0], spike_dtype)
        spike_carry, spike_cotangent_at = self._spike_cotangents(
            statistic_gradients, statistics, final_slots[0]
        )
        lgn_frames = None
        if "lgn" in summed:
            # The LGN frame acting at step t: the delay buffer's frames, oldest
            # first, then the input sequence.
            lgn_frames = inputs
            if cell._lgn_delay_buffer_width:
                buffered = tf.reshape(
                    initial_state[cell._lgn_delay_state_index],
                    [inputs.shape[0], -1, cell.input_dim],
                )
                lgn_frames = tf.concat([buffered[:, ::-1], inputs], axis=1)

        external = {
            "bkg": (cell.bkg_input_csr,
                    _on_replica_device(tf.cast(cell.bkg_input_weights, cell.variable_dtype))),
            "lgn": (cell.input_csr,
                    _on_replica_device(tf.cast(cell.input_weight_values, cell.variable_dtype))),
        }

        def weight_gradients(time, upstream):
            """The BKG and LGN weight gradients of this step, in `summed` order."""
            gradients = []
            for role in summed:
                connectivity, weights = external[role]
                if role == "bkg":
                    values = bkg_activity[time]
                else:
                    values, _ = models._lgn_activity(
                        cell, cell._permute_lgn_input(lgn_frames[:, time, :cell.input_dim])
                    )
                # The dense BKG gradient's one-pass form (tensor or resource
                # connectivity), bitwise the same wherever the fallback's order
                # is fixed.
                gradient = incoming_weight_backward(connectivity, values, upstream, basis)
                if gradient is None and connectivity.resource_name is not None:
                    _, gradient = resource_external_backward(
                        connectivity, values, upstream, weights, basis,
                        compute_activity_gradient=False, compute_weight_gradient=True,
                    )
                elif gradient is None:
                    _, gradient = external_backward(
                        connectivity, values, upstream, weights, basis,
                        connectivity.post_ids, connectivity.synapse_types,
                        connectivity.row_splits, connectivity.edge_ids,
                        connectivity.nonempty_rows, connectivity.pair_ids,
                        connectivity.pair_posts, connectivity.pair_types,
                        compute_activity_gradient=False, compute_weight_gradient=True,
                    )
                gradients.append(gradient)
            return tuple(gradients)

        # The recurrent backward (tensor or resource connectivity) takes the
        # GLIF backward's abs-max of the upstream and the slots' activity masks,
        # carried from step to step like the slots themselves. Both only spare
        # the operator passes over its inputs: same results.
        # FusedGlifStepBackwardPacked reads the recorded words directly (four
        # bases only).
        packed_glif = cell._n_syn_basis == 4

        def recurrent_spike_gradients(slots, masks, upstream, bound, addends):
            connectivity = cell.recurrent_csr
            slots = spike_slots(slots)
            options = dict(
                activity_masks=masks, abs_max_bound=bound, spike_grad_addends=addends
            )
            if connectivity.resource_name is not None:
                gradients, _ = resource_recurrent_backward(
                    connectivity, slots, upstream, recurrent_weights, basis, dampening,
                    **options,
                )
            else:
                gradients, _ = recurrent_backward(
                    connectivity, slots, upstream, recurrent_weights, basis, dampening,
                    connectivity.post_ids, connectivity.synapse_types,
                    connectivity.row_splits, connectivity.edge_ids,
                    connectivity.nonempty_rows, connectivity.pair_ids,
                    connectivity.pair_posts, connectivity.pair_types, **options,
                )
            return gradients

        def frame(index):
            return tf.raw_ops.TensorListGetItem(
                input_handle=frames, index=index,
                element_shape=frames_like.shape, element_dtype=tf.int32,
            )

        def reverse_step(time, voltages, blocked, window, cotangents, partial):
            """Step ``time``'s backward; ``window`` holds z[time - 1 .. time - delay],
            when carried their activity masks, and the spike cotangent's carry."""
            slots, masks, spike_carry = window
            history, voltage_g, asc_g, rise_g, psc_g = cotangents
            if store:
                # After the previous step's GLIF backward (and, at the first
                # step, every offload), so the frame's wait sits right before
                # its consumer on the compute stream.
                with tf.control_dependencies([voltage_g, voltages]):
                    voltage = frame_store.take_frame(
                        time, store, final_state[1],
                        prefetch=self._PREFETCH, group=self._COPY_GROUP,
                    )
            else:
                voltages, voltage = _pop(voltages, final_state[1])
            blocked, packed_blocked = _pop(blocked, frames_like)
            # z[time - 1 - delay], the next step's oldest slot, is frame
            # time - 1; at time 0 the read is unused.
            oldest, step_blocked = _unpack_frames(
                frame(tf.maximum(time - 1, 0)), None if packed_glif else packed_blocked,
                width, spike_dtype,
            )
            spike_cotangent, spike_carry = spike_cotangent_at(time, spike_carry)
            # z[time - 1] is frame time - 1 + delay.
            prev_z, step_blocked = (
                (frame(time - 1 + delay), packed_blocked) if packed_glif
                else (slots[0], step_blocked)
            )
            spike_g, voltage_g, asc_g, rise_g, psc_g, input_g, bound = glif_step_backward(
                cell, prev_z, voltage, step_blocked, constants,
                cell._dt, cell.v_th, cell._gauss_std, cell._dampening_factor, inverse,
                spike_cotangent, history[0], history[1], voltage_g, asc_g,
                rise_g, psc_g, no_voltage, penalty_gradient, penalty=penalty,
                emit_abs_max=True, word_bits=_WORD_BITS if packed_glif else None,
            )
            if cell._scales_recurrent_inputs:
                input_g = input_g * cell._lr_scale
                bound = _scaled_abs_max(bound, cell._lr_scale)
            upstream = tf.reshape(input_g, [-1, cell._n_syn_basis])
            # Each slot's gradient: the recurrent operator's plus what the next
            # step passed back through the slot it moved into (the GLIF
            # backward's for slot 0, which also holds the reset's), added by
            # the operator as it stores its own.
            addends = ((spike_g,) + tuple(history[k + 1] for k in range(1, delay - 1))
                       + (no_addend,))
            history = tuple(recurrent_spike_gradients(
                slots, masks, upstream, bound, addends
            ))
            partial = tuple(
                total + gradient
                for total, gradient in zip(partial, weight_gradients(time, upstream))
            )
            if masks:
                masks = masks[1:] + (spike_activity_masks(oldest),)
            if spike_carry is spike_cotangent:
                # Hand the cotangent on only once the GLIF op has read it, so
                # the next step's update holds its only reference.
                with tf.control_dependencies([spike_g]):
                    spike_carry = tf.identity(spike_carry)
            return (time - 1, voltages, blocked, (slots[1:] + (oldest,), masks, spike_carry),
                    (history, voltage_g, asc_g, rise_g, psc_g), partial)

        frames_like = tf.zeros([final_slots[0].shape[0], -(-width // _WORD_BITS)], tf.int32)
        cotangents = (
            tuple(tf.split(
                self._gradient_or_zeros(state_gradients[0], final_state[0]), delay, axis=1
            )),
        ) + tuple(
            self._gradient_or_zeros(state_gradients[index], final_state[index])
            for index in (1, 3, 4, 5)
        )
        slots = tuple(final_slots[1:]) + (
            models._unpack_spike_words(frame(length - 1), width, spike_dtype, _WORD_BITS),
        )
        masks = tuple(spike_activity_masks(slot) for slot in slots)
        window = (slots, masks, spike_carry)
        totals = tuple(tf.zeros_like(roles[role]) for role in summed)

        def reverse_chunk(chunk, voltages, blocked, window, cotangents, totals):
            start = chunk * self.chunk_size
            stop = tf.minimum(start + self.chunk_size, length)
            time, voltages, blocked, window, cotangents, partial = tf.while_loop(
                lambda time, *_: time >= start,
                reverse_step,
                (stop - 1, voltages, blocked, window, cotangents,
                 tuple(tf.zeros_like(total) for total in totals)),
                parallel_iterations=4,
            )
            totals = tuple(total + chunk_total for total, chunk_total in zip(totals, partial))
            return chunk - 1, voltages, blocked, window, cotangents, totals

        with self._accumulation("recurrent" in roles):
            _, _, _, _, cotangents, totals = tf.while_loop(
                lambda chunk, *_: chunk >= 0,
                reverse_chunk,
                (tf.constant(self.n_chunks - 1), voltages, blocked, window, cotangents,
                 totals),
                parallel_iterations=1,
            )
        gradients = dict(zip(summed, totals))
        if "recurrent" in roles:
            gradients["recurrent"] = self._accumulated_gradient(
                cotangents, roles["recurrent"].dtype
            )
        variable_gradients = tuple(
            gradients[next(role for role, value in roles.items() if value is variable)]
            for variable in variables
        )

        history, voltage_g, asc_g, rise_g, psc_g = cotangents
        state_gradient = {
            0: tf.concat(history, axis=1), 1: voltage_g, 3: asc_g, 4: rise_g, 5: psc_g,
        }
        if cell._lgn_delay_buffer_width:
            # Shifted frame by frame and never read back as a differentiable
            # input, the final buffer's gradient reaches only the initial
            # buffer frames that survive the rollout.
            buffer_gradient = self._gradient_or_zeros(
                state_gradients[cell._lgn_delay_state_index],
                final_state[cell._lgn_delay_state_index],
            )
            shift = min(length * cell.input_dim, cell._lgn_delay_buffer_width)
            state_gradient[cell._lgn_delay_state_index] = tf.concat(
                [buffer_gradient[:, shift:], tf.zeros_like(buffer_gradient[:, :shift])],
                axis=1,
            )
        if cell._track_voltage_penalty:
            state_gradient[len(final_state) - 1] = penalty_gradient
        return (
            tuple(state_gradient.get(index) for index in range(len(final_state))),
            variable_gradients,
        )

    @staticmethod
    def _gradient_or_zeros(gradient, value):
        return tf.zeros_like(value) if gradient is None else gradient
