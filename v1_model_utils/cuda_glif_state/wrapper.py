"""Deep adapter for the differentiable GLIF state and spike transition."""

from pathlib import Path

import tensorflow as tf

from v1_model_utils.cuda_operator_cache import ensure_artifact


HERE = Path(__file__).resolve().parent
BUILD_FLAGS = ("--expt-relaxed-constexpr",)
_GLIF_OPS = None


def _gradient_like(gradient, output):
    """Return an upstream gradient with the forward output's shape and dtype."""
    if gradient is None:
        return tf.zeros_like(output)
    gradient = tf.cast(gradient, output.dtype)
    if gradient.shape.rank == output.shape.rank and gradient.shape.is_compatible_with(
        output.shape
    ):
        return gradient
    gradient = tf.broadcast_to(gradient, tf.shape(output))
    return tf.ensure_shape(gradient, output.shape)


def _load_ops():
    global _GLIF_OPS
    if _GLIF_OPS is None:
        glif = ensure_artifact(
            HERE,
            "glif_state_ops",
            sources=(
                HERE / "build.py",
                HERE / "glif_state_ops.cc",
                HERE / "glif_state_ops.cu.cc",
            ),
            build_module="v1_model_utils.cuda_glif_state.build",
            build_flags=BUILD_FLAGS,
        )
        _GLIF_OPS = tf.load_op_library(str(glif))
    return _GLIF_OPS


def update_glif_state(
    z_history, v, r, asc, psc_rise, psc, rec_inputs, *, cell, penalty_acc=None, bkg=None
):
    """Return spikes, the exposed voltage and the six next recurrent state
    tensors, followed by the advanced voltage-penalty accumulator when
    `penalty_acc` is given.

    `z_history` is the delayed spike history, newest slot first: either a
    tuple of [batch, neurons] slot tensors, or the concatenated
    [batch, slots * neurons] buffer. The next history comes back in the same
    form. The op reads only the newest slot, and shifting a tuple is a
    relabelling of tensors: the spikes become the new first slot and the
    oldest slot is dropped, so no history byte is copied. The buffer form
    splits and re-concatenates around that and costs two copies.

    One fused op updates the state and thresholds the membrane, so the
    float32 membrane is written once and never read back by a separate op.
    The exposed voltage is `new_v` in the compute dtype when the cell returns
    voltage sequences, and None otherwise. Given the [batch, 1]
    float32 `penalty_acc`, the op also adds this step's neuron-mean voltage
    penalty (the cell's range or threshold mode) to it.

    `bkg` (from V1Column._fused_bkg_inputs) makes the op add the four-edge
    background current to `rec_inputs` itself, bit for bit as BkgCsrForward
    would; the BKG weight gradient is then computed here from the op's input
    gradient, with the same kernel and metadata BkgCsrForward's gradient uses.
    """
    glif_ops = _load_ops()
    buffered = not isinstance(z_history, (tuple, list))
    slots = (
        tuple(tf.split(z_history, z_history.shape[1] // v.shape[1], axis=1))
        if buffered else tuple(z_history)
    )
    n_slots = len(slots)
    emit_voltage = cell._return_voltage_sequences
    if penalty_acc is None:
        penalty = "none"
        penalty_acc = tf.zeros([0], tf.float32)
        inverse_neurons = tf.zeros([], tf.float32)  # unread without a penalty
    else:
        penalty = cell._voltage_penalty_mode
        inverse_neurons = tf.math.reciprocal(tf.cast(cell._n_neurons, tf.float32))
    connectivity = None if bkg is None else bkg["connectivity"]
    if bkg is None:
        bkg_tensors = (tf.zeros([0, 0], slots[0].dtype), tf.zeros([0], tf.float32),
                       tf.zeros([0, 4], tf.float32))
        bkg_metadata = ()
    else:
        bkg_tensors = (bkg["activity"], bkg["weights"], bkg["basis"])
        # The gather metadata, then the weight-backward metadata BkgCsrForward's
        # gradient takes: all passed as arguments, so the retraced step
        # captures no tensor (see the constants note below).
        bkg_metadata = (
            connectivity.incoming_pre_ids, connectivity.incoming_edge_ids,
            connectivity.incoming_types, connectivity.post_ids,
            connectivity.synapse_types, connectivity.row_splits,
            connectivity.edge_ids, connectivity.nonempty_rows,
            connectivity.pair_ids, connectivity.pair_posts, connectivity.pair_types,
        )
    empty_gather = (tf.zeros([0], tf.uint32), tf.zeros([0], tf.uint32),
                    tf.zeros([0], tf.uint8))
    # The derived float32 constants, in the order both ops take them. They get
    # no gradient, but they are passed through the custom gradient rather than
    # closed over: a tensor it captures is rejected when the step is retraced
    # inside the segmented-recompute backward pass.
    constants = (
        cell.syn_coeffs, cell.asc_decay, cell.asc_amps, cell.decay,
        cell.asc_factor, cell.reset_coeff, cell.asc_spike_factor,
    )

    # The outputs end with the next history: the spikes again (a separate
    # output, so the gradient of the exposed spike sequence and that of the
    # newest slot arrive apart and the kernel sums them), then every slot but
    # the oldest, passed through.
    @tf.custom_gradient
    def transition(voltage, refractory, adaptation, rise, postsynaptic,
                   inputs, accumulator, bkg_activity, bkg_weights, bkg_basis,
                   t_ref_steps, dt, v_reset, v_th, sigma, amplitude, inverse,
                   *rest):
        history = rest[:n_slots]
        constants, metadata = rest[n_slots:n_slots + 7], rest[n_slots + 7:]
        prev_z = history[0]
        gather = metadata[:3] if metadata else empty_gather
        outputs = glif_ops.fused_glif_step(
            prev_z, voltage, refractory, adaptation, rise, postsynaptic, inputs,
            *constants, t_ref_steps, dt, v_reset, v_th, accumulator, inverse,
            bkg_activity, bkg_weights, *gather, bkg_basis,
            hard_reset=cell._hard_reset, emit_voltage=emit_voltage,
            penalty=penalty,
        )
        (spikes, new_v, _, new_asc, new_rise, new_psc, blocked,
         exposed, new_accumulator) = outputs

        def grad(gs, gv, _gr, ga, grise, gpsc, _gb, gvoltage, gacc, gz, *gpass):
            # The accumulator adds this step's penalty to the running sum, so its
            # gradient passes through unchanged and also drives the membrane.
            gacc = (_gradient_like(gacc, new_accumulator) if penalty != "none"
                    else tf.zeros([0], tf.float32))
            # Only the previous spikes, the membrane and the refractory mask
            # (new_r > 0) enter the backward graph. The refractory counter stays
            # out: its dtype has no GPU TensorList kernel, so retaining its
            # per-timestep history would stage it through host memory.
            # gpass[k] belongs to slot k, passed on as the next step's slot
            # k + 1. The kernel adds gpass[0] to prev_z's own gradient; the
            # others pass straight back, and the oldest slot, dropped from the
            # next history, gets none.
            prev_grad = (_gradient_like(gpass[0], prev_z) if gpass
                         else tf.zeros([0], prev_z.dtype))
            zg, vg, ag, rg, pg, ig = glif_ops.fused_glif_step_backward(
                prev_z, new_v, blocked, *constants, dt, v_th, sigma, amplitude,
                _gradient_like(gs, spikes), _gradient_like(gz, spikes), prev_grad,
                _gradient_like(gv, new_v), _gradient_like(ga, new_asc),
                _gradient_like(grise, new_rise), _gradient_like(gpsc, new_psc),
                _gradient_like(gvoltage, exposed), gacc, inverse,
                surrogate=cell._surrogate_gradient,
                hard_reset=cell._hard_reset,
                detach_reset=cell._detach_reset,
                detach_asc_reset=cell._detach_asc_reset,
                emit_voltage=emit_voltage, penalty=penalty,
            )
            accumulator_grad = gacc if penalty != "none" else None
            bkg_weight_grad = None
            if metadata and bkg["trainable"]:
                from v1_model_utils.cuda_csr_external.wrapper import _load_ops as external_ops
                bkg_weight_grad = external_ops()[1].external_csr_weight_backward(
                    bkg_activity, tf.reshape(ig, [-1, 4]), *metadata[3:8], bkg_basis,
                    *metadata[8:], sparse_activity=connectivity.sparse_activity,
                    n_post=connectivity.n_post, n_edges=connectivity.n_edges,
                )
            history_grads = (zg, *gpass[1:], None)[:n_slots]
            return ((vg, None, ag, rg, pg, ig, accumulator_grad, None,
                     bkg_weight_grad, None) + (None,) * 7 + history_grads
                    + (None,) * (len(rest) - n_slots))

        return outputs + (spikes,) + tuple(history[:-1]), grad

    (spikes, new_v, new_r, new_asc, new_rise, new_psc, _, exposed,
     new_accumulator, *new_history) = transition(
        v, r, asc, psc_rise, psc, rec_inputs, penalty_acc, *bkg_tensors,
        cell.t_ref_steps, cell._dt, cell.v_reset, cell.v_th, cell._gauss_std,
        cell._dampening_factor, inverse_neurons, *slots, *constants, *bkg_metadata,
    )
    new_history = tf.concat(new_history, axis=1) if buffered else tuple(new_history)
    outputs = (
        spikes,
        exposed if emit_voltage else None,
        (new_history, new_v, new_r, new_asc, new_rise, new_psc),
    )
    return outputs if penalty == "none" else (*outputs, new_accumulator)
