"""Deep adapter for the differentiable GLIF state and spike transition."""

from pathlib import Path

import tensorflow as tf

from v1_model_utils.cuda_operator_cache import ensure_artifact


HERE = Path(__file__).resolve().parent
BUILD_FLAGS = ("--expt-relaxed-constexpr",)
_GLIF_OPS = None
_SPIKE_OPS = None


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
    global _GLIF_OPS, _SPIKE_OPS
    if _GLIF_OPS is None:
        sources = (HERE / "build.py",)
        glif = ensure_artifact(
            HERE,
            "glif_state_ops",
            sources=sources + (HERE / "glif_state_ops.cc", HERE / "glif_state_ops.cu.cc"),
            build_module="v1_model_utils.cuda_glif_state.build",
            build_flags=BUILD_FLAGS,
        )
        spike = ensure_artifact(
            HERE,
            "spike_history_ops",
            sources=sources + (
                HERE / "spike_history_ops.cc",
                HERE / "spike_history_ops.cu.cc",
            ),
            build_module="v1_model_utils.cuda_glif_state.build",
            build_flags=BUILD_FLAGS,
        )
        _GLIF_OPS = tf.load_op_library(str(glif))
        _SPIKE_OPS = tf.load_op_library(str(spike))
    return _GLIF_OPS, _SPIKE_OPS


def _dense_state(
    prev_z, v, r, asc, psc_rise, psc, rec_inputs, *, cell
):
    glif_ops, _ = _load_ops()

    @tf.custom_gradient
    def transition(
        z,
        voltage,
        refractory,
        adaptation,
        rise,
        postsynaptic,
        inputs,
        syn_decay,
        psc_initial,
        asc_decay,
        asc_amps,
        decay,
        current_factor,
        t_ref_steps,
        dt,
        v_reset,
    ):
        outputs = glif_ops.fused_glif_single_forward(
            z, voltage, refractory, adaptation, rise, postsynaptic, inputs,
            syn_decay, psc_initial, asc_decay, asc_amps, decay, current_factor,
            t_ref_steps, dt, v_reset,
            hard_reset=cell._hard_reset,
        )

        def grad(gv, _gr, ga, grise, gpsc):
            gv = _gradient_like(gv, outputs[0])
            ga = _gradient_like(ga, outputs[2])
            grise = _gradient_like(grise, outputs[3])
            gpsc = _gradient_like(gpsc, outputs[4])
            # The kernel reads the refractory state only to mask the voltage
            # gradient under hard reset, so under soft reset the value is dead.
            # Referencing the forward tensor anyway would make TensorFlow retain
            # the whole per-timestep history, and its dtype has no GPU TensorList
            # kernel, so that history is staged through host memory: 13,840
            # transfers of 6.5 MB and 21% of a batch-32 training step. Passing
            # zeros keeps it out of the backward graph; gradients are identical.
            backward_refractory = (
                refractory
                if cell._hard_reset
                else tf.zeros_like(z, dtype=refractory.dtype)
            )
            zg, vg, ag, rg, pg, ig = glif_ops.fused_glif_single_backward(
                z, backward_refractory, adaptation, rise,
                syn_decay, psc_initial, asc_decay, decay, current_factor,
                t_ref_steps, asc_amps, dt, gv, ga, grise, gpsc,
                hard_reset=cell._hard_reset,
                detach_reset=cell._detach_reset,
                detach_asc_reset=cell._detach_asc_reset,
            )
            return (zg, vg, None, ag, rg, pg, ig) + (None,) * 9

        return outputs, grad

    return transition(
        prev_z,
        v,
        r,
        asc,
        psc_rise,
        psc,
        rec_inputs,
        tf.cast(cell.syn_decay, v.dtype),
        tf.cast(cell.psc_initial, v.dtype),
        tf.cast(cell.asc_decay, v.dtype),
        tf.cast(cell.asc_amps, v.dtype),
        tf.cast(cell.decay, v.dtype),
        tf.cast(cell.current_factor, v.dtype),
        cell.t_ref_steps,
        tf.cast(cell._dt, v.dtype),
        tf.cast(cell.v_reset, v.dtype),
    )


def _spike_and_shift(voltage, refractory, history, *, cell):
    _, spike_ops = _load_ops()
    refractory = tf.cast(refractory, tf.bool)

    @tf.custom_gradient
    def transition(voltage_value, refractory_value, history_value, dampening):
        spikes, new_history = spike_ops.v1_fused_spike_shift(
            voltage_value, refractory_value, history_value
        )

        def grad(spike_gradient, history_gradient):
            spike_gradient = _gradient_like(spike_gradient, spikes)
            history_gradient = _gradient_like(history_gradient, new_history)
            voltage_gradient, history_gradient_old = (
                spike_ops.v1_fused_spike_shift_backward(
                    voltage_value, refractory_value, spike_gradient,
                    history_gradient,
                    dampening,
                )
            )
            return voltage_gradient, None, history_gradient_old, None

        return (spikes, new_history), grad

    return transition(
        voltage,
        refractory,
        history,
        tf.cast(cell._dampening_factor, voltage.dtype),
    )


def update_glif_state(
    z_buf, v, r, asc, psc_rise, psc, rec_inputs, *, cell
):
    """Return spikes and the six next recurrent state tensors.

    The pseudo-Gaussian surrogate intentionally remains on the TensorFlow
    reference path because this CUDA adapter implements the triangular
    surrogate only.
    """
    if cell._pseudo_gauss:
        raise ValueError("CUDA state transition does not support pseudo_gauss")
    prev_z = z_buf[:, :cell._n_neurons]
    new_v, new_r, new_asc, new_rise, new_psc = _dense_state(
        prev_z, v, r, asc, psc_rise, psc, rec_inputs, cell=cell
    )
    spikes, new_history = _spike_and_shift(
        new_v - cell.v_th, new_r > 0, z_buf, cell=cell
    )
    return spikes, (new_history, new_v, new_r, new_asc, new_rise, new_psc)
