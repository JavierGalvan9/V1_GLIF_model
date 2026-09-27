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
    z_buf, v, r, asc, psc_rise, psc, rec_inputs, *, cell
):
    """Return spikes, the exposed voltage and the six next recurrent state tensors.

    One fused op updates the state, thresholds the membrane and shifts the
    spike history, so the float32 membrane is written once and never read back
    by a separate op. The exposed voltage is `new_v` in the compute dtype when
    the cell returns voltage sequences, and None otherwise.
    """
    glif_ops = _load_ops()
    emit_voltage = cell._return_voltage_sequences
    # The derived float32 constants, in the order both ops take them. They get
    # no gradient, but they are passed through the custom gradient rather than
    # closed over: a tensor it captures is rejected when the step is retraced
    # inside the segmented-recompute backward pass.
    constants = (
        cell.syn_coeffs, cell.asc_decay, cell.asc_amps, cell.decay,
        cell.asc_factor, cell.reset_coeff, cell.asc_spike_factor,
    )

    @tf.custom_gradient
    def transition(z_buf, voltage, refractory, adaptation, rise, postsynaptic,
                   inputs, t_ref_steps, dt, v_reset, v_th, sigma, amplitude,
                   *constants):
        outputs = glif_ops.fused_glif_step(
            z_buf, voltage, refractory, adaptation, rise, postsynaptic, inputs,
            *constants, t_ref_steps, dt, v_reset, v_th,
            hard_reset=cell._hard_reset, emit_voltage=emit_voltage,
        )
        spikes, new_z_buf, new_v, _, new_asc, new_rise, new_psc, blocked, exposed = outputs

        def grad(gs, gz, gv, _gr, ga, grise, gpsc, _gb, gvoltage):
            # Only the spike history, the membrane and the refractory mask
            # (new_r > 0) enter the backward graph. The refractory counter stays
            # out: its dtype has no GPU TensorList kernel, so retaining its
            # per-timestep history would stage it through host memory.
            zg, vg, ag, rg, pg, ig = glif_ops.fused_glif_step_backward(
                z_buf, new_v, blocked, *constants, dt, v_th, sigma, amplitude,
                _gradient_like(gs, spikes), _gradient_like(gz, new_z_buf),
                _gradient_like(gv, new_v), _gradient_like(ga, new_asc),
                _gradient_like(grise, new_rise), _gradient_like(gpsc, new_psc),
                _gradient_like(gvoltage, exposed),
                surrogate=cell._surrogate_gradient,
                hard_reset=cell._hard_reset,
                detach_reset=cell._detach_reset,
                detach_asc_reset=cell._detach_asc_reset,
                emit_voltage=emit_voltage,
            )
            return (zg, vg, None, ag, rg, pg, ig) + (None,) * (6 + len(constants))

        return outputs, grad

    spikes, new_z_buf, new_v, new_r, new_asc, new_rise, new_psc, _, exposed = transition(
        z_buf, v, r, asc, psc_rise, psc, rec_inputs, cell.t_ref_steps,
        cell._dt, cell.v_reset, cell.v_th, cell._gauss_std,
        cell._dampening_factor, *constants,
    )
    return (
        spikes,
        exposed if emit_voltage else None,
        (new_z_buf, new_v, new_r, new_asc, new_rise, new_psc),
    )
