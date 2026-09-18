import numpy as np


def reset_metric(metric):
    """Reset a Keras metric across TF/Keras versions."""
    if hasattr(metric, "reset_state"):
        metric.reset_state()
    elif hasattr(metric, "reset_states"):
        metric.reset_states()
    else:
        raise AttributeError(f"Metric {type(metric)} has no reset_state/reset_states method.")

def reset_metrics(metrics):
    for metric in metrics:
        reset_metric(metric)


MAX_LGN_INPUTS = 17400


def parse_delays(delays, seq_len=None):
    parts = [a.strip() for a in str(delays).split(",") if a.strip() != ""]
    if len(parts) != 2:
        raise ValueError(
            f"Invalid --delays value '{delays}'. Expected two comma-separated integers, e.g. '100,0'."
        )
    pre_delay, post_delay = int(parts[0]), int(parts[1])
    if pre_delay < 0 or post_delay < 0:
        raise ValueError(
            f"Invalid --delays value '{delays}'. Delays must be non-negative."
        )
    if seq_len is not None and pre_delay + post_delay >= seq_len:
        raise ValueError(
            f"Invalid --delays value '{delays}' for seq_len={seq_len}. "
            "Delay trimming must leave at least one timestep."
        )
    return pre_delay, post_delay


def validate_n_input(n_input, max_n_input=MAX_LGN_INPUTS):
    """Return a valid LGN input count or raise a user-facing error."""
    n_input = int(n_input)
    if not 1 <= n_input <= max_n_input:
        raise ValueError(
            f"n_input must be between 1 and {max_n_input}, got {n_input}."
        )
    return n_input


def infer_effective_sequence_length(flags):
    """Infer logged sequence length from training mode."""
    seq_len = int(flags.seq_len)
    has_split_training_flags = hasattr(flags, "spontaneous_training") or hasattr(
        flags, "sequential_stimuli"
    )
    if not has_split_training_flags:
        return seq_len
    if getattr(flags, "spontaneous_training", False):
        return seq_len
    if getattr(flags, "sequential_stimuli", False):
        return seq_len
    return 2 * seq_len


def require_int32_safe_rnn_output(batch_size, seq_len, n_neurons,
                                  full_tensor_element_limit=None):
    """Refuse an un-checkpointed run whose RNN output exceeds the int32 limit.

    Without gradient checkpointing, Keras assembles the RNN output with
    ``TensorArray.stack()`` -- an axis-0 concat of ``seq_len`` slices. With 16
    or more inputs TensorFlow's concat switches to a pointer-array GPU kernel
    whose offsets are int32, so above 2**31 elements it returns the wrong
    spikes with no error raised at all (measured on TF 2.21: 99.9% of the
    elements wrong for a 500-slice stack at 1.52x the limit), and every loss
    or statistic downstream is then computed on garbage. There is no cheap way
    to make Keras' stack safe, so refuse the configuration rather than produce
    plausible-looking numbers.

    The segmented runner is unaffected: it rejoins its chunks along axis 1,
    where TensorFlow picks the index type from ``seq_len * n_neurons`` rather
    than from the element count -- measured exact at 1.52x the limit with both
    2 and 100 chunks.
    """
    if full_tensor_element_limit is None:
        full_tensor_element_limit = int(np.iinfo(np.int32).max)
    elements = int(batch_size) * int(seq_len) * int(n_neurons)
    if elements > full_tensor_element_limit:
        raise ValueError(
            "Gradient checkpointing is disabled, but the RNN output "
            f"[{batch_size}, {seq_len}, {n_neurons}] is {elements:,} elements, "
            f"past the {full_tensor_element_limit:,} TensorFlow's stock GPU "
            "kernels index correctly. Keras assembles that output with "
            "TensorArray.stack(), which is silently wrong above the limit. "
            "Pass --gradient_checkpointing, or lower the per-replica batch "
            "size / --seq_len so the product stays under it."
        )
    return elements
