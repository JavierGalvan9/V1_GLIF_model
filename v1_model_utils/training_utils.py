
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
