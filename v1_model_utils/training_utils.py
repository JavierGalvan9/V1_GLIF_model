
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


def parse_delays(delays):
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
    return pre_delay, post_delay


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

