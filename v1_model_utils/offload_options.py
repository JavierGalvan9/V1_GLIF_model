"""Shared offload modes and optional-value command-line syntax."""

VOLTAGE_OFFLOAD_MODES = ("none", "adaptive", "raw", "compressed")
HISTORY_OFFLOAD_MODES = ("none", "adaptive", "raw")


def normalize_offload_args(argv):
    """Expand bare offload flags to adaptive, leaving explicit values intact.

    Abseil enums require a value, so normalize before training bootstrap builds
    worker commands. Invalid explicit values remain for the CLI parser to reject.
    Arguments after the standard ``--`` terminator are never rewritten.
    """
    argv = tuple(argv)
    result = []
    flags = {"--voltage_offload", "--history_offload"}
    for index, value in enumerate(argv):
        if value == "--":
            result.extend(argv[index:])
            break
        if value in flags and (index + 1 == len(argv) or argv[index + 1].startswith("-")):
            value += "=adaptive"
        result.append(value)
    return tuple(result)
