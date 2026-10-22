"""Per-device stores that park GPU frames in pinned host memory."""

from .wrapper import available, offload_frame, take_frame

__all__ = ["available", "offload_frame", "take_frame"]
