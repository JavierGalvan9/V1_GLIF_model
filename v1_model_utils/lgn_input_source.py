"""LGN input frames for a rollout without its full [batch, seq_len, n_input] input.

A rollout reads one [batch, n_input] LGN frame per time step. A ``FrameSource``
returns that frame, ``frame(tensors, time)``, as a pure function of its
``tensors`` and the step, so the recorded adjoint can pass the tensors through
its custom gradient and read any frame again in its reverse loop. A rollout
loop reads the frames in order through ``advance``, which may carry a buffer
from step to step.

- ``DenseFrames``: a [batch, time, n_input] tensor (the current input).
- ``ConcatFrames``: the batch rows of several sources (the grating and gray
  halves), so the full-batch concatenation never exists.
- ``PackedFrames``: bit-packed frames, [batch, time, words] uint32 of 32 spikes
  per word (models._pack_spike_words), 8 times smaller than bool. uint32, not
  int32: TensorFlow keeps a function's int32 outputs in host memory, uint32
  ones on the GPU (when packed inside a tf.function; eager uint32 ops run on
  the CPU).
- ``DelayedFrames``: the frames acting on the steps of a cell whose LGN input
  arrives late (its delay buffer's, then a source's).

Gradients: the recorded adjoint hands each step's frame cotangent (the LGN
activity gradient, cell._compute_lgn_activity_gradient) to ``add_gradient``,
which a reverse loop carries in ``gradient_buffer`` and ``gradients`` returns
per tensor. Only float frames carry one (``differentiable``): bool and packed
frames end the gradient at the spikes, as the bool model input does. A source
whose float frames are a function of LGN parameters (a straight-through frame
``spikes + p - stop_gradient(p)``) gets their gradient from the default
``add_gradient``, which regenerates the frame under a tape.
"""

import tensorflow as tf

from v1_model_utils import models

_WORD_BITS = 32


def _cond(predicate, true_fn, false_fn):
    """tf.cond, or the branch itself for a Python bool (a step given as an int)."""
    if isinstance(predicate, bool):
        return true_fn() if predicate else false_fn()
    return tf.cond(predicate, true_fn, false_fn)


class FrameSource:
    """``tensors``: the tensors the frames are a function of. ``frame(tensors,
    time)``: the [batch, n_input] frame at ``time`` (bool, or float32 with a
    straight-through gradient)."""

    batch = None
    differentiable = False

    def frame(self, tensors, time):
        raise NotImplementedError

    def gradient_buffer(self, tensors, length):
        """The reverse loop's carry of the gradients of ``tensors``."""
        return tuple(tf.zeros_like(tensor) if self.differentiable and tensor.dtype.is_floating
                     else tf.zeros([0]) for tensor in tensors)

    def add_gradient(self, tensors, time, frame_gradient, buffer):
        """``buffer`` plus the vector-Jacobian product of ``frame(tensors,
        time)`` with ``frame_gradient()`` (called only when the frame carries
        a gradient): the frame is regenerated under a tape."""
        if not self.differentiable:
            return buffer
        # Identities inside the caller's loop body: a tape cannot watch the
        # loop's captures of the outer tensors.
        tensors = tuple(tf.identity(tensor) for tensor in tensors)
        floating = [tensor for tensor in tensors if tensor.dtype.is_floating]
        with tf.GradientTape() as tape:
            tape.watch(floating)
            frame = self.frame(tensors, time)
        products = iter(tape.gradient(
            frame, floating, output_gradients=tf.cast(frame_gradient(), frame.dtype),
            unconnected_gradients=tf.UnconnectedGradients.ZERO))
        return tuple(total + next(products) if tensor.dtype.is_floating else total
                     for tensor, total in zip(tensors, buffer))

    def gradients(self, tensors, buffer):
        """The gradients of ``tensors`` (None for the ones without)."""
        return tuple(total if self.differentiable and tensor.dtype.is_floating else None
                     for tensor, total in zip(tensors, buffer))

    def initial_buffer(self):
        return tf.zeros([0], tf.bool)

    def advance(self, tensors, time, buffer):
        """``(frame, buffer)`` of step ``time``, the steps read in order."""
        return self.frame(tensors, time), buffer

    def materialize(self, tensors, length):
        """The full [batch, length, n_input] input (for the sequence outputs)."""
        return tf.stack([self.frame(tensors, time) for time in range(length)], axis=1)


class DenseFrames(FrameSource):
    def __init__(self, inputs):
        self.tensors = (inputs,)
        self.batch = inputs.shape[0]
        self.differentiable = inputs.dtype.is_floating

    def frame(self, tensors, time):
        return tensors[0][:, time]

    def materialize(self, tensors, length):
        return tensors[0]

    # One write per frame: a tape's slice gradient would add a whole
    # [batch, time, n_input] tensor every step. Every frame gets its write
    # (it acts on a step, or ends in the cell's LGN delay buffer).
    def gradient_buffer(self, tensors, length):
        if not self.differentiable:
            return super().gradient_buffer(tensors, length)
        inputs = tensors[0]
        return (tf.TensorArray(inputs.dtype, size=inputs.shape[1],
                               element_shape=(inputs.shape[0], inputs.shape[2])),)

    def add_gradient(self, tensors, time, frame_gradient, buffer):
        if not self.differentiable:
            return buffer
        return (buffer[0].write(time, tf.cast(frame_gradient(), tensors[0].dtype)),)

    def gradients(self, tensors, buffer):
        if not self.differentiable:
            return (None,)
        return (tf.transpose(buffer[0].stack(), (1, 0, 2)),)


class DelayedFrames(FrameSource):
    """The frames acting on the steps of a cell whose LGN input arrives late:
    the ``buffered`` [batch, held, n_input] frames of its delay buffer, oldest
    first, then the ``source``'s, whose own gradients flow only with
    ``source_gradients``."""

    def __init__(self, buffered, source, source_gradients):
        self.buffered, self.source = DenseFrames(buffered), source
        self.held, self.source_gradients = buffered.shape[1], source_gradients
        self.tensors = (buffered,) + tuple(source.tensors)
        self.batch, self.differentiable = source.batch, True

    def frame(self, tensors, time):
        buffered = tensors[0]
        return _cond(
            time < self.held,
            lambda: buffered[:, tf.minimum(time, self.held - 1)],
            lambda: tf.cast(self.source.frame(tensors[1:], tf.maximum(time - self.held, 0)),
                            buffered.dtype),
        )

    def gradient_buffer(self, tensors, length):
        source = (self.source.gradient_buffer(tensors[1:], length) if self.source_gradients
                  else tuple(tf.zeros([0]) for _ in tensors[1:]))
        return self.buffered.gradient_buffer(tensors[:1], self.held), source

    def add_gradient(self, tensors, time, frame_gradient, buffer):
        held, source = buffer

        def to_buffered():
            return self.buffered.add_gradient(
                tensors[:1], tf.minimum(time, self.held - 1), frame_gradient, held), source

        def to_source():
            if not self.source_gradients:
                return held, source
            return held, self.source.add_gradient(
                tensors[1:], tf.maximum(time - self.held, 0), frame_gradient, source)

        return _cond(time < self.held, to_buffered, to_source)

    def gradients(self, tensors, buffer):
        held, source = buffer
        return self.buffered.gradients(tensors[:1], held) + (
            self.source.gradients(tensors[1:], source) if self.source_gradients
            else (None,) * len(source))


def pack_frames(spikes):
    """[batch, time, n_input] binary -> PackedFrames' [batch, time, words] uint32."""
    batch, length, width = spikes.shape
    words = models._pack_spike_words(tf.reshape(spikes, (-1, width)), _WORD_BITS, tf.uint32)
    return tf.reshape(words, (batch, length, -1))


class PackedFrames(FrameSource):
    def __init__(self, words, width):
        self.tensors = (words,)
        self.batch, self.width = words.shape[0], width

    def frame(self, tensors, time):
        return models._unpack_spike_words(tensors[0][:, time], self.width, tf.bool, _WORD_BITS)

    def materialize(self, tensors, length):
        words = tensors[0]
        spikes = models._unpack_spike_words(
            tf.reshape(words, (-1, words.shape[-1])), self.width, tf.bool, _WORD_BITS)
        return tf.reshape(spikes, (self.batch, length, self.width))


class ConcatFrames(FrameSource):
    """The batch rows of ``sources``, in order."""

    def __init__(self, *sources):
        self.sources = sources
        self.tensors = tuple(tensor for source in sources for tensor in source.tensors)
        self.batch = sum(source.batch for source in sources)
        self.differentiable = any(source.differentiable for source in sources)

    def _split(self, tensors):
        parts, start = [], 0
        for source in self.sources:
            parts.append(tensors[start:start + len(source.tensors)])
            start += len(source.tensors)
        return parts

    def frame(self, tensors, time):
        return tf.concat([source.frame(part, time)
                          for source, part in zip(self.sources, self._split(tensors))], axis=0)

    def initial_buffer(self):
        return tuple(source.initial_buffer() for source in self.sources)

    def advance(self, tensors, time, buffer):
        frames, buffers = zip(*(
            source.advance(part, time, part_buffer) for source, part, part_buffer
            in zip(self.sources, self._split(tensors), buffer)))
        return tf.concat(frames, axis=0), tuple(buffers)

    def gradient_buffer(self, tensors, length):
        return tuple(source.gradient_buffer(part, length)
                     for source, part in zip(self.sources, self._split(tensors)))

    def add_gradient(self, tensors, time, frame_gradient, buffer):
        if not self.differentiable:
            return buffer
        parts = tf.split(frame_gradient(), [source.batch for source in self.sources], axis=0)
        return tuple(
            source.add_gradient(part, time, lambda gradient=gradient: gradient, part_buffer)
            for source, part, gradient, part_buffer
            in zip(self.sources, self._split(tensors), parts, buffer))

    def gradients(self, tensors, buffer):
        return tuple(gradient for source, part, part_buffer
                     in zip(self.sources, self._split(tensors), buffer)
                     for gradient in source.gradients(part, part_buffer))


def as_source(inputs):
    return inputs if isinstance(inputs, FrameSource) else DenseFrames(inputs)
