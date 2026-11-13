"""Frame sources read the frames of their input, and their gradient hook,
driven by a reverse tf.while_loop as the recorded adjoint drives it, returns
what a tape over the frames returns."""

import numpy as np
import pytest
import tensorflow as tf

from v1_model_utils import lgn_input_source as sources

BATCH, STEPS, WIDTH = 3, 7, 70  # 70 inputs: three words, the last one partial


class _Logistic(sources.FrameSource):
    """Float frames that are a function of parameters, as a straight-through
    LGN frame is (the default, taped gradient hook)."""

    differentiable = True

    def __init__(self, parameters):
        self.tensors = (parameters,)
        self.batch = BATCH

    def frame(self, tensors, time):
        return tf.broadcast_to(tf.sigmoid(tensors[0][time]), (BATCH, WIDTH))


def _spikes(seed):
    return tf.constant(np.random.default_rng(seed).random((BATCH, STEPS, WIDTH)) < 0.3)


def _gradients(source, tensors, cotangents, last):
    """``source``'s gradients from a reverse loop over steps ``last`` .. 0
    (cotangents[t] at step t), and a tape's over the same frames."""

    @tf.function
    def hooked(tensors, cotangents):
        def step(time, buffer):
            return time - 1, source.add_gradient(
                tensors, time, lambda: cotangents[time], buffer)

        _, buffer = tf.while_loop(lambda time, _: time >= 0, step,
                                  (tf.constant(last), source.gradient_buffer(tensors, last + 1)))
        return source.gradients(tensors, buffer)

    @tf.function
    def taped(tensors, cotangents):
        floating = [tensor for tensor in tensors if tensor.dtype.is_floating]
        with tf.GradientTape() as tape:
            tape.watch(floating)
            loss = sum(tf.reduce_sum(tf.cast(source.frame(tensors, time), tf.float32)
                                     * cotangents[time]) for time in range(last + 1))
        return tape.gradient(loss, floating)

    return hooked(tensors, cotangents), taped(tensors, cotangents)


def test_packed_and_concatenated_frames_are_the_inputs():
    grating, gray = _spikes(0), _spikes(1)
    packed = sources.PackedFrames(sources.pack_frames(gray), WIDTH)
    assert packed.tensors[0].shape == (BATCH, STEPS, 3)
    np.testing.assert_array_equal(packed.materialize(packed.tensors, STEPS), gray)
    both = sources.ConcatFrames(sources.DenseFrames(grating), packed)
    buffer = both.initial_buffer()
    for time in range(STEPS):
        frame, buffer = both.advance(both.tensors, tf.constant(time), buffer)
        expected = tf.concat([grating[:, time], gray[:, time]], axis=0)
        np.testing.assert_array_equal(frame, expected)
        np.testing.assert_array_equal(both.frame(both.tensors, time), expected)
    assert not both.differentiable


def test_packed_frames_stay_in_gpu_memory():
    """Words packed in a tf.function, as the training input is, are held by
    the GPU allocator (int32 words would be returned in host memory)."""
    if not tf.config.list_physical_devices("GPU"):
        pytest.skip("CUDA GPU required")
    spikes = tf.random.uniform((64, 500, 17400)) < 0.1
    before = tf.config.experimental.get_memory_info("GPU:0")["current"]
    words = tf.function(sources.pack_frames)(spikes)
    assert words.dtype == tf.uint32
    assert tf.config.experimental.get_memory_info("GPU:0")["current"] - before >= words.shape.num_elements() * 4
    np.testing.assert_array_equal(sources.PackedFrames(words, 17400).frame((words,), 499), spikes[:, 499])


def test_gradient_hook_matches_a_tape():
    dense = tf.cast(_spikes(2), tf.float32)
    parameters = tf.constant(np.random.default_rng(3).normal(size=(STEPS, WIDTH)), tf.float32)
    # Float frames of every part: a rollout's frames share one dtype.
    source = sources.ConcatFrames(
        sources.DenseFrames(dense),
        sources.DenseFrames(tf.cast(_spikes(4), tf.float32)),
        _Logistic(parameters),
    )
    cotangents = tf.constant(
        np.random.default_rng(5).normal(size=(STEPS, 3 * BATCH, WIDTH)), tf.float32)
    hooked, taped = _gradients(source, source.tensors, cotangents, STEPS - 1)
    for actual, expected in zip(hooked, taped):
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(hooked[0], taped[0])  # one write per frame: exact


def test_packed_frames_have_no_gradient():
    packed = sources.PackedFrames(sources.pack_frames(_spikes(6)), WIDTH)
    source = sources.ConcatFrames(sources.DenseFrames(tf.cast(_spikes(7), tf.float32)), packed)
    cotangents = tf.ones((STEPS, 2 * BATCH, WIDTH))
    buffer = source.gradient_buffer(source.tensors, STEPS)
    for time in range(STEPS):
        buffer = source.add_gradient(source.tensors, time, lambda: cotangents[time], buffer)
    dense_gradient, packed_gradient = source.gradients(source.tensors, buffer)
    np.testing.assert_array_equal(dense_gradient, tf.ones((BATCH, STEPS, WIDTH)))
    assert packed_gradient is None


def test_delayed_frames_put_the_buffer_first():
    """Step t reads buffered frame t, then source frame t - held; the
    gradients go back the same way (a Python int step included, as the
    adjoint passes the final buffer's frames)."""
    held = 2
    buffered = tf.constant(np.random.default_rng(8).random((BATCH, held, WIDTH)), tf.float32)
    inputs = tf.cast(_spikes(9), tf.float32)
    for source_gradients in (True, False):
        delayed = sources.DelayedFrames(buffered, sources.DenseFrames(inputs), source_gradients)
        tensors = delayed.tensors
        for time in range(STEPS + held):
            expected = buffered[:, time] if time < held else inputs[:, time - held]
            np.testing.assert_array_equal(delayed.frame(tensors, tf.constant(time)), expected)
        cotangents = tf.constant(
            np.random.default_rng(10).normal(size=(STEPS + held, BATCH, WIDTH)), tf.float32)
        buffer = delayed.gradient_buffer(tensors, STEPS + held)
        for time in range(STEPS + held):
            step = time if time >= STEPS else tf.constant(time)
            buffer = delayed.add_gradient(tensors, step, lambda: cotangents[time], buffer)
        buffered_gradient, input_gradient = delayed.gradients(tensors, buffer)
        np.testing.assert_array_equal(buffered_gradient,
                                      tf.transpose(cotangents[:held], (1, 0, 2)))
        if source_gradients:
            np.testing.assert_array_equal(input_gradient,
                                          tf.transpose(cotangents[held:], (1, 0, 2)))
        else:
            assert input_gradient is None
