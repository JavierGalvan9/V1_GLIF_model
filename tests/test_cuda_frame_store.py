"""The host frame store must hand back exactly the frames it was given.

`offload_frame` copies a GPU frame to pinned host memory on its own stream and
`take_frame` brings it back (prefetching the frames before it), so a reverse
loop over a forward loop's frames sees the same bits, whether a frame comes
back from the host or, while its copy out still runs, from the GPU. With the
TQ-pack codec, float32 frames go out raw while the link keeps up and encoded
otherwise (other dtypes always raw); encoded ones come back exactly on the
grid, rint(x * 2**16) * 2**-16 (raw blocks, those holding a non-finite value or
|x| >= 127, exactly), so within 2**-17, whether decoded from the host copy,
from the ring arena or from the packet still on the GPU. ``tq_slow`` runs the
codec on a simulated link of 0 GB/s (V1_FRAME_STORE_LINK_GBPS), where every
float32 frame is encoded.
"""

import numpy as np
import pytest
import tensorflow as tf

from v1_model_utils import cuda_frame_store

pytestmark = pytest.mark.skipif(
    not cuda_frame_store.available(), reason="CUDA frame store unavailable"
)


def _frames(length, shape, dtype, seed):
    rng = np.random.default_rng(seed)
    if dtype == tf.int32:
        values = rng.integers(-2**31, 2**31 - 1, (length, *shape), dtype=np.int64)
        return values.astype(np.int32)
    values = rng.normal(size=(length, *shape)).astype(dtype.as_numpy_dtype)
    values.reshape(length, -1)[:, :3] = [0.0, -0.0, np.inf]  # signed zero and inf survive
    return values


# TQ-pack's grid step is 2**-16: finite values come back within half.
TQ_ERROR = 2.0**-17


def _op(codec):
    """The store's codec attribute of a test codec."""
    return codec.removesuffix("_slow")


def _tq_grid(frame):
    """A frame as TQ-pack reconstructs it: rint(x * 2**16) * 2**-16, except the
    blocks of 32 consecutive values holding a non-finite value or |x| >= 127,
    which are stored raw."""
    flat = frame.reshape(-1)
    padded = np.zeros(-(-flat.size // 32) * 32, flat.dtype)
    padded[:flat.size] = flat
    with np.errstate(invalid="ignore"):
        raw = ~(np.abs(padded) < 127.0).reshape(-1, 32).all(axis=1)
        grid = np.rint(padded * 2.0**16) * 2.0**-16
    return np.where(np.repeat(raw, 32), padded, grid)[:flat.size].reshape(frame.shape)


def _assert_restored(taken, frames, codec):
    if _op(codec) == "tq" and frames.dtype == np.float32:
        # Each frame went out raw (exact) or encoded (exactly on the grid).
        for got, frame in zip(taken, frames):
            if not np.array_equal(got.view(np.uint32), frame.view(np.uint32)):
                np.testing.assert_array_equal(got, _tq_grid(frame))
                finite = np.isfinite(frame)
                assert np.abs(got[finite] - frame[finite]).max() <= TQ_ERROR
    else:
        np.testing.assert_array_equal(taken.view(np.uint8), frames.view(np.uint8))


def _stats(capfd):
    """The last V1_FRAME_STORE_STATS line printed (by a take of frame 0)."""
    line = capfd.readouterr().err.split("V1_FRAME_STORE_STATS ")[-1].splitlines()[0]
    return dict(item.split("=") for item in line.split())


def _round_trip(frames, key, prefetch, group=1, resident_from=None, codec="raw"):
    """Offload the frames in a forward while loop, then take them in a
    reverse one (as the recorded adjoint does: the take, with its prefetch,
    after the previous step's work); the frames in forward order."""
    length = frames.shape[0]
    like = tf.zeros(frames.shape[1:], tf.as_dtype(frames.dtype))

    @tf.function
    def forward():
        def step(time, token):
            return time + 1, cuda_frame_store.offload_frame(
                tf.identity(tf.constant(frames)[time]), time, key, length, resident_from, group,
                codec=_op(codec),
            )
        return tf.while_loop(lambda time, _: time < length, step,
                             (tf.constant(0), tf.zeros([0])), parallel_iterations=8)[1]

    @tf.function
    def reverse(token):
        def step(time, token, taken):
            with tf.control_dependencies([token]):
                frame = cuda_frame_store.take_frame(time, key, like, prefetch=prefetch, group=group)
            return time - 1, token, taken.write(length - 1 - time, frame)
        taken = tf.TensorArray(like.dtype, size=length)
        return tf.while_loop(lambda time, *_: time >= 0, step,
                             (tf.constant(length - 1), token, taken),
                             parallel_iterations=4)[2].stack()

    with tf.device("/GPU:0"):
        return reverse(forward()).numpy()[::-1]


@pytest.fixture(params=["raw", "tq", "tq_slow"])
def codec(request, monkeypatch):
    if request.param.endswith("_slow"):
        monkeypatch.setenv("V1_FRAME_STORE_LINK_GBPS", "0")
    return request.param


def _error_bound(codec):
    return 0.0 if _op(codec) == "raw" else TQ_ERROR


@pytest.mark.parametrize("dtype", [tf.float32, tf.float16, tf.int32])
@pytest.mark.parametrize("prefetch", [0, 3])
def test_round_trip_is_bitwise(dtype, prefetch, codec):
    """Bitwise for every dtype, except encoded float32 (on the grid, within
    2**-17; infinities and NaNs exact); float16 and int32 fall back to raw
    copies."""
    frames = _frames(9, (4, 1031), dtype, seed=7)
    if dtype == tf.float32:
        frames[:, 1, :2] = np.nan
    taken = _round_trip(frames, f"test_{dtype.name}_{prefetch}_{codec}", prefetch, codec=codec)
    _assert_restored(taken, frames, codec)


@pytest.mark.parametrize("group,resident_from", [(1, 9), (2, 7), (3, 4), (4, 0), (5, 11),
                                                 (32, 11)])
def test_groups_and_resident_tail(group, resident_from, codec):
    """Copies in groups (a partial last group flushed by the first take) and a
    resident tail of any length."""
    frames = _frames(11, (3, 517), tf.float32, seed=group)
    taken = _round_trip(frames, f"test_group_{group}_{resident_from}_{codec}", 4,
                        group=group, resident_from=resident_from, codec=codec)
    _assert_restored(taken, frames, codec)


def test_large_frames_from_host_and_in_flight(codec):
    """52 MB frames: early ones finish their copy out (host path), the last
    ones may still be copying when the reverse loop starts (GPU path)."""
    frames = _frames(6, (64, 203816), tf.float32, seed=3)
    taken = _round_trip(frames, "test_large", prefetch=2, codec=codec)
    _assert_restored(taken, frames, codec)


def _tq_distance(taken, frame):
    """0 if `taken` is `frame` (sent raw) or its TQ grid (no raw blocks here),
    else the smaller of the two largest differences."""
    grid = tf.round(frame * 2.0**16) * 2.0**-16  # half to even
    return tf.minimum(tf.reduce_max(tf.abs(taken - frame)), tf.reduce_max(tf.abs(taken - grid)))


def _generated_round_trip(key, length, shape, codec, group=4, resident_from=None, prefetch=4,
                          again=0, walk=False):
    """Offloads `length` frames generated on the GPU (no host fixture), takes
    them back in a reverse loop and, with ``again``, takes frames again - 1 .. 0
    a second time (from their host copies). Returns the largest absolute error
    (float32 frames) over both passes; for TQ, the largest distance of a frame
    from both itself and its grid (0: each frame exact or exactly on the grid).
    ``walk`` frames drift slowly, like the membrane voltage (TQ predicts well)."""
    like = tf.zeros(shape)

    def frame(time):
        noise = tf.random.stateless_normal(shape, tf.stack([91, time]))
        if not walk:
            return noise
        base = tf.random.stateless_normal(shape, [92, 0])
        return base + 0.01 * tf.cast(time, tf.float32) * base + 1e-3 * noise

    def distance(taken, time):
        if _op(codec) == "tq":
            return _tq_distance(taken, frame(time))
        return tf.reduce_max(tf.abs(taken - frame(time)))

    def reverse_loop(first, token):
        def step(time, token, error):
            with tf.control_dependencies([token]):
                taken = cuda_frame_store.take_frame(time, key, like, prefetch=prefetch, group=group)
            error = tf.maximum(error, distance(taken, time))
            # The next take waits for this step's work.
            return time - 1, tf.zeros([0]) * error, error
        return tf.while_loop(lambda time, *_: time >= 0, step,
                             (first, token, tf.constant(0.0)), parallel_iterations=1)[2]

    @tf.function
    def cycle():
        def step(time, token):
            return time + 1, cuda_frame_store.offload_frame(
                frame(time), time, key, length, resident_from, group, codec=_op(codec)
            )
        token = tf.while_loop(lambda time, _: time < length, step,
                              (tf.constant(0), tf.zeros([0])), parallel_iterations=4)[1]
        error = reverse_loop(tf.constant(length - 1), token)
        if again:
            with tf.control_dependencies([error]):
                error = tf.maximum(error, reverse_loop(tf.constant(again - 1), tf.zeros([0])))
        return error

    with tf.device("/GPU:0"):
        return float(cycle())


def test_long_loop_and_second_pass_over_the_host_copies(codec):
    """500 small frames, groups of 4, an 8-frame resident tail; copied-out
    frames can be taken twice (frames kept on the GPU leave with their first
    take). The first two groups are always admitted: the forward outlasts
    their copies by far."""
    error = _generated_round_trip(f"test_loop500_{codec}", 500, (4, 1031), codec,
                                  resident_from=492, again=8)
    assert error <= _error_bound(codec)


def test_large_frames_reuse_the_store_across_layouts(codec):
    """52 MB frames generated on the GPU: large groups and a resident tail,
    with the store reconfigured in between."""
    bound = _error_bound(codec)
    assert _generated_round_trip("test_large_reuse", 6, (64, 203816), codec) <= bound
    assert _generated_round_trip("test_large_reuse", 3, (2, 257), codec, group=1) <= bound
    assert _generated_round_trip("test_large_reuse", 6, (64, 203816), codec, group=32,
                                 resident_from=4) <= bound


def test_take_right_after_the_offload(codec):
    """A take that depends only on its offload finds the frame raw, encoded on
    the GPU or on the host, depending on timing; all are the same frame."""
    shape = (64, 203816)

    @tf.function
    def immediate():
        frame = tf.random.stateless_normal(shape, [5, 0])
        token = cuda_frame_store.offload_frame(frame, tf.constant(0), f"test_now_{codec}", 1,
                                               codec=_op(codec))
        with tf.control_dependencies([token]):
            taken = cuda_frame_store.take_frame(tf.constant(0), f"test_now_{codec}", frame)
        return tf.reduce_max(tf.abs(taken - frame))

    with tf.device("/GPU:0"):
        errors = [float(immediate()) for _ in range(3)]
    assert max(errors) <= _error_bound(codec)


def test_frames_the_copy_engine_cannot_take_stay_on_the_gpu(codec, capfd, monkeypatch):
    """A forward much faster than PCIe: copies are admitted only as earlier
    ones finish, and the frames still queued when the forward ends are kept on
    the GPU (taken in place or decoded from their packets), not copied out and
    straight back. Packets are smaller than the frames."""
    monkeypatch.setenv("V1_FRAME_STORE_STATS", "1")
    length, shape, key = 24, (64, 203816), f"test_admission_{codec}"
    like = tf.zeros(shape)

    def frame(time):
        return tf.random.stateless_uniform(shape, tf.stack([time, 7]))

    @tf.function
    def forward():
        def step(time, token):
            return time + 1, cuda_frame_store.offload_frame(frame(time), time, key, length,
                                                            codec=_op(codec))
        return tf.while_loop(lambda time, _: time < length, step,
                             (tf.constant(0), tf.zeros([0])))[1]

    @tf.function
    def reverse(token):
        def step(time, token, mismatches):
            with tf.control_dependencies([token]):
                taken = cuda_frame_store.take_frame(time, key, like, prefetch=2)
            if _op(codec) == "tq":
                differs = _tq_distance(taken, frame(time)) > 0.0
            else:
                differs = tf.reduce_any(
                    tf.bitcast(taken, tf.int32) != tf.bitcast(frame(time), tf.int32))
            return time - 1, taken[0, :0], mismatches + tf.cast(differs, tf.int32)
        return tf.while_loop(lambda time, *_: time >= 0, step,
                             (tf.constant(length - 1), token, tf.constant(0)))[2]

    with tf.device("/GPU:0"):
        assert reverse(forward()).numpy() == 0
    stats = _stats(capfd)
    copied, kept = int(stats["copied_out"]), int(stats["not_admitted"])
    assert copied >= 1 and kept >= 1 and copied + kept <= length
    assert int(stats["fetched_back"]) + int(stats["taken_on_gpu"]) == length
    assert stats["codec"] == _op(codec)
    if codec.endswith("_slow"):
        # Every frame is encoded (unless the ring was full); those copied
        # out left from their buffer (or a packet), those kept on the GPU are
        # packed and decoded there.
        full = int(stats["ring_full"])
        assert int(stats["encoded"]) + full == length
        assert int(stats["sent_encoded"]) + int(stats["sent_raw"]) == copied
        assert int(stats["sent_raw"]) <= full
        assert int(stats["kept_packed"]) + full >= kept
        assert int(stats["direct_decodes"]) >= int(stats["kept_packed"])
        assert 0 < int(stats["packet_bytes"]) < int(stats["raw_bytes"])


def _adaptive_rollout(key, length, shape, capfd):
    """One forward of `length` generated frames (groups of 4) and its reverse
    pass with the adaptive codec; the store's statistics and the largest error."""
    error = _generated_round_trip(key, length, shape, "tq")
    return _stats(capfd), error


@pytest.mark.parametrize("link,low,high", [("1e6", 0.0, 0.0), ("0", 0.9, 1.0),
                                           ("fit", 0.15, 0.85)])
def test_adaptive_codec_encodes_what_the_link_cannot_take(link, low, high, capfd, monkeypatch):
    """The share of encoded frames follows the projected link shortfall: none
    on a link far faster than the forward, all on a 0 GB/s link, and a mix
    (raw and encoded interleaved) on a link that takes ~75 % of the raw
    stream in the forward's time. The forward's frame rate and the encoded
    length come from a calibration rollout of the same, warmed store."""
    monkeypatch.setenv("V1_FRAME_STORE_STATS", "1")
    length, shape, key = 120, (64, 20000), f"test_adaptive_{link}"
    if link == "fit":
        # Allocate the store and warm the kernels before calibrating the link.
        # The first rollout's cold frame rate understates the steady workload.
        _adaptive_rollout(key, length, shape, capfd)
    first, _ = _adaptive_rollout(key, length, shape, capfd)
    if link == "fit":
        # The link carries 0.75 of the raw bytes the forward produces (the
        # projection counts on 0.9 of it).
        frame_bytes = 4 * np.prod(shape)
        link = str(0.75 / 0.9 * float(first["frames_per_s"]) * frame_bytes / 1e9)
    monkeypatch.setenv("V1_FRAME_STORE_LINK_GBPS", link)
    stats, error = _adaptive_rollout(key, length, shape, capfd)
    assert error <= TQ_ERROR
    encoded, offloaded = int(stats["encoded"]), length
    assert low * offloaded <= encoded <= high * offloaded, stats
    assert int(stats["sent_raw"]) + int(stats["sent_encoded"]) == int(stats["copied_out"])
    if encoded == 0:
        assert int(stats["packet_bytes"]) == 0 and int(stats["sent_encoded"]) == 0
    assert float(stats["frames_per_s"]) > 0


def test_measured_link_rate_is_reported(capfd, monkeypatch):
    """Without a simulated link the store measures its copies' rate."""
    monkeypatch.setenv("V1_FRAME_STORE_STATS", "1")
    stats, error = _adaptive_rollout("test_measured_link", 40, (64, 20000), capfd)
    assert error <= TQ_ERROR
    assert float(stats["link_gbps"]) > 0.5 and int(stats["copied_out"]) >= 1


def test_reused_store_and_second_reverse_pass():
    key = "test_reuse"
    first = _frames(5, (3, 257), tf.float32, seed=1)
    np.testing.assert_array_equal(_round_trip(first, key, 2), first)
    # Same key, new contents and a new layout: the store reconfigures.
    second = _frames(7, (2, 999), tf.float32, seed=2)
    np.testing.assert_array_equal(_round_trip(second, key, 2), second)
    # A copied-out frame can be taken again (from its host copy). Frame 0 is
    # always copied out: nothing is in flight when the new layout starts.
    like = tf.zeros((2, 999))
    with tf.device("/GPU:0"):
        again = cuda_frame_store.take_frame(tf.constant(0), key, like)
    np.testing.assert_array_equal(again.numpy(), second[0])
    # The codec changes on the same key (and back): the store reconfigures.
    third = _frames(7, (2, 999), tf.float32, seed=3)
    _assert_restored(_round_trip(third, key, 2, codec="tq"), third, "tq")
    np.testing.assert_array_equal(_round_trip(second, key, 2), second)


def test_keys_are_independent():
    a = _frames(3, (2, 64), tf.float32, seed=4)
    b = _frames(3, (2, 64), tf.float32, seed=5)
    with tf.device("/GPU:0"):
        for time in range(3):
            cuda_frame_store.offload_frame(tf.constant(a[time]), tf.constant(time), "test_a", 3)
            cuda_frame_store.offload_frame(tf.constant(b[time]), tf.constant(time), "test_b", 3)
        like = tf.zeros((2, 64))
        for time in reversed(range(3)):
            np.testing.assert_array_equal(
                cuda_frame_store.take_frame(tf.constant(time), "test_b", like).numpy(), b[time])
            np.testing.assert_array_equal(
                cuda_frame_store.take_frame(tf.constant(time), "test_a", like).numpy(), a[time])


def test_rejects_bad_requests():
    with tf.device("/GPU:0"):
        cuda_frame_store.offload_frame(tf.zeros((2, 8)), tf.constant(0), "test_bad", 2)
        with pytest.raises(tf.errors.InvalidArgumentError, match="outside"):
            cuda_frame_store.offload_frame(tf.zeros((2, 8)), tf.constant(2), "test_bad", 2)
        with pytest.raises(tf.errors.FailedPreconditionError, match="never offloaded"):
            cuda_frame_store.take_frame(tf.constant(1), "test_bad", tf.zeros((2, 8)))
        with pytest.raises(tf.errors.InvalidArgumentError, match="expects"):
            cuda_frame_store.take_frame(tf.constant(0), "test_bad", tf.zeros((2, 9)))


def _drifting_frames(length, shape, seed):
    """Frames drifting slowly in time, like the membrane voltage, with raw
    blocks (NaN, infinities, |x| >= 127) and values half a grid step apart
    (ties round to even)."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=shape)
    steps = rng.normal(scale=1e-3, size=(length, *shape))
    frames = (base + np.cumsum(steps, axis=0)).astype(np.float32)
    flat = frames.reshape(length, -1)
    flat[:, :3] = [0.0, -0.0, np.inf]
    flat[1::3, 40] = np.nan
    flat[::2, 100] = 127.0
    flat[:, 170] = -300.5
    flat[:, 200] = 126.99999
    flat[:, 230:234] = np.array([1, 3, -1, -5]) * 2.0**-17
    return frames


def test_tq_frames_come_back_exactly_on_the_grid(capfd, monkeypatch):
    """Every offloaded frame encoded (0 GB/s link): each comes back as
    rint(x * 2**16) * 2**-16, raw blocks exactly, the error within 2**-17;
    the two resident frames (the chain's anchors) exactly. Frames of 20,665
    values span several tiles and a partial block."""
    monkeypatch.setenv("V1_FRAME_STORE_STATS", "1")
    monkeypatch.setenv("V1_FRAME_STORE_LINK_GBPS", "0")
    length, resident_from = 12, 10
    frames = _drifting_frames(length, (5, 4133), seed=2)
    taken = _round_trip(frames, "test_tq_grid", 3, group=2, resident_from=resident_from,
                        codec="tq")
    for step, (got, frame) in enumerate(zip(taken, frames)):
        np.testing.assert_array_equal(got, frame if step >= resident_from else _tq_grid(frame))
        finite = np.isfinite(frame)
        assert np.abs(got[finite] - frame[finite]).max() <= TQ_ERROR
    stats = _stats(capfd)
    assert stats["codec"] == "tq" and int(stats["encoded"]) == resident_from
    assert int(stats["ring_full"]) == 0
    assert 0 < int(stats["packet_bytes"]) < int(stats["raw_bytes"])


def test_tq_retake_rebuilds_the_chain(capfd, monkeypatch):
    """After a reverse pass, frame 0 taken alone again: its references are
    gone, so the store rebuilds them up the chain from the host copies to the
    anchors (the resident frames it kept)."""
    monkeypatch.setenv("V1_FRAME_STORE_STATS", "1")
    monkeypatch.setenv("V1_FRAME_STORE_LINK_GBPS", "0")
    frames = _drifting_frames(9, (3, 517), seed=4)
    key = "test_tq_retake"
    taken = _round_trip(frames, key, 2, resident_from=7, codec="tq")
    _assert_restored(taken, frames, "tq")
    with tf.device("/GPU:0"):
        again = cuda_frame_store.take_frame(tf.constant(0), key, tf.zeros((3, 517))).numpy()
    np.testing.assert_array_equal(again, _tq_grid(frames[0]))
    assert int(_stats(capfd)["chain_walks"]) >= 1


def test_tq_chain_across_raw_encoded_host_and_gpu_frames(capfd, monkeypatch):
    """52 MB drifting frames produced faster than PCIe, every frame chosen for
    encoding into a ring of two frames: frames go out encoded or (ring full)
    raw, others stay on the GPU packed or raw, and the reverse chain crosses
    all of them; a second pass takes frames 7 .. 0 again from the host."""
    monkeypatch.setenv("V1_FRAME_STORE_STATS", "1")
    monkeypatch.setenv("V1_FRAME_STORE_LINK_GBPS", "0")
    monkeypatch.setenv("V1_FRAME_STORE_RING_FRAMES", "2")
    error = _generated_round_trip("test_tq_chain", 24, (64, 203816), "tq",
                                  resident_from=22, again=8, walk=True)
    assert error == 0.0
    stats = _stats(capfd)
    assert int(stats["encoded"]) >= 1 and int(stats["ring_full"]) >= 1
    # A frame the reverse takes (or the forward's end keeps) before its encode
    # finished stays raw and is in neither count.
    assert int(stats["encoded"]) + int(stats["ring_full"]) <= 22
    assert float(stats["ring_peak_mb"]) <= float(stats["ring_mb"])


def test_tq_full_ring_keeps_frames_raw(capfd, monkeypatch):
    """A one-frame ring on a 0 GB/s link with the host far ahead of the GPU:
    frames the ring cannot take stay raw (counted) instead of waiting; all
    come back exact or on the grid."""
    monkeypatch.setenv("V1_FRAME_STORE_STATS", "1")
    monkeypatch.setenv("V1_FRAME_STORE_LINK_GBPS", "0")
    monkeypatch.setenv("V1_FRAME_STORE_RING_FRAMES", "1")
    error = _generated_round_trip("test_tq_full_ring", 16, (64, 203816), "tq", group=32)
    assert error == 0.0
    stats = _stats(capfd)
    assert int(stats["ring_full"]) >= 1 and int(stats["encoded"]) >= 1
    assert int(stats["encoded"]) + int(stats["ring_full"]) == 16
