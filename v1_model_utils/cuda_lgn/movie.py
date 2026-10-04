"""CUDA LGN response to arbitrary movies: spatial filter, temporal filter, rates.

See lgn_movie_ops.cc for the math. The constants are derived here once, in
float64 from the float32 LGN constants, and stored as float32: the separable
filter taps, each subunit's four bilinear corners with the corner weight, edge
normalization and amplitude folded in, and the temporal kernels by lag, in
blocks of 4 lags ([lags / 4, columns, 4]), with the range of nonzero lags of
every 32-column tile.
"""

import numpy as np
from scipy.signal import correlate2d
import tensorflow as tf

from v1_model_utils.tf_utils import replica_local_constant
from .wrapper import MOVIE, _bilinear, _load_ops, _separable_taps


TILE = 32  # subunit columns per lag range (one warp of the temporal kernel)
LAG_PADDING = 32  # trailing zero lags, so the op's lag blocks never check bounds


def _lag_major(kernels):
    """[columns, lags]: lag m weighs the frame m steps back (lag 0 the current frame),
    zero-padded to a multiple of LAG_PADDING lags with at least LAG_PADDING zeros."""
    lagged = np.asarray(kernels, np.float32)[::-1]
    lags = (lagged.shape[0] // LAG_PADDING + 2) * LAG_PADDING
    return np.ascontiguousarray(np.pad(lagged, ((0, lags - lagged.shape[0]), (0, 0))).T)


def _time_blocked(kernels):
    """[lags / 4, columns, 4] of [columns, lags]: a warp's float4 loads are contiguous."""
    columns, lags = kernels.shape
    return np.ascontiguousarray(kernels.reshape(columns, lags // 4, 4).transpose(1, 0, 2))


def _lag_ranges(kernels):
    """[tiles, 2] first and last nonzero lag of each TILE-column tile of [columns, lags]
    kernels ([LAG_PADDING, 0], empty, when all are zero)."""
    columns, lags = kernels.shape
    tiles = -(-columns // TILE)
    nonzero = np.zeros((tiles * TILE, lags), bool)
    nonzero[:columns] = kernels != 0
    nonzero = nonzero.reshape(tiles, TILE, lags).any(axis=1)
    first, last = nonzero.argmax(axis=1), lags - 1 - nonzero[:, ::-1].argmax(axis=1)
    return np.where(nonzero.any(axis=1)[:, None], np.stack((first, last), axis=1), [LAG_PADDING, 0])


class MovieLGNKernel:
    """The CUDA movie ops and their constants for one LGN and edge-normalization choice.

    Constants are replica-local (tf_utils.replica_local_constant), so under a
    distribution strategy every replica reads its own GPU's copy. `chunk`
    samples are sampled per op call: 1 (sequential) keeps one sample's movie
    and uniforms alive at a time, for about 3 ms more per 32 x 500 batch than 4.
    """

    def __init__(self, host, *, rows, cols, bmtk_compat=True, chunk=1):
        if not 1 <= chunk <= 16:
            raise ValueError(f"chunk must be in [1, 16], got {chunk}")
        self._ops = _load_ops(MOVIE)
        f64 = lambda value: np.asarray(value, np.float32).astype(np.float64)  # noqa: E731
        filters = [f64(kernel) for kernel in host["gaussian_filters"]]
        vertical, horizontal = _separable_taps(filters)
        composite = np.asarray(host["is_composite"], bool)
        units = composite.size
        composite_ids = np.flatnonzero(composite)
        slot = np.full(units, -1, np.int64)
        slot[composite_ids] = np.arange(composite_ids.size)
        amplitude, non_dom_amplitude = f64(host["amplitude"]), f64(host["non_dom_amplitude"])
        x, y = f64(host["x"]), f64(host["y"])
        non_dom_x, non_dom_y = f64(host["non_dominant_x"]), f64(host["non_dominant_y"])

        samples, weights, offsets = [], [], [0]
        for kernel, indices in zip(filters, host["spatial_range_indices"]):
            # bmtk_compat: divide by the filtered all-ones frame ('SAME', zero padding).
            reciprocal = (1.0 / correlate2d(np.ones((rows, cols)), kernel, mode="same")
                          if bmtk_compat else np.ones((rows, cols)))
            indices = np.asarray(indices, np.int64)
            selected = indices[composite[indices]]
            for column, px, py, scale in (
                    (indices, x[indices], y[indices], amplitude[indices]),
                    (units + slot[selected], non_dom_x[selected], non_dom_y[selected],
                     non_dom_amplitude[selected])):
                corners, weight = _bilinear(px, py, reciprocal, scale)
                x0, x1, y0, y1 = corners.T
                samples.append((y0 * cols + x0) | (x1 - x0) << 30 | (y1 - y0) << 31 | column << 32)
                weights.append(weight)
            offsets.append(offsets[-1] + indices.size + selected.size)
        if offsets[-1] != units + composite_ids.size:
            raise ValueError("Every LGN unit needs exactly one spatial filter bin")

        kernels = _lag_major(host["dom_temporal_kernels"])
        composite_kernels = _lag_major(np.asarray(host["non_dom_temporal_kernels"])[:, composite_ids])
        spontaneous = np.asarray(host["spontaneous_firing_rates"], np.float32)
        self.units, self.chunk = units, chunk
        self._attrs = dict(rows=rows, cols=cols)
        self._constants = [replica_local_constant(value, dtype) for value, dtype in (
            (np.stack((vertical, horizontal)), tf.float32),
            ([(f.shape[0] - 1) // 2 for f in filters], tf.int64),
            (offsets, tf.int64),
            (np.concatenate(samples), tf.int64),
            (np.concatenate(weights), tf.float32),
            (_time_blocked(kernels), tf.float32), (_lag_ranges(kernels), tf.int64),
            (spontaneous, tf.float32), (slot, tf.int64),
            (_time_blocked(composite_kernels), tf.float32),
            (_lag_ranges(composite_kernels), tf.int64),
            (spontaneous[composite_ids], tf.float32))]

    @staticmethod
    def _movie(movie):
        """The movie [batch, time, rows, cols] in float16 or float32 (others become float32)."""
        movie = tf.convert_to_tensor(movie)
        return movie if movie.dtype in (tf.float16, tf.float32) else tf.cast(movie, tf.float32)

    def response(self, movie, output="rates"):
        """Float32 rates in Hz (output='rates') or 1 ms spike probabilities
        (output='probabilities'), [batch, time, units], of a movie [batch, time, rows, cols]."""
        return self._ops.lgn_movie_response(self._movie(movie), *self._constants, output=output,
                                            **self._attrs)

    def spikes(self, movie, spike_seeds, dtype=tf.float32, batch=None):
        """Spikes [batch, time, units]: TensorFlow's stateless uniforms of each sample's
        seed [batch, 2], in `dtype`, below the probabilities, `chunk` samples at a time.

        `movie` is the batch's movie [batch, time, rows, cols], or a function
        movie(begin, end) giving the movie of samples [begin, end) of a `batch`
        of samples: each chunk's movie is then built only once the previous
        chunk is sampled, so at most one chunk of movie is ever alive.

        float16 uniforms take only 1024 values, so P(u < p) = ceil(1024 p) / 1024
        rather than p; use float32.
        """
        if callable(movie):
            if batch is None:
                raise ValueError("a movie function needs the batch size")
            chunk_movie = lambda begin, end: self._movie(movie(begin, end))  # noqa: E731
            steps = None
        else:
            movie = self._movie(movie)
            batch, steps = movie.shape[0], movie.shape[1]
            chunk_movie = None

        def uniform(seed, steps):
            return tf.random.stateless_uniform((steps, self.units), seed=seed, dtype=dtype)

        def sample(spikes, movie, uniforms, offset, whole_batch=0):
            return self._ops.lgn_movie_spikes(spikes, movie, *self._constants, uniforms,
                                              offset=offset, batch=whole_batch, **self._attrs)

        if chunk_movie is not None:
            spikes = tf.zeros((0,), tf.bool)
            for begin in range(0, batch, self.chunk):
                end = min(begin + self.chunk, batch)
                with tf.control_dependencies([spikes] if begin else []):
                    part = chunk_movie(begin, end)
                    uniforms = [uniform(spike_seeds[b], part.shape[1]) for b in range(begin, end)]
                spikes = sample(spikes, part, uniforms, begin, batch)
            return spikes
        shape = (steps, self.units)
        if batch is None or steps is None:  # one chunk when the shape is only known at run time
            steps = tf.shape(movie)[1]
            uniforms = tf.map_fn(lambda seed: uniform(seed, steps), spike_seeds,
                                 fn_output_signature=dtype)
            return sample(tf.zeros((0,), tf.bool), movie, [uniforms], 0)
        if batch == 0:
            return tf.zeros((0,) + shape, tf.bool)
        # As GratingLGNKernel.spikes: each chunk's op writes into the previous
        # chunk's output in place, and a chunk's uniforms are drawn only once the
        # previous chunk is sampled, so only one chunk of uniforms is ever alive.
        spikes = tf.zeros((0,), tf.bool)
        for begin in range(0, batch, self.chunk):
            with tf.control_dependencies([spikes] if begin else []):
                uniforms = [uniform(spike_seeds[b], steps)
                            for b in range(begin, min(begin + self.chunk, batch))]
            spikes = sample(spikes, movie, uniforms, begin)
        return spikes
