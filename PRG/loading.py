"""Read PRG gzip/shuffle chunks with parallel decompression and serial HDF5 I/O.

HDF5 normally decompresses a dataset read serially. Reading the stored chunks
on the caller thread lets worker threads decompress and copy independent tiles.
Unsupported layouts keep h5py's ordinary read path; no disk cache is created.
"""

from concurrent.futures import ThreadPoolExecutor
from itertools import batched
from operator import index
import zlib

import h5py
import numpy as np


def read_window(dataset, start, stop, n_columns, workers=4):
    """Return ``dataset[start:stop, :n_columns]`` with its original dtype.

    The fast path supports numeric chunks filtered with gzip, optionally
    preceded by shuffle. Only decompression and writes to disjoint output tiles
    run concurrently. At most ``2 * workers`` compressed chunks are queued, so
    temporary memory stays bounded independently of the requested window size.
    Set ``workers=1`` to use the ordinary HDF5 reader.
    Writable files use that reader because newly written chunk masks may be stale.

    Raw-chunk filter masks indicate filters which were skipped when writing:
    https://api.h5py.org/h5d.html#h5py.h5d.DatasetID.read_direct_chunk
    """
    start, stop, n_columns, workers = map(index, (start, stop, n_columns, workers))
    if dataset.ndim != 2:
        raise ValueError("Expected a two-dimensional time-by-neuron dataset.")
    if not 0 <= start <= stop <= dataset.shape[0]:
        raise ValueError("The requested time window is outside the dataset.")
    if not 0 <= n_columns <= dataset.shape[1] or workers < 1:
        raise ValueError("Column count must be within the dataset and workers must be positive.")
    shape = (stop - start, n_columns)
    if not all(shape):
        return np.empty(shape, dtype=dataset.dtype)
    properties = dataset.id.get_create_plist()
    filters = tuple(properties.get_filter(i)[0] for i in range(properties.get_nfilters()))
    shuffle, deflate = h5py.h5z.FILTER_SHUFFLE, h5py.h5z.FILTER_DEFLATE
    if (workers == 1 or dataset.file.mode != "r" or dataset.chunks is None
            or dataset.dtype.kind not in "biufc"
            or filters not in ((deflate,), (shuffle, deflate))):
        return dataset[start:stop, :n_columns]

    dtype, chunks = dataset.dtype, dataset.chunks
    chunk_rows, chunk_columns = chunks
    output = np.empty(shape, dtype=dtype)

    def decode(task):
        row, column, mask, raw = task
        for position in range(len(filters) - 1, -1, -1):
            if mask & (1 << position):
                continue
            if filters[position] == deflate:
                raw = zlib.decompress(raw)
            elif dtype.itemsize > 1:
                raw = np.frombuffer(raw, np.uint8).reshape(dtype.itemsize, -1).T.copy()
        values = np.frombuffer(raw, dtype=dtype).reshape(chunks)
        first, last = max(start, row), min(stop, row + chunk_rows)
        last_column = min(n_columns, column + chunk_columns)
        output[first - start:last - start, column:last_column] = (
            values[first - row:last - row, :last_column - column]
        )

    offsets = (
        (row, column)
        for row in range(start // chunk_rows * chunk_rows, stop, chunk_rows)
        for column in range(0, n_columns, chunk_columns)
    )
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for batch in batched(offsets, 2 * workers):
            tasks = []
            for row, column in batch:
                try:
                    mask, raw = dataset.id.read_direct_chunk((row, column))
                except (OSError, RuntimeError):
                    # Unallocated chunks and unsupported raw reads retain HDF5's
                    # fill-value/error handling instead of inventing decoded data.
                    return dataset[start:stop, :n_columns]
                tasks.append((row, column, mask, raw))
            # All HDF5 calls stay above, on this thread. Workers see bytes only.
            list(pool.map(decode, tasks))
    return output
