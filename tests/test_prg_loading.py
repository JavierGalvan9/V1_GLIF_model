"""Compare the parallel PRG reader against HDF5's own filter pipeline."""

import zlib

import h5py
import numpy as np
import pytest

from PRG.loading import read_window


@pytest.mark.parametrize("dtype", ["u1", "<f2", ">f2", "<f4"])
@pytest.mark.parametrize("shuffle", [False, True])
def test_parallel_windows_match_hdf5_for_partial_chunks(tmp_path, dtype, shuffle):
    values = (np.arange(13 * 9).reshape(13, 9) / 8).astype(dtype)
    with h5py.File(tmp_path / "data.h5", "w") as f:
        f.create_dataset(
            "values", data=values, chunks=(4, 5), compression="gzip", shuffle=shuffle,
        )
    with h5py.File(tmp_path / "data.h5", "r") as f:
        data = f["values"]
        for start, stop, columns in [(0, 13, 9), (3, 12, 7), (12, 13, 9), (2, 2, 5), (0, 13, 0)]:
            expected = data[start:stop, :columns]
            actual = read_window(data, start, stop, columns, workers=2)
            assert actual.dtype == expected.dtype
            assert actual.shape == expected.shape
            assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("filter_mask", [1, 2, 3])
def test_skipped_chunk_filters_match_hdf5(tmp_path, filter_mask):
    values = np.arange(20, dtype=np.float16).reshape(4, 5)
    with h5py.File(tmp_path / "masked.h5", "w") as f:
        data = f.create_dataset(
            "values", shape=values.shape, dtype=values.dtype,
            chunks=values.shape, compression="gzip", shuffle=True,
        )
        raw = values.tobytes()
        if not filter_mask & 1:  # HDF5's shuffle filter remains enabled.
            raw = values.view(np.uint8).reshape(-1, values.dtype.itemsize).T.copy().tobytes()
        if not filter_mask & 2:  # HDF5's deflate filter remains enabled.
            raw = zlib.compress(raw)
        data.id.write_direct_chunk((0, 0), raw, filter_mask=filter_mask)
    # Reopen so HDF5 refreshes the direct-write filter-mask metadata.
    with h5py.File(tmp_path / "masked.h5", "r") as f:
        data = f["values"]
        expected = data[:]
        np.testing.assert_array_equal(expected, values)
        assert read_window(data, 1, 4, 4, workers=2).tobytes() == expected[1:4, :4].tobytes()


@pytest.mark.parametrize("options", [
    {}, {"chunks": (4, 5)},
    {"chunks": (4, 5), "compression": "gzip", "fletcher32": True},
])
def test_other_storage_layouts_and_single_worker_use_native_reader(tmp_path, options):
    values = np.arange(13 * 9, dtype=np.float32).reshape(13, 9)
    with h5py.File(tmp_path / "fallback.h5", "w") as f:
        f.create_dataset("values", data=values, **options)
    with h5py.File(tmp_path / "fallback.h5", "r") as f:
        data = f["values"]
        for workers in [1, 2]:
            np.testing.assert_array_equal(read_window(data, 3, 12, 7, workers), values[3:12, :7])


def test_unwritten_chunks_preserve_hdf5_fill_values(tmp_path):
    with h5py.File(tmp_path / "partial.h5", "w") as f:
        data = f.create_dataset(
            "values", shape=(13, 9), dtype=np.float16, chunks=(4, 5),
            compression="gzip", shuffle=True, fillvalue=-0.5,
        )
        data[:4, :5] = 1
    with h5py.File(tmp_path / "partial.h5", "r") as f:
        data = f["values"]
        assert read_window(data, 0, 13, 9, workers=2).tobytes() == data[:].tobytes()


@pytest.mark.parametrize("arguments", [(-1, 2, 2, 2), (0, 14, 2, 2), (3, 2, 2, 2), (0, 2, 10, 2), (0, 2, 2, 0)])
def test_invalid_windows_are_rejected(tmp_path, arguments):
    with h5py.File(tmp_path / "invalid.h5", "w") as f:
        data = f.create_dataset("values", shape=(13, 9), dtype=np.uint8)
        with pytest.raises(ValueError):
            read_window(data, *arguments)
