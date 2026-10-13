import pytest

from multi_training import (
    resolve_stimulus_batch_sizes,
    split_combined_outputs,
)


def test_resolve_stimulus_batch_sizes_accepts_unequal_components():
    assert resolve_stimulus_batch_sizes(4, 3, 1) == (3, 1)


@pytest.mark.parametrize(
    "batch_size, grating_batch_size, gray_batch_size",
    [(4, 0, 4), (4, 4, 0), (4, 3, 2)],
)
def test_resolve_stimulus_batch_sizes_rejects_invalid_allocations(
    batch_size, grating_batch_size, gray_batch_size
):
    with pytest.raises(ValueError):
        resolve_stimulus_batch_sizes(
            batch_size, grating_batch_size, gray_batch_size
        )


def test_split_combined_outputs_uses_grating_boundary():
    gratings, gray = split_combined_outputs(list(range(4)), 3)

    assert gratings == [0, 1, 2]
    assert gray == [3]
