import pytest

from lgn_model import lgn
from v1_model_utils import training_utils


def test_parse_delays_rejects_trimming_entire_sequence():
    with pytest.raises(ValueError, match="leave at least one timestep"):
        training_utils.parse_delays("60,40", seq_len=100)


@pytest.mark.parametrize("n_input", [0, -1, 17401])
def test_validate_n_input_rejects_values_outside_lgn_population(n_input):
    with pytest.raises(ValueError, match="between 1 and 17400"):
        training_utils.validate_n_input(n_input)


@pytest.mark.parametrize("n_input", [1, 17400])
def test_validate_n_input_accepts_population_bounds(n_input):
    assert training_utils.validate_n_input(n_input) == n_input


@pytest.mark.parametrize("n_input", [0, -1, 4])
def test_lgn_rejects_input_counts_outside_loaded_population(
    monkeypatch, tmp_path, n_input
):
    data_file = tmp_path / "tf_data" / "lgn_full_col_cells_120x80.csv"
    data_file.parent.mkdir()
    data_file.touch()
    monkeypatch.setattr(
        lgn.pd,
        "read_csv",
        lambda *args, **kwargs: lgn.pd.DataFrame({"model_id": ["ON_a"] * 3}),
    )

    with pytest.raises(ValueError, match="between 1 and 3"):
        lgn.LGN(data_dir=str(tmp_path), n_input=n_input)
