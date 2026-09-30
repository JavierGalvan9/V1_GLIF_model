import sys

import pytest

import parallel_training_testing as wrapper


@pytest.mark.parametrize("restored", [False, True])
def test_initial_evaluation_submitted_for_fresh_and_restored_runs(
    restored, tmp_path, monkeypatch, capsys
):
    arguments = [
        "parallel_training_testing.py", "--neurons", "935980", "--n_runs", "1",
        "--results_dir", str(tmp_path), "--print_only",
    ]
    if restored:
        checkpoint_dir = tmp_path / "source" / "Best_model"
        checkpoint_dir.mkdir(parents=True)
        arguments.extend(["--restore_from", str(checkpoint_dir)])
    monkeypatch.setattr(sys, "argv", arguments)

    wrapper.main()

    commands = [line for line in capsys.readouterr().out.splitlines() if line.startswith("run ")]
    initial = [command for command in commands if "initial_test" in command]
    assert len(commands) == 2
    assert len(initial) == 1
    assert "--dtype float32" in initial[0]
    assert ("--evaluate_untrained" in initial[0]) is not restored
