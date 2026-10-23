import shlex
import sys

import pytest

import parallel_training_testing as wrapper


@pytest.mark.parametrize("restored", [False, True])
def test_single_run_submits_training_for_fresh_and_restored_runs(
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
    assert len(commands) == 1
    assert "python multi_training.py " in commands[0]
    assert "initial_test" not in commands[0]
    training_args = shlex.split(shlex.split(commands[0])[-1])
    assert training_args[training_args.index("--restore_from") + 1] == (
        str(checkpoint_dir) if restored else ""
    )


@pytest.mark.parametrize("mode", ["none", "raw"])
def test_history_offload_reaches_training_only(mode, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", [
        "parallel_training_testing.py", "--neurons", "66652", "--n_runs", "2",
        "--results_dir", str(tmp_path), "--history_offload", mode, "--print_only",
    ])
    wrapper.main()
    commands = [line for line in capsys.readouterr().out.splitlines() if line.startswith("run ")]
    training = [command for command in commands if "python multi_training.py " in command]
    evaluation = [command for command in commands if "python osi_dsi_estimator.py " in command]
    assert len(training) == len(evaluation) == 2
    assert all(f"--history_offload {mode}" in command for command in training)
    assert all("--history_offload" not in command for command in evaluation)


def test_history_offload_defaults_to_none():
    assert wrapper.parser.parse_args([]).history_offload == "none"
