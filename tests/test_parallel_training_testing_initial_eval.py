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


def test_talaia_jobs_are_sbatch_commands_with_dependencies(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", [
        "parallel_training_testing.py", "--neurons", "66652", "--n_runs", "2", "--cluster", "talaia",
        "--n_gpus", "2", "--memory_gb", "256", "--results_dir", str(tmp_path), "--print_only",
    ])
    wrapper.main()
    commands = [
        shlex.split(line) for line in capsys.readouterr().out.splitlines() if line.startswith("sbatch ")
    ]
    training = [command for command in commands if "python multi_training.py " in command[-1]]
    evaluation = [command for command in commands if "python osi_dsi_estimator.py " in command[-1]]
    assert len(training) == len(evaluation) == 2
    for command in commands:
        assert command[-2] == "--wrap"
        assert command[command.index("-A") + 1] == "uib_focosineu_acc"
        assert command[command.index("--qos") + 1] == "normal"
        assert "-p" not in command and not any(arg.startswith("--partition") for arg in command)
        assert "OMP_NUM_THREADS=" in next(arg for arg in command if arg.startswith("--export=ALL,"))
        assert "--cluster" not in command[-1]
    assert "--gres=gpu:b200:2" in training[0]
    assert "--mem=256G" in training[0] and "--mem=200G" in evaluation[0]
    assert "--memory_gb" not in training[0][-1]
    assert training[0][training[0].index("-t") + 1] == "48:00:00"
    assert not any(arg.startswith("--dependency") for arg in training[0])
    assert any(arg.startswith("--dependency=afterok:<") for arg in training[1])
    assert "--gres=gpu:1" in evaluation[0]
    assert any(arg.startswith("--dependency=afterok:<") for arg in evaluation[0])


@pytest.mark.parametrize("arguments", [
    ["--cluster", "talaia", "--gpu_type", "rtxpro6000"],
    ["--cluster", "talaia", "--gpu_type", "h200", "--n_gpus", "4"],
])
def test_talaia_rejects_unavailable_gpu_requests(arguments, tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "parallel_training_testing.py", "--neurons", "1000", "--results_dir", str(tmp_path),
        "--print_only", *arguments,
    ])
    with pytest.raises(SystemExit):
        wrapper.main()


def test_nuredduna_defaults_to_rtxpro6000(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", [
        "parallel_training_testing.py", "--neurons", "1000", "--n_runs", "1",
        "--results_dir", str(tmp_path), "--print_only",
    ])
    wrapper.main()
    command = next(line for line in capsys.readouterr().out.splitlines() if line.startswith("run "))
    assert shlex.split(command)[:7] == ["run", "-g", "1", "-G", "rtxpro6000", "-c", "8"]
