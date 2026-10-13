"""Every command a cluster wrapper prints must start the script it targets.

The wrappers serialize their own argparse namespace into absl command lines,
so a flag the target script does not define, an unquoted empty value that
swallows the next flag, or an inconsistent batch split only surfaces once the
scheduler runs the job. Each printed command is parsed here by the target
script's real flag definitions, in a subprocess because absl flags are global.
"""

import os
import shlex
import subprocess
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Replaces absl.app.run so that the script's __main__ block only defines and
# parses its flags, then checks the training launch plan.
PARSE_ONLY = """
import runpy, sys, absl.app, absl.flags
from v1_model_utils import training_orchestration
script, args = sys.argv[1], sys.argv[2:]
def parse_only(main, argv=None):
    absl.flags.FLAGS([script] + args)
    if script == "multi_training.py":
        training_orchestration.plan_training_launch(args, compute_capabilities=("89",))
    print("restore_from=" + repr(absl.flags.FLAGS.restore_from))
    raise SystemExit(0)
absl.app.run = parse_only
sys.argv = [script]
runpy.run_path(script, run_name="__main__")
"""


def _python_commands(wrapper, arguments, tmp_path, monkeypatch, capsys):
    module = __import__(wrapper)
    monkeypatch.setattr(sys, "argv", [
        f"{wrapper}.py", "--neurons", "1000", "--n_runs", "2",
        "--results_dir", str(tmp_path), "--print_only", *arguments,
    ])
    module.main()
    commands = []
    for line in capsys.readouterr().out.splitlines():
        if line.startswith("run "):
            commands.append(shlex.split(line)[-1])
        elif line.startswith("python "):
            commands.append(line)
    # One command per target script and job kind is enough to cover every flag.
    unique = {}
    for command in commands:
        tokens = [token for token in shlex.split(command) if token != "-u"]
        restoring = "--evaluate_untrained" in tokens
        unique.setdefault((tokens[1], restoring), tokens[1:])
    return list(unique.values())


@pytest.mark.parametrize("wrapper", ["parallel_training_testing", "parallel_training_testing_allen"])
@pytest.mark.parametrize("restore", [False, True])
def test_printed_commands_parse_in_their_target_scripts(
    wrapper, restore, tmp_path, monkeypatch, capsys
):
    arguments = []
    if restore:
        checkpoint = tmp_path / "source" / "Best_model"
        checkpoint.mkdir(parents=True)
        arguments = ["--restore_from", str(checkpoint)]
    commands = _python_commands(wrapper, arguments, tmp_path, monkeypatch, capsys)
    assert {command[0] for command in commands} == {
        "multi_training.py", "osi_dsi_estimator.py"
    }
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=REPO_ROOT)
    for script, *args in commands:
        result = subprocess.run(
            [sys.executable, "-c", PARSE_ONLY, script, *args],
            cwd=REPO_ROOT, env=environment, capture_output=True, text=True,
        )
        assert result.returncode == 0, f"{script}: {result.stderr[-2000:]}"
        # An unquoted empty value would have consumed the following flag.
        restored = result.stdout.strip().splitlines()[-1]
        assert not restored.startswith("restore_from='--"), restored
