#!/usr/bin/env bash
# Submit from Nuredduna: bash PRG/long_gray_simulation.sh --restore_from /path/to/Best_model
set -euo pipefail

if [[ $# -eq 0 || ${1-} == --help || ${1-} == -h ]]; then
    cat <<'USAGE'
Usage: bash PRG/long_gray_simulation.sh [--restore_from /path/to/Best_model] [flags...]

Submit one RTX 3090 GPU job on Nuredduna (8 CPUs, 80 GB RAM, 1 hour).
Defaults: hard reset, 5 s warm-up, 60 s recording, 203816 neurons, 400 um core.
Extra flags go to long_gray_simulation.py, including --seed, --run_session,
--duration_ms, --warmup_ms, --seq_len, --plot_core_radius and --output_file.
Model settings must match training; no config.json is loaded automatically.
Without --restore_from, the simulation uses the untrained model.
USAGE
    exit 0
fi

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd -- "$repo_root"
# Conda's compiler hooks may read unset backup variables during reactivation.
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate neuro_tf221
set -u
if ! command -v run >/dev/null 2>&1; then
    echo 'Nuredduna run helper is unavailable; no job was submitted.' >&2
    exit 1
fi

session_name="prg_$(date -u +%Y%m%d_%H%M%S)"
output_file="$repo_root/PRG/results/$session_name/simulation_data.hdf5"
mkdir -p PRG/Out PRG/Error
# Later flags override this generated output path if --output_file is supplied.
command_args=(python3 -u PRG/long_gray_simulation.py --output_file "$output_file" "$@")
# run accepts one shell command string; quote each argument before handing it off.
printf -v job_command '%q ' "${command_args[@]}"
run -g 1 -G rtx3090 -c 8 -m 80 -t 1:00 \
    -o "PRG/Out/$session_name.out" -e "PRG/Error/$session_name.err" \
    -j "$session_name" "$job_command"
