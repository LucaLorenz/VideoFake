#!/usr/bin/env bash
# Fresh Runpod GPU Pod -> ready notebook. No Docker build or old Pod needed.
set -Eeuo pipefail
trap 'printf "\n\033[1;31mINSTALLAZIONE FALLITA alla riga %s. Non avviare il video.\033[0m\n" "$LINENO" >&2' ERR

project_dir="${VIDEOFAKE_PROJECT_DIR:-/workspace/VideoFake-tools}"
runtime_dir="${VIDEOFAKE_RUNTIME_DIR:-/opt/videofake}"
repository="https://github.com/LucaLorenz/VideoFake.git"

command -v python3 >/dev/null
command -v nvidia-smi >/dev/null
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
gpu_capability="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sed -n '1p')"
gpu_major="${gpu_capability%%.*}"
if [[ "$gpu_major" =~ ^[0-9]+$ ]] && (( gpu_major >= 10 )); then
    printf 'Questa GPU richiede uno stack più recente. Usa A40/A5000/A6000/RTX 3090/RTX 4090 con questo ambiente.\n' >&2
    exit 1
fi

missing=()
for executable in git ffmpeg ffprobe gcc g++ make; do
    command -v "$executable" >/dev/null || missing+=("$executable")
done
if (( ${#missing[@]} )); then
    if (( EUID != 0 )) || ! command -v apt-get >/dev/null; then
        printf 'Usa il template ufficiale Runpod PyTorch con accesso root; mancano: %s\n' "${missing[*]}" >&2
        exit 1
    fi
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    apt-get install -y -qq git ffmpeg build-essential ca-certificates
fi

mkdir -p "$(dirname "$project_dir")" "$runtime_dir" /workspace/input /workspace/output
if [[ ! -d "$project_dir/.git" ]]; then
    if [[ -e "$project_dir" ]]; then
        printf 'La cartella %s esiste già: nessun file sovrascritto.\n' "$project_dir" >&2
        exit 1
    fi
    git clone --depth 1 "$repository" "$project_dir"
fi

# Reuse the same completed installation on reruns. Never reset an existing
# checkout or overwrite an edited notebook; updates are a separate git pull.
export VIDEOFAKE_HOME="$runtime_dir"
export VIDEOFAKE_INSTALL_PROFILE=lean
export UV_PYTHON_INSTALL_DIR="$runtime_dir/python"
export UV_CACHE_DIR="$runtime_dir/uv-cache"
export PIP_NO_CACHE_DIR=1
export PYTHONUNBUFFERED=1

cd "$project_dir"
python3 - "$runtime_dir" <<'PY'
import json, os, sys
from pathlib import Path
path = Path('.runtime.json')
new = {'runtime': sys.argv[1], 'profile': 'lean'}
if path.is_file() and json.loads(path.read_text()) != new:
    raise SystemExit('Configurazione esistente diversa: nessuna modifica. Usa una cartella progetto nuova.')
path.write_text(json.dumps(new, indent=2))
import colab_runner as runner
marker = runner.ROOT / 'installed.json'
ready = marker.is_file() and json.loads(marker.read_text()).get('install_id') == runner.INSTALL_ID
if ready:
    print('Ambiente già installato: verifico soltanto dipendenze e GPU.', flush=True)
    runner.LOG = runner.ROOT / 'verify-bootstrap.log'
    runner.prepare()
    runner.verify()
else:
    import subprocess
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    runner.run_step('setup', revision)
PY

python3 prepare_models.py
printf '\n\033[1;32mPRONTO.\033[0m Apri VideoFake-tools/Runpod_VideoFake.ipynb in JupyterLab.\n'
printf 'Carica i file in /workspace/input. Esegui solo gli step 0, 4 e 5.\n'
printf 'Risultati in /workspace/output. Nessun video avviato automaticamente.\n'
