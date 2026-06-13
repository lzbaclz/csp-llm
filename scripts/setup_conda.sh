#!/usr/bin/env bash
# Create / refresh the repo-local conda env `csp-llm`.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "${CONDA_EXE%/bin/conda}/etc/profile.d/conda.sh"

ENV=csp-llm

if conda env list | awk '{print $1}' | grep -qx "$ENV"; then
  echo "[setup] env $ENV exists, activating"
else
  echo "[setup] creating env $ENV (python 3.11)"
  conda create -n "$ENV" python=3.11 -y
fi

conda activate "$ENV"

echo "[setup] installing PyTorch cu124 + xqp[eval]"
pip install -q "torch==2.6.0" --index-url https://download.pytorch.org/whl/cu124
pip install -q -e "$ROOT[eval]"

python -c "
import torch
from importlib.metadata import version
print('env:', '$ENV')
print('python:', __import__('sys').version.split()[0])
print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())
print('xqp:', version('xqp'))
print('gpu0:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')
"

echo "[setup] done — run: conda activate $ENV"
