#!/usr/bin/env bash
# Linux or WSL2. Creates .venv with Python 3.12 and CUDA 12.8 builds (RTX 50 needs them).
set -euo pipefail
cd "$(dirname "$0")"
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
[ -d .venv ] || uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python -r requirements-local.txt
echo
echo "Next:  source .venv/bin/activate && hf auth login && python run_local.py doctor"
