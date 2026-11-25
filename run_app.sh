#!/usr/bin/env bash
# Starts the demo: FastAPI/uvicorn on 127.0.0.1:8000 and Streamlit on http://localhost:8501. Ctrl-C stops both.
set -euo pipefail
cd "$(dirname "$0")"
[ -x .venv/bin/python ] || { echo "Run setup_local.sh first."; exit 1; }
.venv/bin/python -c "import fastapi, uvicorn, multipart, pypdf, docx" 2>/dev/null || uv pip install --python .venv/bin/python -r app/requirements-api.txt
[ -x .venv_ui/bin/streamlit ] || { uv venv --python 3.12 .venv_ui; uv pip install --python .venv_ui/bin/python -r app/requirements-ui.txt; }
API_PORT="${API_PORT:-8000}"; UI_PORT="${UI_PORT:-8501}"
export PYTHONUTF8=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export MALLOC_ARENA_MAX=2
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs/app
.venv/bin/python -m uvicorn app.api:app --host 127.0.0.1 --port "$API_PORT" > runs/app/api.log 2>&1 & A=$!
FACTINDIC_API="http://127.0.0.1:$API_PORT" .venv_ui/bin/streamlit run app/ui.py --server.address 127.0.0.1 --server.port "$UI_PORT" \
  --server.headless true --server.maxUploadSize 10 --browser.gatherUsageStats false > runs/app/ui.log 2>&1 & B=$!
trap 'kill $A $B 2>/dev/null; wait' EXIT
trap 'exit 0' INT TERM
echo "UI:  http://localhost:$UI_PORT"; echo "API: http://127.0.0.1:$API_PORT/docs"; echo "Logs: runs/app/  (the models need about a minute)"
wait -n $A $B; echo "one of the two stopped; see runs/app/*.log"
