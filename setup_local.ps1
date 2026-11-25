# Native Windows. IndicTrans2 has no Windows wheel, so this path translates with NLLB-200 (FACTINDIC_MT=nllb).
# Prefer WSL2 + setup_local.sh when it is available.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}
if (-not (Test-Path .venv)) { uv venv --python 3.12 .venv }
uv pip install --python .venv\Scripts\python.exe torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv\Scripts\python.exe -r requirements-local.txt
Write-Host "Next:  .venv\Scripts\activate ; hf auth login ; `$env:FACTINDIC_MT='nllb' ; python run_local.py doctor"
