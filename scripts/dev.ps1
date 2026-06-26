$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
if (!(Test-Path ".venv")) {
  python -m venv .venv
}
& .\.venv\Scripts\python -m pip install -U pip
& .\.venv\Scripts\python -m pip install -e ".[dev]"
Write-Host "Backend: .\.venv\Scripts\python -m uvicorn apps.api.main:app --reload --port 8000"
Write-Host "Frontend: cd apps\web; npm install; npm run dev"

