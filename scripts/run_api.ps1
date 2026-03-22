# Run the forecasting API on http://0.0.0.0:8000
# Usage: .\scripts\run_api.ps1
Set-Location $PSScriptRoot\..
python -m uvicorn src.api:app --host 0.0.0.0 --port 8000
