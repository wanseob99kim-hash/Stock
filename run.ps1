# PowerShell launcher: .\run.ps1
Set-Location -Path $PSScriptRoot
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Host "[1/3] Creating virtual environment..."
    python -m venv .venv
}
Write-Host "[2/3] Installing packages..."
& $py -m pip install -q --upgrade pip
& $py -m pip install -q -r requirements.txt
Write-Host "[3/3] Starting app - your browser will open. Press Ctrl+C to stop."
& $py -m streamlit run app.py
