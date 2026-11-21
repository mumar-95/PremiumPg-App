# Run Premium PG Assistant with the project virtualenv (not system Python 3.14).
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    Write-Host "Creating .venv..."
    python -m venv .venv
    .\.venv\Scripts\python.exe -m pip install -r requirements.txt
}

Write-Host "Using: $(.\.venv\Scripts\python.exe -c 'import sys; print(sys.executable)')"
Write-Host "Starting Streamlit on http://localhost:8501 ..."
Write-Host "Do NOT use bare 'streamlit' (may hit Python 3.14). Use this script or:"
Write-Host "  .\.venv\Scripts\python.exe -m streamlit run app.py"
& .\.venv\Scripts\python.exe -m streamlit run app.py `
    --server.port 8501 `
    --server.headless true `
    --server.fileWatcherType none
