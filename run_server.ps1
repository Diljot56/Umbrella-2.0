Write-Host "Starting Project Umbrella Server..." -ForegroundColor Cyan
python -m uvicorn src.main:app --host 127.0.0.1 --port 8000 --reload
