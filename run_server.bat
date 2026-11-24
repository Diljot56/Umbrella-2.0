@echo off
title Project Umbrella Server
cd /d "%~dp0"
python -m uvicorn src.main:app --host 127.0.0.1 --port 8000 --reload
pause
