@echo off
title AgriSmart AI Backend Launcher
echo ====================================================
echo             AgriSmart AI Backend Launcher
echo ====================================================
echo.
echo Starting Python FastAPI Backend (Port 8000)...
start "AgriSmart Python FastAPI Backend" cmd /k "cd python_backend && py -m uvicorn app:app --host 127.0.0.1 --port 8000"

echo Starting Node.js Express Gateway & SQLite (Port 5000)...
start "AgriSmart Node.js Express Gateway" cmd /k "cd node_backend && npm start"

echo.
echo ====================================================
echo Both backends are launching in separate windows!
echo - Python FastAPI API: http://127.0.0.1:8000
echo - Node.js Express Gateway: http://127.0.0.1:5000
echo.
echo Database: SQLite database 'agri_smart.db' initialized in node_backend/
echo ====================================================
pause
