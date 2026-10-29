@echo off
title FaceSense AI - Facial Expression Recognition Dev Server
color 0B
cls

echo =========================================================================
echo   FaceSense AI - Facial Expression Recognition System
echo   BCA Semester 5 - Neural Networks Mini Project
echo =========================================================================
echo.
echo [*] Starting Development Server (npm run dev)...
echo [*] Local URL: http://127.0.0.1:5000
echo.

:: Open default browser after 2 seconds in background
start "" cmd /c "timeout /t 2 /nobreak >nul & start http://127.0.0.1:5000"

:: Launch the Flask app via Python
python app.py

pause
