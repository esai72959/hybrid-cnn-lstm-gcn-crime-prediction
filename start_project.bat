@echo off
title Hybrid CNN-LSTM Crime Prediction

cd /d D:\Hybrid_CNN_LSTM_Crime_Prediction

echo.
echo ==========================================
echo   Hybrid CNN-LSTM Crime Prediction
echo ==========================================
echo.

call venv\Scripts\activate

echo Starting Django server...
echo Please wait...

start "" /B cmd /c "python manage.py runserver"

timeout /t 5 /nobreak >nul

echo Opening website in Chrome...

start "" "C:\Program Files\Google\Chrome\Application\chrome.exe" "http://127.0.0.1:8000/"

echo.
echo Website started successfully.
echo Keep this window open while using the project.
echo.

pause