@echo off
title TLC - ESP32 LoRa Chat

echo ========================================
echo   TLC - He Thong Chat Qua LoRa
echo   ESP32-C3 + LoRa SX1278 (433MHz)
echo ========================================
echo.

REM Kiem tra Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [LOI] Python chua duoc cai dat!
    pause
    exit /b 1
)

echo [OK] Python: 
python --version
echo.

REM Cai dat thu vien
echo [INFO] Dang cai dat thu vien...
pip install flask flask-cors pyserial crcmod
echo.

echo ========================================
echo   Chon che do:
echo ========================================
echo [1] Chay Web Server + Chat (can ESP32)
echo [2] Chi chay Web TLLa
echo [3] Thoat
echo.

set /p choice="Nhap lua chon (1-3): "

if "%choice%"=="1" goto run_full
if "%choice%"=="2" goto run_web
if "%choice%"=="3" goto exit

:run_full
echo.
echo [INFO] Khoi dong Web + Chat Server...
echo [WARN] Dam bao ESP32 da ket noi qua USB!
echo [WARN] Neu loi, set ESP32_PORT=COMx
echo.
set ESP32_PORT=COM3
start "TLC Web" cmd /c python server.py
timeout /t 2 >nul
start "LoRa Chat" cmd /c python server-chat-real.py
echo.
echo Da khoi dong xong!
echo - Web: http://localhost:5000
echo - Chat API: http://localhost:5001
echo.
pause >nul
goto exit

:run_web
python server.py
goto exit

:exit
echo.
echo Tam biet!
pause