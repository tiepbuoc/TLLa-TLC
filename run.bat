@echo off
title Hệ Thống Dịch TLLa

echo ========================================
echo   Trình Khởi Động Hệ Thống Dịch TLLa
echo ========================================
echo.

REM Kiểm tra xem Python đã được cài đặt
python --version >nul 2>&1
if errorlevel 1 (
    echo [LỖI] Python không được cài đặt hoặc không trong PATH.
    echo Vui lòng cài đặt Python 3.7 trở lên trước.
    pause
    exit /b 1
)

echo [OK] Tìm thấy Python: 
python --version
echo.

REM Kiểm tra requirements.txt có tồn tại
if not exist "requirements.txt" (
    echo [LỖI] Không tìm thấy requirements.txt!
    pause
    exit /b 1
)

echo [THÔNG TIN] Đang kiểm tra và cài đặt các gói còn thiếu...
echo.

REM Cài đặt các gói thiếu một cách im lặng (chỉ hiện cảnh báo/lỗi)
pip install -q flask 2>nul
if errorlevel 1 (
    echo [CẢNH BÁO] Không thể cài đặt flask, đang thử --user...
    pip install -q --user flask
)

pip install -q torch 2>nul
if errorlevel 1 (
    echo [CẢNH BÁO] Không thể cài đặt torch, đang thử --user...
    pip install -q --user torch
)

pip install -q numpy 2>nul
if errorlevel 1 (
    echo [CẢNH BÁO] Không thể cài đặt numpy, đang thử --user...
    pip install -q --user numpy
)

echo [OK] Kiểm tra gói hoàn tất.
echo.

REM Kiểm tra server.py có tồn tại
if not exist "server.py" (
    echo [LỖI] Không tìm thấy server.py!
    pause
    exit /b 1
)

echo [THÔNG TIN] Đang khởi động Máy chủ Dịch TLLa...
echo.
echo ========================================
echo   Máy chủ đang chạy tại: http://localhost:5000
echo   Nhấn Ctrl+C để dừng máy chủ
echo ========================================
echo.

python server.py

pause