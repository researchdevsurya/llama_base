@echo off
title 🔧 Reset MySQL (XAMPP)
echo ================================================
echo        XAMPP MySQL Reset Utility
echo ================================================
echo.

:: 1️⃣ Stop MySQL if running
echo [1/5] Stopping MySQL service if running...
taskkill /F /IM mysqld.exe >nul 2>&1
net stop mysql >nul 2>&1
echo ✅ MySQL stopped (if it was running)
echo.

:: 2️⃣ Change directory to XAMPP MySQL folder
cd /d C:\xampp\mysql

:: 3️⃣ Backup old data folder
if exist data (
    echo [2/5] Backing up existing data folder...
    set BACKUP_FOLDER=data_backup_%DATE:~-4%%DATE:~4,2%%DATE:~7,2%_%TIME:~0,2%%TIME:~3,2%
    set BACKUP_FOLDER=%BACKUP_FOLDER: =0%
    ren data "%BACKUP_FOLDER%"
    echo ✅ Old data folder renamed to "%BACKUP_FOLDER%"
) else (
    echo ⚠️  No existing data folder found, skipping backup.
)
echo.

:: 4️⃣ Copy clean data folder from backup
if exist backup (
    echo [3/5] Restoring clean default MySQL data...
    xcopy backup data /E /I /H /Y >nul
    echo ✅ Fresh MySQL data copied successfully.
) else (
    echo ❌ ERROR: "backup" folder not found in C:\xampp\mysql
    echo Please reinstall XAMPP or copy the backup folder manually.
    pause
    exit /b
)
echo.

:: 5️⃣ Delete any leftover log files
echo [4/5] Cleaning old log files...
del /F /Q data\aria_log.* >nul 2>&1
del /F /Q data\ib_logfile* >nul 2>&1
del /F /Q data\ibdata1 >nul 2>&1
echo ✅ Log cleanup complete.
echo.

:: 6️⃣ Restart XAMPP MySQL
echo [5/5] Starting MySQL again...
start "" "C:\xampp\xampp-control.exe"
echo ✅ XAMPP Control Panel started. Please click START for MySQL.
echo.

echo ================================================
echo 🎉 MySQL reset completed successfully!
echo - Open XAMPP Control Panel
echo - Click START next to MySQL
echo - Visit: http://localhost/phpmyadmin
echo   (Username: root, Password: leave empty)
echo ================================================
pause
