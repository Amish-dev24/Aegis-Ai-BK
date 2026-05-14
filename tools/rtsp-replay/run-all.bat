@echo off
setlocal

set "VIDEO_FILE=%~1"
if "%VIDEO_FILE%"=="" set "VIDEO_FILE=C:\Users\Admin\Downloads\Violence Detection Demo.mp4"

set "STREAM_PATH=%~2"
if "%STREAM_PATH%"=="" set "STREAM_PATH=cam1"

if not exist "%VIDEO_FILE%" (
  echo Video file not found:
  echo   %VIDEO_FILE%
  echo.
  echo Usage:
  echo   run-all.bat "C:\path\to\video.mp4" [stream_path]
  exit /b 1
)

echo Starting RTSP server in a new window...
start "RTSP Server (MediaMTX)" cmd /k ""%~dp0start-rtsp-server.bat""

echo Waiting 4 seconds for server startup...
timeout /t 4 /nobreak >nul

echo Starting publisher in a new window...
start "RTSP Publisher (FFmpeg)" cmd /k ""%~dp0publish-video.bat" "%VIDEO_FILE%" %STREAM_PATH%"

echo.
echo Started.
echo Stream URL: rtsp://localhost:8554/%STREAM_PATH%
echo.
echo Tip: keep both windows open while streaming.

endlocal
