@echo off
setlocal

if "%~1"=="" (
  echo Usage: publish-video.bat "C:\path\to\video.mp4" [stream_path]
  exit /b 1
)

set "INPUT_FILE=%~1"
set "STREAM_PATH=%~2"
if "%STREAM_PATH%"=="" set "STREAM_PATH=cam1"

echo Publishing "%INPUT_FILE%" to rtsp://localhost:8554/%STREAM_PATH%
echo Press Ctrl+C to stop publishing.
echo.

where ffmpeg >nul 2>nul
if %errorlevel%==0 (
  ffmpeg -re -stream_loop -1 -i "%INPUT_FILE%" ^
    -c copy ^
    -rtsp_transport tcp ^
    -f rtsp "rtsp://localhost:8554/%STREAM_PATH%"
  goto :eof
)

echo Local ffmpeg not found. Trying Docker ffmpeg...
docker run --rm --add-host=host.docker.internal:host-gateway ^
  -v "C:/Users/Admin/Downloads:/media" jrottenberg/ffmpeg:6.1-ubuntu ^
  -re -stream_loop -1 -i "/media/%~nx1" ^
  -c copy ^
  -rtsp_transport tcp ^
  -f rtsp "rtsp://host.docker.internal:8554/%STREAM_PATH%"

if not %errorlevel%==0 (
  echo.
  echo Could not start publisher.
  echo Ensure Docker Desktop is running in Linux container mode,
  echo or install ffmpeg and rerun this script.
  exit /b 1
)

endlocal
