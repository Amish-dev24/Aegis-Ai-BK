@echo off
setlocal

echo Starting MediaMTX RTSP server on port 8554...
echo.

docker run --rm ^
  -p 8554:8554 ^
  -v "%~dp0mediamtx.yml:/mediamtx.yml" ^
  bluenviron/mediamtx

endlocal
