# RTSP Replay Toolkit (Separate from app)

This folder is intentionally standalone and does not modify the main backend/frontend runtime.

It lets you:
- start a local RTSP server (MediaMTX in Docker)
- publish a pre-recorded video file as a looped RTSP stream

## Requirements

- Docker Desktop running
- FFmpeg installed and available in `PATH`

## 1) Start RTSP server

Run:

```bat
start-rtsp-server.bat
```

This creates path `cam1` and serves RTSP at port `8554`.

## 2) Publish a video file to RTSP

Run:

```bat
publish-video.bat "C:\path\to\video.mp4" cam1
```

- First argument: input video path (required)
- Second argument: RTSP path name (optional, default `cam1`)

## One-click start (server + publisher)

Run:

```bat
run-all.bat
```

Defaults to:

- video: `C:\Users\Admin\Downloads\Violence Detection Demo.mp4`
- stream path: `cam1`

Or with custom values:

```bat
run-all.bat "C:\path\to\video.mp4" cam1
```

Publisher URL pattern:

`rtsp://localhost:8554/<path>`

## URLs to use

- Same machine: `rtsp://localhost:8554/cam1`
- Another LAN machine: `rtsp://<YOUR_PC_IP>:8554/cam1`

## Stop

- Close the FFmpeg terminal to stop publishing.
- Close the Docker/MediaMTX terminal to stop server.

## Notes

- Script uses H264 + AAC for compatibility.
- Stream loops forever (`-stream_loop -1`) so it behaves like a camera feed.
