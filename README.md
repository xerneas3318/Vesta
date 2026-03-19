# Surveillance Unified UI

Unified Flask app for two video tasks in one flow:

- Person detection with YOLO (GPU-first, with optional CPU fallback)
- Temporal-mosaic video understanding via a llama.cpp OpenAI-compatible endpoint

Upload one video, then the app:

1. Detects person-containing time segments and exports annotated clips.
2. Builds person-only temporal mosaics.
3. Generates per-mosaic captions and a final video-level summary.

## Features

- Single web UI with upload + dual-stage progress bars.
- Person clip extraction with timestamps and cached outputs.
- Dynamic number of mosaics based on detected person-frame density.
- Async analysis job endpoints (`/analyze/start`, `/analyze/status/<job_id>`, `/analyze/result/<job_id>`).

## Project Structure

- `main.py`: unified Flask app.
- `templates/index.html`: web UI.
- `runtime/uploads`, `runtime/outputs`, `runtime/cache`: runtime artifacts.
- `person-detect/`: YOLO assets (`yolo26s.pt` / `yolo26s.onnx`) and related scripts.
- `video-understanding/`: earlier standalone experiments.

## Prerequisites

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/)
- `ffmpeg` + `ffprobe` available on `PATH`
- CUDA GPU recommended (default mode requires GPU)
- Running llama.cpp-compatible server exposing `POST /v1/chat/completions`

## Install

```bash
uv sync
```

## Run

```bash
./run.sh
```

Default URL:

- http://127.0.0.1:33263

Equivalent command:

```bash
uv run flask --app main:app run --host 0.0.0.0 --port 33263
```

## Environment Variables

- `LLAMACPP_BASE_URL` (default: `http://127.0.0.1:8078`)
- `LLAMACPP_MODEL` (default: `local-model`)
- `REQUIRE_GPU` (default: `1`)
  - `1`/`true`: fail if CUDA is unavailable
  - `0`/`false`: allow CPU fallback
- `VIDEO_BATCH_SIZE` (default: `32`)
- `MAX_DYNAMIC_MOSAICS` (default: `24`)
- `MOSAIC_SCALE_DIVISOR` (default: `8`)

Example (allow CPU fallback):

```bash
REQUIRE_GPU=0 ./run.sh
```

## Output Artifacts

- Source uploads copied to `runtime/outputs/` for playback.
- Person clips cached under `runtime/cache/video_clips/<cache_key>/`.
- Clip metadata stored as `manifest.json` in each cache directory.
- Files are served via `/files/<path>` from the `runtime/` directory.

## Notes

- Supported video types: `.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`.
- Person class filtering is automatic based on model label names.
- If both detection and understanding fail, UI reports a combined failure.

## Related Docs

- `person-detect/README.md` for the standalone person-detection demo.
