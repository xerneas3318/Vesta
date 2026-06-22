# Vesta

**Vesta** is a self-hosted, privacy-first surveillance assistant built for schools.
It pairs on-device person detection with a local large-language-model that describes
what is actually happening on camera, without ever sending video, frames, or
captions to a third-party cloud.

Vesta was developed for **Stratford Schools / Spring Education Group** as a tool
to help staff investigate and counter on-campus theft. Footage and analysis stay
on hardware the school controls.

---

## Why Vesta

Most "AI camera" products ship footage to a vendor's cloud for analysis. For a
school that is unacceptable:

- Student and staff video is sensitive data.
- Network egress of raw footage is expensive and slow.
- Vendor lock-in makes evidence harder to retrieve when something goes wrong.

Vesta is built around the opposite trade-off:

- **Local-first.** Detection, captioning, and summarization all run on a machine you own.
- **Localhost-able.** The entire stack can run on a single workstation behind the
  school firewall and be reached at `http://127.0.0.1:33263`.
- **Auditable.** Inputs, outputs, and cached artifacts live on disk in plain
  files you can inspect, archive, or delete.
- **Built for an actual problem.** Designed with the loss-prevention workflow of
  Stratford Schools in mind: upload a clip from a hallway or stockroom camera,
  get back the time ranges that contain people plus a short written description
  of what each person is doing.

## What it does

Upload one video, and Vesta:

1. **Detects** people frame-by-frame with YOLO (GPU when available, CPU
   fallback supported).
2. **Extracts** annotated per-person clips with timestamps.
3. **Builds** temporal mosaics, grids of person-only frames sampled across each
   clip, so a vision-language model can reason over motion in a single image.
4. **Captions** each mosaic and **summarizes** the whole video using a local
   llama.cpp server speaking the OpenAI `chat/completions` protocol.
5. **Serves** all of this through a single Flask web UI with dual-stage progress
   bars and replayable cached outputs.

## Screenshots

**Watch dashboard.** The day-to-day landing page. Active event banner up top, the camera you're watching front-and-center, and the most recent auto-saved clips on the right.

![Watch dashboard](screenshots/01-watch-dashboard.png)

**Live view.** A single camera feed with one-click start/stop recording. Useful when staff want to keep an eye on a specific area in real time.

![Live camera view](screenshots/02-live-camera.png)

**Recordings library with AI search.** Every clip is auto-captioned by the local vision-LLM, so you can search by *what happened* ("two people walking past utility cabinets") instead of by filename or timestamp. The right rail is a calendar/timeline of clip density.

![Recordings library with AI captions](screenshots/03-recordings-grid.png)

**Clip detail.** Quick metadata pane for a single recording: camera, timestamp, duration, file size, plus a star/rename for easier recall later.

![Recording detail](screenshots/04-recording-detail.png)

## Architecture at a glance

```
       browser  ──►  Flask UI  ──►  YOLO  (person detection, GPU/CPU)
                          │
                          └──►  llama.cpp server  (vision-LLM captions + summary)
                          │
                          └──►  runtime/  (uploads, clips, mosaics, cache)
```

Two processes on one host. No outbound network calls are required at inference
time.

## Status

`v0.1.0` is the first usable release. Single-user web UI, async job endpoints, and
documented install path. Multi-camera ingest and live RTSP are tracked on
feature branches (`feature-rtsp-stream`, `ui-polish`) and are not part of this
release.

## Getting started

Vesta has two components that need to be running side by side:

1. **The local LLM backend** ([llama.cpp](https://github.com/ggml-org/llama.cpp))
   that does the vision captioning and summarization.
2. **The Vesta web app**, which does YOLO person detection and talks to the LLM
   over HTTP.

### 1. Bring up the local LLM server

Build llama.cpp (see [their build docs](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md))
and download a vision-capable GGUF plus its matching `mmproj` projector file.
Known-good models: **Qwen2-VL 7B Instruct** or **LLaVA 1.6 Mistral 7B**.

Start the server on port `8078` (the default Vesta expects):

```bash
./build/bin/llama-server \
  --host 127.0.0.1 --port 8078 \
  -m   ~/models/Qwen2-VL-7B-Instruct-Q4_K_M.gguf \
  --mmproj ~/models/mmproj-Qwen2-VL-7B-Instruct-f16.gguf \
  -c 8192 \
  --n-gpu-layers 99   # drop this flag for CPU-only
```

Sanity check: `curl http://127.0.0.1:8078/v1/models` should return JSON.

Override the URL with `LLAMACPP_BASE_URL` if you run the server elsewhere.

### 2. Start Vesta

```bash
uv sync
./run.sh
# open http://127.0.0.1:33263
```

Full step-by-step (system packages, YOLO weights, systemd units, troubleshooting):
see [**INSTALL.md**](INSTALL.md).

## Project layout

- `main.py`: unified Flask app, job runner, YOLO + LLM glue.
- `templates/index.html`: single-page web UI.
- `person-detect/`: YOLO assets (`yolo26s.onnx`, optional `yolo26s.pt`) and the
  earlier standalone detection demo.
- `video-understanding/`: earlier mosaic-captioning experiments.
- `runtime/`: created on first run; holds `uploads/`, `outputs/`, and `cache/`.

## License & use

Vesta is an internal tool developed for Stratford Schools / Spring Education
Group. Use outside that context is at your own discretion; review the code
before deploying it anywhere that handles minors' video.

---

Maintained by [@xerneas3318](https://github.com/xerneas3318).
