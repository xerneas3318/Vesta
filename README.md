# Vesta

**Vesta** is a self-hosted, privacy-first surveillance assistant built for schools.
It pairs on-device person detection with a local large-language-model that describes
what is actually happening on camera — without ever sending video, frames, or
captions to a third-party cloud.

Vesta was developed for **Stratford Schools / Spring Education Group** as a tool
to help staff investigate and counter on-campus theft. Footage and analysis stay
on hardware the school controls.

> Named for Vesta, Roman goddess of hearth and home — a quiet guardian.

---

## Why Vesta

Most "AI camera" products ship footage to a vendor's cloud for analysis. For a
school that is unacceptable:

- Student and staff video is sensitive data.
- Network egress of raw footage is expensive and slow.
- Vendor lock-in makes evidence harder to retrieve when something goes wrong.

Vesta is built around the opposite trade-off:

- **Local-first.** Everything — detection, captioning, summarization — runs on a
  machine you own.
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
3. **Builds** temporal mosaics — grids of person-only frames sampled across each
   clip — so a vision-language model can reason over motion in a single image.
4. **Captions** each mosaic and **summarizes** the whole video using a local
   llama.cpp server speaking the OpenAI `chat/completions` protocol.
5. **Serves** all of this through a single Flask web UI with dual-stage progress
   bars and replayable cached outputs.

## Screenshots

**Watch dashboard** — the day-to-day landing page. Active event banner up top, the camera you're watching front-and-center, and the most recent auto-saved clips on the right.

![Watch dashboard](screenshots/01-watch-dashboard.png)

**Live view** — a single camera feed with one-click start/stop recording. Useful when staff want to keep an eye on a specific area in real time.

![Live camera view](screenshots/02-live-camera.png)

**Recordings library with AI search** — every clip is auto-captioned by the local vision-LLM, so you can search by *what happened* ("two people walking past utility cabinets") instead of by filename or timestamp. The right rail is a calendar/timeline of clip density.

![Recordings library with AI captions](screenshots/03-recordings-grid.png)

**Clip detail** — quick metadata pane for a single recording: camera, timestamp, duration, file size, plus a star/rename for easier recall later.

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

`v0.1.0` — first usable release. Single-user web UI, async job endpoints, and
documented install path. Multi-camera ingest and live RTSP are tracked on
feature branches (`feature-rtsp-stream`, `ui-polish`) and are not part of this
release.

## Getting started

See [**INSTALL.md**](INSTALL.md) for the full install + startup guide, including
how to bring up a local llama.cpp server with a vision model.

Short version:

```bash
uv sync
# in a second terminal: start your local llama.cpp server (see INSTALL.md)
./run.sh
# open http://127.0.0.1:33263
```

## Project layout

- `main.py` — unified Flask app, job runner, YOLO + LLM glue.
- `templates/index.html` — single-page web UI.
- `person-detect/` — YOLO assets (`yolo26s.onnx`, optional `yolo26s.pt`) and the
  earlier standalone detection demo.
- `video-understanding/` — earlier mosaic-captioning experiments.
- `runtime/` — created on first run; holds `uploads/`, `outputs/`, and `cache/`.

## License & use

Vesta is an internal tool developed for Stratford Schools / Spring Education
Group. Use outside that context is at your own discretion; review the code
before deploying it anywhere that handles minors' video.

---

Maintained by [@xerneas3318](https://github.com/xerneas3318).
