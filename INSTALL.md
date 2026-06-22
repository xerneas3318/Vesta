# Vesta — Install & Startup Guide

This guide walks you from a fresh machine to a running Vesta server reachable at
`http://127.0.0.1:33263`, with a **fully local** LLM backend. No cloud accounts,
no API keys, no outbound traffic at inference time.

There are three pieces:

1. **Vesta** — the Flask app + YOLO person detector. (This repo.)
2. **YOLO weights** — `yolo26s.onnx` (already in `person-detect/`) and optionally
   `yolo26s.pt` for GPU acceleration.
3. **A local llama.cpp server** — an OpenAI-compatible vision-LLM endpoint that
   Vesta talks to over HTTP.

---

## 1. System requirements

**Recommended (production-style):**
- Linux (Ubuntu 22.04+) or macOS 14+.
- NVIDIA GPU with CUDA 12+ (8 GB+ VRAM) for both YOLO and the vision LLM.
- 32 GB RAM.
- 50 GB free disk (models + cached clips).

**Minimum (CPU-only, slow):**
- Any x86-64 or Apple Silicon machine.
- 16 GB RAM.
- Expect minute-scale latency per minute of footage.

**Required on `PATH`:**
- Python **3.12 or newer**
- [`uv`](https://docs.astral.sh/uv/) — Python package + venv manager
- `ffmpeg` and `ffprobe`
- `git`

Install `uv` (one-time):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Install `ffmpeg`:

- macOS: `brew install ffmpeg`
- Ubuntu/Debian: `sudo apt install -y ffmpeg`

---

## 2. Clone Vesta

```bash
git clone https://github.com/xerneas3318/Vesta.git
cd Vesta
```

Install Python deps into a managed virtualenv:

```bash
uv sync
```

This pulls in Flask, PyTorch, ultralytics (YOLO), OpenCV, ONNX runtime, and
Pillow. On a fresh machine this can take a few minutes; PyTorch wheels are
large.

---

## 3. YOLO weights

The ONNX export ships with the repo:

```
person-detect/yolo26s.onnx
```

That's enough to run on CPU or on GPU via ONNX Runtime.

If you have a CUDA GPU and want the faster PyTorch path, place the matching
`.pt` weights at:

```
person-detect/yolo26s.pt
```

(`.pt` files are gitignored — bring your own and copy them in.) Vesta will
prefer `.pt` on CUDA and fall back to `.onnx` otherwise.

---

## 4. Stand up a local LLM server (llama.cpp)

Vesta needs a vision-capable, OpenAI-compatible chat endpoint at
`POST /v1/chat/completions`. The reference backend is
[`llama.cpp`](https://github.com/ggml-org/llama.cpp).

### 4a. Build llama.cpp

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
# CPU build:
cmake -B build
cmake --build build --config Release -j
# Or, with CUDA:
# cmake -B build -DGGML_CUDA=ON
# cmake --build build --config Release -j
```

The server binary will land at `build/bin/llama-server`.

### 4b. Pull a vision model

You need a multimodal GGUF — Vesta sends mosaics as image inputs. Two known-good
options:

- **Qwen2-VL 7B Instruct** (good quality, ~6 GB at Q4):
  - `Qwen2-VL-7B-Instruct-Q4_K_M.gguf`
  - matching `mmproj-Qwen2-VL-7B-Instruct-f16.gguf`
- **LLaVA 1.6 Mistral 7B** (lighter):
  - `llava-v1.6-mistral-7b.Q4_K_M.gguf`
  - matching `mmproj-llava-v1.6-mistral-7b-f16.gguf`

Download both files from Hugging Face into a directory of your choice, e.g.
`~/models/`. You need **both** the main model **and** its `mmproj` projector
file for vision to work.

### 4c. Start the server on port 8078

```bash
./build/bin/llama-server \
  --host 127.0.0.1 --port 8078 \
  -m ~/models/Qwen2-VL-7B-Instruct-Q4_K_M.gguf \
  --mmproj ~/models/mmproj-Qwen2-VL-7B-Instruct-f16.gguf \
  -c 8192 \
  --n-gpu-layers 99   # drop this flag for CPU-only
```

Sanity check from another shell:

```bash
curl http://127.0.0.1:8078/v1/models
```

You should get back a JSON list of one model. Keep this server running in its
own terminal (or behind `systemd` / `tmux` / `launchd` for a long-lived
install).

---

## 5. Configure Vesta

All knobs are environment variables — defaults are fine for the localhost
setup above. Override them in your shell or in a `.env` file you `source`
before `./run.sh`.

| Variable | Default | Purpose |
|---|---|---|
| `LLAMACPP_BASE_URL` | `http://127.0.0.1:8078` | Base URL of the llama.cpp server |
| `LLAMACPP_MODEL` | `local-model` | Model name sent in the request (llama.cpp accepts anything) |
| `REQUIRE_GPU` | `1` | Set `0` to allow CPU fallback for YOLO |
| `VIDEO_BATCH_SIZE` | `32` | YOLO inference batch size |
| `YOLO_FRAME_WORKERS` | `4` | Parallel frame decoders |
| `MAX_DYNAMIC_MOSAICS` | `24` | Cap on mosaics built per video |
| `MOSAIC_SCALE_DIVISOR` | `8` | Larger = smaller mosaic tiles |
| `LLM_MAX_BATCH_REQUESTS` | `16` | Concurrent in-flight captioning calls |

CPU-only example:

```bash
REQUIRE_GPU=0 ./run.sh
```

---

## 6. Start Vesta

From the repo root:

```bash
./run.sh
```

Equivalent long form:

```bash
uv run flask --app main:app run --host 0.0.0.0 --port 33263
```

Then open:

- `http://127.0.0.1:33263` on the host itself, or
- `http://<host-lan-ip>:33263` from another machine on the same network.

Upload a video and you should see two progress bars: one for person detection,
one for mosaic captioning + summary.

---

## 7. Where things live

After the first run, the repo gains a `runtime/` directory:

```
runtime/
├── uploads/             # raw uploads
├── outputs/             # copies for browser playback
└── cache/
    └── video_clips/
        └── <key>/
            ├── clip-*.mp4
            └── manifest.json
```

Cached outputs are reused on repeat uploads of the same video (keyed by file
hash). To wipe state, delete `runtime/`.

---

## 8. Running as a service (optional)

For an always-on school deployment, run llama.cpp and Vesta under a process
supervisor.

Minimal `systemd` units (Linux):

`/etc/systemd/system/vesta-llm.service`
```
[Unit]
Description=Vesta local LLM (llama.cpp)
After=network.target

[Service]
ExecStart=/opt/llama.cpp/build/bin/llama-server \
  --host 127.0.0.1 --port 8078 \
  -m /opt/models/Qwen2-VL-7B-Instruct-Q4_K_M.gguf \
  --mmproj /opt/models/mmproj-Qwen2-VL-7B-Instruct-f16.gguf \
  -c 8192 --n-gpu-layers 99
Restart=on-failure
User=vesta

[Install]
WantedBy=multi-user.target
```

`/etc/systemd/system/vesta.service`
```
[Unit]
Description=Vesta web UI
After=vesta-llm.service
Requires=vesta-llm.service

[Service]
WorkingDirectory=/opt/Vesta
Environment=LLAMACPP_BASE_URL=http://127.0.0.1:8078
ExecStart=/opt/Vesta/run.sh
Restart=on-failure
User=vesta

[Install]
WantedBy=multi-user.target
```

Enable both:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now vesta-llm vesta
```

---

## 9. Troubleshooting

**"GPU mode is required but CUDA is unavailable."**
Either install a CUDA-enabled PyTorch build, or rerun with `REQUIRE_GPU=0`.

**"Could not reach llama.cpp at http://127.0.0.1:8078"**
The LLM server isn't running, isn't bound to that port, or isn't reachable.
Verify with `curl http://127.0.0.1:8078/v1/models`.

**Mosaics generate but captions are empty / nonsense.**
Your model is loaded but the **`mmproj` file is missing** — without it,
llama.cpp ignores the image content. Restart with `--mmproj <path>`.

**Uploads succeed but no clips appear.**
Check that `ffmpeg` and `ffprobe` are both on `PATH` (`which ffmpeg ffprobe`).

**Out-of-memory on the GPU.**
Drop `--n-gpu-layers`, switch to a smaller quant (e.g. `Q3_K_M`), or lower
`VIDEO_BATCH_SIZE` for YOLO.

---

If you get stuck, open an issue on the [Vesta repo](https://github.com/xerneas3318/Vesta/issues).
