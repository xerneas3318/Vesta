# Person Detection Web Demo

This project converts `yolo26s.pt` to ONNX and serves a local web demo where you can upload an image or video and see person detections directly in the browser.

## 1) Install dependencies (UV project)

```bash
uv sync
```

## 2) Convert model to ONNX

```bash
uv run python convert_to_onnx.py
```

This creates `yolo26s.onnx`.

## 3) Run web app (port exposed on this device)

```bash
uv run uvicorn app:app --host 0.0.0.0 --port 8000
```

Open:

- http://127.0.0.1:8000
- or from another device on the same network: `http://<this-device-ip>:8000`

## Notes

- The app filters detections to class `person`.
- Inference now auto-selects GPU (`cuda:0`) when available by using `yolo26s.pt`; otherwise it falls back to ONNX/CPU.
- Video uploads generate separate person-only clips per detected time range (with person bounding boxes drawn).
- Clip metadata includes start/end timestamps and duration for each clip.
- Clipped outputs are cached in `cache/video_clips/` and reused on repeat uploads.
- Uploaded files are stored in `uploads/`.
- Processed outputs are stored in `outputs/`.
