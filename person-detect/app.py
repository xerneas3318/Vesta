from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Iterator

import cv2
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import torch
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
PT_MODEL_PATH = BASE_DIR / "yolo26s.pt"
ONNX_MODEL_PATH = BASE_DIR / "yolo26s.onnx"
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
CACHE_DIR = BASE_DIR / "cache"
CLIPS_CACHE_DIR = CACHE_DIR / "video_clips"
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
REQUIRE_GPU = os.getenv("REQUIRE_GPU", "1").strip().lower() not in {"0", "false", "no"}
VIDEO_BATCH_SIZE = max(1, int(os.getenv("VIDEO_BATCH_SIZE", "32")))
YOLO_FRAME_WORKERS = max(1, int(os.getenv("YOLO_FRAME_WORKERS", "4")))

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
CLIPS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Person Detection Demo")
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")
app.mount("/cache", StaticFiles(directory=CACHE_DIR), name="cache")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

model: YOLO | None = None
loaded_model_path: Path | None = None
inference_device: str = "cpu"
inference_half: bool = False


def allowed_media(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        return "image"
    if ext in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
        return "video"
    raise HTTPException(
        status_code=400,
        detail="Unsupported file type. Upload an image or video file.",
    )


def select_model_and_device() -> tuple[Path, str, bool]:
    if torch.cuda.is_available():
        if PT_MODEL_PATH.exists():
            return PT_MODEL_PATH, "cuda:0", True
        if ONNX_MODEL_PATH.exists():
            return ONNX_MODEL_PATH, "cuda:0", False
        raise HTTPException(
            status_code=500,
            detail=(
                f"CUDA is available but no model was found. Expected {PT_MODEL_PATH.name} "
                f"or {ONNX_MODEL_PATH.name} in {BASE_DIR}."
            ),
        )

    if REQUIRE_GPU:
        raise HTTPException(
            status_code=500,
            detail=(
                "GPU mode is required but CUDA is unavailable. "
                "Set REQUIRE_GPU=0 to allow CPU fallback."
            ),
        )

    if ONNX_MODEL_PATH.exists():
        return ONNX_MODEL_PATH, "cpu", False
    if PT_MODEL_PATH.exists():
        return PT_MODEL_PATH, "cpu", False

    raise HTTPException(
        status_code=500,
        detail=(
            f"No model found. Expected either {PT_MODEL_PATH.name} "
            f"or {ONNX_MODEL_PATH.name} in {BASE_DIR}."
        ),
    )


def get_model() -> YOLO:
    global model, loaded_model_path, inference_device, inference_half
    model_path, device, half = select_model_and_device()

    if model is None or loaded_model_path != model_path:
        model = YOLO(str(model_path))
        loaded_model_path = model_path

    inference_device = device
    inference_half = half and device.startswith("cuda")
    return model


def get_person_class_ids(detector: YOLO) -> list[int]:
    names = detector.names
    person_ids: set[int] = set()
    name_items = names.items() if isinstance(names, dict) else enumerate(names)
    for class_id, class_name in name_items:
        if str(class_name).lower() == "person":
            person_ids.add(int(class_id))
    if not person_ids:
        return [0]
    return sorted(person_ids)


def unique_name(prefix: str, suffix: str) -> str:
    return f"{prefix}_{int(time.time() * 1000)}{suffix}"


def run_cmd(cmd: list[str], err_prefix: str) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode == 0:
        return
    stderr_tail = "\n".join(proc.stderr.splitlines()[-12:])
    raise HTTPException(status_code=500, detail=f"{err_prefix}\n{stderr_tail}")


def runtime_status() -> dict[str, str]:
    model_name = loaded_model_path.name if loaded_model_path else "not loaded yet"
    return {
        "model": model_name,
        "device": inference_device,
        "precision": "fp16" if inference_half else "fp32",
        "policy": "gpu-only" if REQUIRE_GPU else "auto-fallback",
        "video_batch_size": str(VIDEO_BATCH_SIZE),
        "yolo_frame_workers": str(YOLO_FRAME_WORKERS),
    }


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        while True:
            chunk = file_obj.read(block_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def format_timestamp(seconds: float) -> str:
    total_ms = int(seconds * 1000)
    mins, rem_ms = divmod(total_ms, 60_000)
    secs, ms = divmod(rem_ms, 1000)
    return f"{mins:02d}:{secs:02d}.{ms:03d}"


def probe_video_metadata(src_path: Path) -> tuple[float, float]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=r_frame_rate,duration",
        "-of",
        "json",
        str(src_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise HTTPException(status_code=400, detail="Could not read video metadata.")
    data = json.loads(proc.stdout)
    streams = data.get("streams", [])
    if not streams:
        raise HTTPException(status_code=400, detail="No video stream found in the uploaded file.")

    stream = streams[0]
    fps_raw = str(stream.get("r_frame_rate", "25/1"))
    duration_raw = stream.get("duration")
    try:
        numerator, denominator = fps_raw.split("/")
        fps = float(numerator) / max(float(denominator), 1.0)
    except Exception:
        fps = 25.0
    if fps <= 0:
        fps = 25.0

    duration = float(duration_raw) if duration_raw is not None else 0.0
    if duration <= 0:
        duration = 0.001
    return fps, duration


def iter_video_frame_batches(src_path: Path, batch_size: int) -> Iterator[tuple[int, list[Any]]]:
    capture = cv2.VideoCapture(str(src_path))
    if not capture.isOpened():
        raise HTTPException(status_code=400, detail="Could not open uploaded video for decoding.")

    next_frame_idx = 0
    batch: list[Any] = []
    batch_start_idx = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if not batch:
                batch_start_idx = next_frame_idx
            batch.append(frame)
            next_frame_idx += 1
            if len(batch) >= batch_size:
                yield batch_start_idx, batch
                batch = []
        if batch:
            yield batch_start_idx, batch
    finally:
        capture.release()


def detect_person_segments(
    src_path: Path,
    detector: YOLO,
    person_class_ids: list[int],
    conf: float,
    pre_pad_s: float = 0.4,
    post_pad_s: float = 0.4,
) -> tuple[list[tuple[float, float]], int]:
    fps, duration = probe_video_metadata(src_path)

    active_start: int | None = None
    frame_segments: list[tuple[int, int]] = []
    processed_frames = 0
    for batch_start_idx, frame_batch in iter_video_frame_batches(src_path, batch_size=VIDEO_BATCH_SIZE):
        batch_results = detector.predict(
            source=frame_batch,
            conf=conf,
            classes=person_class_ids,
            device=inference_device,
            half=inference_half,
            workers=YOLO_FRAME_WORKERS,
            stream=False,
            save=False,
            verbose=False,
        )
        for idx_in_batch, result in enumerate(batch_results):
            frame_idx = batch_start_idx + idx_in_batch
            processed_frames = frame_idx + 1
            has_person = result.boxes is not None and len(result.boxes) > 0
            if has_person and active_start is None:
                active_start = frame_idx
            elif not has_person and active_start is not None:
                frame_segments.append((active_start, frame_idx - 1))
                active_start = None

    if active_start is not None:
        frame_segments.append((active_start, max(processed_frames - 1, active_start)))

    if not frame_segments:
        return [], processed_frames

    time_segments: list[tuple[float, float]] = []
    for start_frame, end_frame in frame_segments:
        start_s = max(0.0, (start_frame / fps) - pre_pad_s)
        end_s = min(duration, ((end_frame + 1) / fps) + post_pad_s)
        if end_s > start_s:
            time_segments.append((start_s, end_s))

    merged: list[tuple[float, float]] = []
    for start_s, end_s in time_segments:
        if not merged:
            merged.append((start_s, end_s))
            continue
        prev_start, prev_end = merged[-1]
        if start_s <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end_s))
        else:
            merged.append((start_s, end_s))

    return merged, processed_frames


def cut_video_segment(src_path: Path, start_s: float, end_s: float, out_path: Path) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start_s:.3f}",
        "-to",
        f"{end_s:.3f}",
        "-i",
        str(src_path),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(out_path),
    ]
    run_cmd(cmd, "Failed to cut a person clip segment.")


def create_annotated_source_video(
    src_path: Path,
    clip_dir: Path,
    detector: YOLO,
    person_class_ids: list[int],
    conf: float,
) -> Path:
    ann_root = clip_dir / "annotated_source"
    detector.predict(
        source=str(src_path),
        conf=conf,
        classes=person_class_ids,
        device=inference_device,
        half=inference_half,
        workers=YOLO_FRAME_WORKERS,
        stream=False,
        save=True,
        project=str(clip_dir),
        name="annotated_source",
        exist_ok=True,
        verbose=False,
    )
    default_out = ann_root / src_path.name
    if default_out.exists():
        return default_out

    candidates = sorted(
        (p for p in ann_root.iterdir() if p.is_file()),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise HTTPException(status_code=500, detail="Failed to produce annotated video.")
    return candidates[-1]


def run_video_clipper(src_path: Path, conf: float) -> tuple[list[dict[str, Any]], str]:
    detector = get_model()
    person_class_ids = get_person_class_ids(detector)
    file_hash = sha256_file(src_path)
    model_name = loaded_model_path.stem if loaded_model_path else "unknown"
    runtime_name = "gpu" if inference_device.startswith("cuda") else "cpu"
    cache_key = f"{file_hash[:16]}_{model_name}_{runtime_name}_c{int(conf * 1000):03d}"
    clip_dir = CLIPS_CACHE_DIR / cache_key
    manifest_path = clip_dir / "manifest.json"

    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            clips = manifest.get("clips", [])
            if clips:
                return clips, "Cache hit: reused existing person clips."
        except Exception:
            pass

    clip_dir.mkdir(parents=True, exist_ok=True)
    segments, processed_frames = detect_person_segments(src_path, detector, person_class_ids, conf=conf)
    if not segments:
        raise HTTPException(status_code=404, detail="No person detected in this video.")

    annotated_video = create_annotated_source_video(src_path, clip_dir, detector, person_class_ids, conf)
    clips: list[dict[str, Any]] = []
    for idx, (start_s, end_s) in enumerate(segments):
        clip_name = f"clip_{idx:03d}.mp4"
        segment_path = clip_dir / clip_name
        if not segment_path.exists():
            cut_video_segment(annotated_video, start_s, end_s, segment_path)
        clips.append(
            {
                "index": idx + 1,
                "url": f"/cache/video_clips/{cache_key}/{clip_name}",
                "start_s": round(start_s, 3),
                "end_s": round(end_s, 3),
                "duration_s": round(end_s - start_s, 3),
                "start_label": format_timestamp(start_s),
                "end_label": format_timestamp(end_s),
            }
        )

    manifest = {
        "source_sha256": file_hash,
        "confidence": conf,
        "model": loaded_model_path.name if loaded_model_path else "unknown",
        "device": inference_device,
        "segments": [{"start_s": start_s, "end_s": end_s} for start_s, end_s in segments],
        "segment_count": len(segments),
        "processed_frames": processed_frames,
        "annotated_source": str(annotated_video.name),
        "clips": clips,
        "created_at_ms": int(time.time() * 1000),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return clips, f"Created {len(clips)} person clip(s) with bounding boxes and cached them."


def run_inference(
    src_path: Path,
    conf: float,
    media_kind: str,
) -> tuple[str | None, str | None, list[dict[str, Any]]]:
    if media_kind == "video":
        clips, note = run_video_clipper(src_path, conf=conf)
        return None, note, clips

    detector = get_model()
    person_class_ids = get_person_class_ids(detector)
    run_dir_name = unique_name("result", "")

    detector.predict(
        source=str(src_path),
        conf=conf,
        classes=person_class_ids,
        device=inference_device,
        half=inference_half,
        workers=YOLO_FRAME_WORKERS,
        stream=False,
        save=True,
        project=str(OUTPUT_DIR),
        name=run_dir_name,
        exist_ok=False,
        verbose=False,
    )

    out_dir = OUTPUT_DIR / run_dir_name
    default_out = out_dir / src_path.name
    if default_out.exists():
        return f"/outputs/{run_dir_name}/{default_out.name}", None, []

    output_files = sorted(
        (p for p in out_dir.iterdir() if p.is_file()),
        key=lambda p: p.stat().st_mtime,
    )
    if not output_files:
        raise HTTPException(status_code=500, detail="Inference finished but no output was produced.")
    output_file = output_files[-1]

    return f"/outputs/{run_dir_name}/{output_file.name}", None, []


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    get_model()
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result_url": None,
            "media_kind": None,
            "error": None,
            "runtime": runtime_status(),
            "result_note": None,
            "clip_results": [],
            "processing_seconds": None,
        },
    )


@app.post("/detect", response_class=HTMLResponse)
async def detect(
    request: Request,
    file: UploadFile = File(...),
    conf: float = Form(0.25),
) -> HTMLResponse:
    processing_seconds: float | None = None
    try:
        media_kind = allowed_media(file.filename or "")
        input_ext = Path(file.filename or "").suffix.lower()
        upload_name = unique_name("upload", input_ext)
        upload_path = UPLOAD_DIR / upload_name

        with upload_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        inference_started = time.perf_counter()
        result_url, result_note, clip_results = run_inference(upload_path, conf=conf, media_kind=media_kind)
        processing_seconds = round(time.perf_counter() - inference_started, 2)
        if media_kind == "video" and upload_path.exists():
            upload_path.unlink()

        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "result_url": result_url,
                "media_kind": media_kind,
                "error": None,
                "runtime": runtime_status(),
                "result_note": result_note,
                "clip_results": clip_results,
                "processing_seconds": processing_seconds,
            },
        )
    except HTTPException as exc:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "result_url": None,
                "media_kind": None,
                "error": exc.detail,
                "runtime": runtime_status(),
                "result_note": None,
                "clip_results": [],
                "processing_seconds": processing_seconds,
            },
            status_code=exc.status_code,
        )
    except Exception as exc:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "result_url": None,
                "media_kind": None,
                "error": f"Unexpected error: {exc}",
                "runtime": runtime_status(),
                "result_note": None,
                "clip_results": [],
                "processing_seconds": processing_seconds,
            },
            status_code=500,
        )
    finally:
        if "media_kind" in locals() and media_kind == "video" and "upload_path" in locals() and upload_path.exists():
            upload_path.unlink()
