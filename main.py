from __future__ import annotations

import base64
import concurrent.futures
import hashlib
import json
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from io import BytesIO
from pathlib import Path
from urllib import error, request

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, render_template, request as flask_request, send_from_directory
from PIL import Image, ImageOps
from ultralytics import YOLO


ROOT_DIR = Path(__file__).resolve().parent
PERSON_DIR = ROOT_DIR / "person-detect"
RUNTIME_DIR = ROOT_DIR / "runtime"
UPLOAD_DIR = RUNTIME_DIR / "uploads"
OUTPUT_DIR = RUNTIME_DIR / "outputs"
CACHE_DIR = RUNTIME_DIR / "cache"
CLIPS_CACHE_DIR = CACHE_DIR / "video_clips"

PT_MODEL_PATH = PERSON_DIR / "yolo26s.pt"
ONNX_MODEL_PATH = PERSON_DIR / "yolo26s.onnx"

LLAMACPP_BASE_URL = os.getenv("LLAMACPP_BASE_URL", "http://127.0.0.1:8078")
LLAMACPP_MODEL = os.getenv("LLAMACPP_MODEL", "local-model")
REQUIRE_GPU = os.getenv("REQUIRE_GPU", "1").strip().lower() not in {"0", "false", "no"}
VIDEO_BATCH_SIZE = max(1, int(os.getenv("VIDEO_BATCH_SIZE", "32")))
MAX_DYNAMIC_MOSAICS = max(1, int(os.getenv("MAX_DYNAMIC_MOSAICS", "24")))
MOSAIC_SCALE_DIVISOR = max(1, int(os.getenv("MOSAIC_SCALE_DIVISOR", "8")))
LLM_MAX_BATCH_REQUESTS = max(1, int(os.getenv("LLM_MAX_BATCH_REQUESTS", "4")))

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CLIPS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)

model: YOLO | None = None
loaded_model_path: Path | None = None
inference_device: str = "cpu"
inference_half: bool = False
JOBS: dict[str, dict[str, object]] = {}
JOBS_LOCK = threading.Lock()


def unique_name(prefix: str, suffix: str) -> str:
    return f"{prefix}_{int(time.time() * 1000)}{suffix}"


def create_job(initial: dict[str, object]) -> str:
    job_id = uuid.uuid4().hex
    with JOBS_LOCK:
        JOBS[job_id] = initial
    return job_id


def update_job(job_id: str, **fields: object) -> None:
    with JOBS_LOCK:
        if job_id in JOBS:
            JOBS[job_id].update(fields)


def get_job(job_id: str) -> dict[str, object] | None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            return None
        return dict(job)


def allowed_media(filename: str, *, include_images: bool = True, include_videos: bool = True) -> str:
    ext = Path(filename).suffix.lower()
    if include_images and ext in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        return "image"
    if include_videos and ext in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
        return "video"
    raise ValueError("Unsupported file type.")


def select_model_and_device() -> tuple[Path, str, bool]:
    if torch.cuda.is_available():
        if PT_MODEL_PATH.exists():
            return PT_MODEL_PATH, "cuda:0", True
        if ONNX_MODEL_PATH.exists():
            return ONNX_MODEL_PATH, "cuda:0", False
        raise RuntimeError(f"No model found at {PT_MODEL_PATH} or {ONNX_MODEL_PATH}.")

    if REQUIRE_GPU:
        raise RuntimeError("GPU mode is required but CUDA is unavailable. Set REQUIRE_GPU=0 for CPU fallback.")

    if ONNX_MODEL_PATH.exists():
        return ONNX_MODEL_PATH, "cpu", False
    if PT_MODEL_PATH.exists():
        return PT_MODEL_PATH, "cpu", False
    raise RuntimeError(f"No model found at {PT_MODEL_PATH} or {ONNX_MODEL_PATH}.")


def get_model() -> YOLO:
    global model, loaded_model_path, inference_device, inference_half
    model_path, device, half = select_model_and_device()

    if model is None or loaded_model_path != model_path:
        model = YOLO(str(model_path))
        loaded_model_path = model_path

    inference_device = device
    inference_half = half and device.startswith("cuda")
    return model


def runtime_status() -> dict[str, str]:
    model_name = loaded_model_path.name if loaded_model_path else "not loaded yet"
    return {
        "model": model_name,
        "device": inference_device,
        "precision": "fp16" if inference_half else "fp32",
        "policy": "gpu-only" if REQUIRE_GPU else "auto-fallback",
        "video_batch_size": str(VIDEO_BATCH_SIZE),
        "llm_max_batch_requests": str(LLM_MAX_BATCH_REQUESTS),
    }


def get_person_class_ids(detector: YOLO) -> list[int]:
    names = detector.names
    person_ids: set[int] = set()
    name_items = names.items() if isinstance(names, dict) else enumerate(names)
    for class_id, class_name in name_items:
        if str(class_name).lower() == "person":
            person_ids.add(int(class_id))
    return sorted(person_ids) if person_ids else [0]


def run_cmd(cmd: list[str], err_prefix: str) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        stderr_tail = "\n".join(proc.stderr.splitlines()[-12:])
        raise RuntimeError(f"{err_prefix}\n{stderr_tail}")


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
        raise RuntimeError("Could not read video metadata.")

    data = json.loads(proc.stdout)
    streams = data.get("streams", [])
    if not streams:
        raise RuntimeError("No video stream found in uploaded file.")

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


def iter_video_frame_batches(src_path: Path, batch_size: int):
    capture = cv2.VideoCapture(str(src_path))
    if not capture.isOpened():
        raise RuntimeError("Could not open uploaded video for decoding.")

    next_frame_idx = 0
    batch: list[np.ndarray] = []
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
    progress_cb: callable | None = None,
) -> tuple[list[tuple[float, float]], int]:
    fps, duration = probe_video_metadata(src_path)
    capture = cv2.VideoCapture(str(src_path))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if capture.isOpened() else 0
    capture.release()

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
        if progress_cb is not None and total_frames > 0:
            scan_pct = min(75, max(5, int((processed_frames / total_frames) * 75)))
            progress_cb(scan_pct, "Scanning video for person detections")

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
    run_cmd(cmd, "Failed to cut person clip segment.")


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

    candidates = sorted((p for p in ann_root.iterdir() if p.is_file()), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise RuntimeError("Failed to produce annotated video.")
    return candidates[-1]


def run_video_clipper(
    src_path: Path,
    conf: float,
    progress_cb: callable | None = None,
) -> tuple[list[dict[str, object]], str]:
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
                if progress_cb is not None:
                    progress_cb(100, "Detection complete (cache hit)")
                return clips, "Cache hit: reused existing person clips."
        except Exception:
            pass

    clip_dir.mkdir(parents=True, exist_ok=True)
    if progress_cb is not None:
        progress_cb(3, "Preparing person detection")
    segments, processed_frames = detect_person_segments(
        src_path,
        detector,
        person_class_ids,
        conf,
        progress_cb=progress_cb,
    )
    if not segments:
        raise RuntimeError("No person detected in this video.")

    if progress_cb is not None:
        progress_cb(85, "Rendering annotated detection video")
    annotated_video = create_annotated_source_video(src_path, clip_dir, detector, person_class_ids, conf)
    clips: list[dict[str, object]] = []
    for idx, (start_s, end_s) in enumerate(segments):
        clip_name = f"clip_{idx:03d}.mp4"
        segment_path = clip_dir / clip_name
        if not segment_path.exists():
            cut_video_segment(annotated_video, start_s, end_s, segment_path)
        clips.append(
            {
                "index": idx + 1,
                "url": f"/files/cache/video_clips/{cache_key}/{clip_name}",
                "start_s": round(start_s, 3),
                "end_s": round(end_s, 3),
                "duration_s": round(end_s - start_s, 3),
                "start_label": format_timestamp(start_s),
                "end_label": format_timestamp(end_s),
            }
        )
        if progress_cb is not None:
            clip_pct = 90 + int(((idx + 1) / max(len(segments), 1)) * 10)
            progress_cb(min(100, clip_pct), f"Cutting person clips ({idx + 1}/{len(segments)})")

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

    if progress_cb is not None:
        progress_cb(100, "Detection complete")
    return clips, f"Created {len(clips)} person clip(s) with bounding boxes and cached them."


def run_person_detection(
    src_path: Path,
    conf: float,
    media_kind: str,
    progress_cb: callable | None = None,
) -> tuple[str | None, str | None, list[dict[str, object]]]:
    if media_kind == "video":
        clips, note = run_video_clipper(src_path, conf, progress_cb=progress_cb)
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
        return f"/files/outputs/{run_dir_name}/{default_out.name}", None, []

    output_files = sorted((p for p in out_dir.iterdir() if p.is_file()), key=lambda p: p.stat().st_mtime)
    if not output_files:
        raise RuntimeError("Inference finished but no output was produced.")
    output_file = output_files[-1]
    return f"/files/outputs/{run_dir_name}/{output_file.name}", None, []


def sample_frames(video_path: str, sample_count: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video.")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        raise RuntimeError("Could not read video frame count.")

    sample_count = max(1, sample_count)
    indices = np.linspace(0, max(frame_count - 1, 0), num=sample_count, dtype=int)

    frames: list[np.ndarray] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame_bgr = cap.read()
        if not ok:
            continue
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    cap.release()
    if not frames:
        raise RuntimeError("No frames could be extracted.")
    return frames


def make_mosaic(frames: list[np.ndarray], n: int, cell_size: int = 224) -> Image.Image:
    if not frames:
        raise RuntimeError("Cannot build a mosaic with zero frames.")
    n = max(1, n)
    cells = n * n
    canvas = Image.new("RGB", (n * cell_size, n * cell_size), color=(0, 0, 0))
    for i in range(cells):
        frame = Image.fromarray(frames[i % len(frames)])
        fitted = ImageOps.fit(frame, (cell_size, cell_size), method=Image.Resampling.LANCZOS)
        r, c = divmod(i, n)
        canvas.paste(fitted, (c * cell_size, r * cell_size))
    return canvas


def build_temporal_mosaics(video_path: str, n: int, t: int, cell_size: int = 224) -> list[Image.Image]:
    n = max(1, int(n))
    t = max(1, int(t))
    cells = n * n
    sampled = sample_frames(video_path, cells * t)
    sampled_count = len(sampled)
    if sampled_count == 0:
        raise RuntimeError("No frames were sampled from the video.")
    mosaics: list[Image.Image] = []
    for offset in range(t):
        # Interleave across the full sampled timeline so each mosaic spans
        # the entire video while staying temporally staggered from others.
        frame_group = sampled[offset::t]
        if not frame_group:
            frame_group = [sampled[offset % sampled_count]]
        frame_group = frame_group[:cells]
        mosaics.append(make_mosaic(frame_group, n=n, cell_size=cell_size))
    return mosaics


def collect_person_frames_for_mosaic(
    video_path: Path,
    conf: float,
    progress_cb: callable | None = None,
) -> list[np.ndarray]:
    detector = get_model()
    person_class_ids = get_person_class_ids(detector)
    selected: list[np.ndarray] = []
    capture = cv2.VideoCapture(str(video_path))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if capture.isOpened() else 0
    capture.release()
    processed_frames = 0

    for batch_start_idx, frame_batch in iter_video_frame_batches(video_path, batch_size=VIDEO_BATCH_SIZE):
        batch_results = detector.predict(
            source=frame_batch,
            conf=conf,
            classes=person_class_ids,
            device=inference_device,
            half=inference_half,
            stream=False,
            save=False,
            verbose=False,
        )
        for idx_in_batch, result in enumerate(batch_results):
            has_person = result.boxes is not None and len(result.boxes) > 0
            if not has_person:
                continue
            frame_rgb = cv2.cvtColor(frame_batch[idx_in_batch], cv2.COLOR_BGR2RGB)
            selected.append(frame_rgb)
        processed_frames = batch_start_idx + len(frame_batch)
        if progress_cb is not None and total_frames > 0:
            scan_pct = min(45, max(5, int((processed_frames / total_frames) * 45)))
            progress_cb(scan_pct, "Collecting person-detected frames")

    return selected


def build_temporal_mosaics_from_frames(person_frames: list[np.ndarray], n: int, t: int, cell_size: int = 224) -> list[Image.Image]:
    if not person_frames:
        raise RuntimeError("No person-detected frames available for mosaics.")
    n = max(1, int(n))
    t = max(1, int(t))
    cells = n * n
    mosaics: list[Image.Image] = []
    frame_count = len(person_frames)
    for offset in range(t):
        # Interleave detected frames to produce staggered mosaics.
        frame_group = person_frames[offset::t]
        if not frame_group:
            frame_group = [person_frames[offset % frame_count]]
        frame_group = frame_group[:cells]
        mosaics.append(make_mosaic(frame_group, n=n, cell_size=cell_size))
    return mosaics


def pil_to_data_url(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def extract_message_content(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text)
                    continue
                alt_text = item.get("content")
                if isinstance(alt_text, str) and alt_text.strip():
                    parts.append(alt_text)
        if parts:
            return "\n".join(parts)
    return str(content)


def parse_llamacpp_caption(response_body: str) -> str:
    data = json.loads(response_body)
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(f"Unexpected llama.cpp response: {response_body[:400]}")

    first = choices[0]
    if not isinstance(first, dict):
        return str(first)

    message = first.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        text = extract_message_content(content).strip()
        if text:
            return text

        reasoning = message.get("reasoning_content")
        if isinstance(reasoning, str) and reasoning.strip():
            return reasoning.strip()

    text = first.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip()

    delta = first.get("delta")
    if isinstance(delta, dict):
        d_content = delta.get("content")
        d_text = extract_message_content(d_content).strip()
        if d_text:
            return d_text

    raise RuntimeError(f"llama.cpp returned no assistant text. Raw excerpt: {response_body[:400]}")


def _build_image_content(images: list[Image.Image]) -> list[dict[str, object]]:
    return [{"type": "image_url", "image_url": {"url": pil_to_data_url(img)}} for img in images]


def query_llamacpp_with_images(
    images: list[Image.Image],
    prompt: str,
    max_tokens: int = 192,
    system_prompt: str | None = None,
) -> str:
    if not images:
        raise RuntimeError("At least one image is required for llama.cpp query.")

    messages: list[dict[str, object]] = []
    if system_prompt and system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt})
    content: list[dict[str, object]] = [{"type": "text", "text": prompt}]
    content.extend(_build_image_content(images))
    messages.append(
        {
            "role": "user",
            "content": content,
        }
    )

    payload = {
        "model": LLAMACPP_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    endpoint = f"{LLAMACPP_BASE_URL.rstrip('/')}/v1/chat/completions"
    req = request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=300) as resp:
            body = resp.read().decode("utf-8")
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"llama.cpp server error ({exc.code}): {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Could not reach llama.cpp at {endpoint}.") from exc

    return parse_llamacpp_caption(body)


def query_llamacpp(mosaic: Image.Image, prompt: str, max_tokens: int = 192, system_prompt: str | None = None) -> str:
    return query_llamacpp_with_images([mosaic], prompt, max_tokens=max_tokens, system_prompt=system_prompt)


def summarize_mosaic_answers(first_mosaic: Image.Image | None, mosaic_answers: str) -> str:
    if first_mosaic is None:
        raise RuntimeError("No first mosaic available.")
    if not mosaic_answers.strip():
        raise RuntimeError("No mosaic answers available.")

    system_prompt = (
        "You are a cautious surveillance analyst summarizing per-mosaic findings from a video. "
        "Prioritize potential public-safety and property-crime signals, including theft, arson, "
        "vandalism, trespassing, assault, and weapon-like behavior. "
        "The provided image is only the first temporal mosaic and may miss information "
        "that appears in the mosaic answers. Prefer the full mosaic answers when conflicts appear. "
        "If confidence is low, state uncertainty briefly instead of inventing details."
    )
    user_prompt = (
        "Create one final understanding of the full video using ONLY:\n"
        "1) the first mosaic image\n"
        "2) the mosaic answers text below\n\n"
        f"Mosaic answers:\n{mosaic_answers}\n\n"
        "Focus on suspicious behavior and threat-relevant context when present. "
        "Return only one short paragraph."
    )
    return query_llamacpp(first_mosaic, user_prompt, max_tokens=384, system_prompt=system_prompt)


def parse_threat_assessment(raw_text: str) -> tuple[int, str]:
    score = 0
    assessment = raw_text.strip()
    try:
        parsed = json.loads(raw_text)
        if isinstance(parsed, dict):
            maybe_score = parsed.get("threat_score")
            if isinstance(maybe_score, (int, float)):
                score = int(max(0, min(100, round(float(maybe_score)))))
            maybe_assessment = parsed.get("assessment")
            if isinstance(maybe_assessment, str) and maybe_assessment.strip():
                assessment = maybe_assessment.strip()
    except Exception:
        pass

    if score == 0:
        match = re.search(r"\b(100|[1-9]?\d)\b", raw_text)
        if match:
            score = int(match.group(1))

    if not assessment:
        assessment = "Threat assessment unavailable."
    return score, assessment


def generate_threat_assessment(
    first_mosaic: Image.Image | None,
    second_mosaic: Image.Image | None,
    first_caption: str,
    second_caption: str,
    final_summary: str,
) -> tuple[int, str]:
    if first_mosaic is None:
        raise RuntimeError("No first mosaic available for threat assessment.")

    images = [first_mosaic]
    if second_mosaic is not None:
        images.append(second_mosaic)

    system_prompt = (
        "You are a vigilant video security analyst. Output only strict JSON with keys "
        "`threat_score` (integer 0-100) and `assessment` (one short sentence)."
    )
    user_prompt = (
        "Create a threat assessment out of 100 using these inputs:\n"
        f"- First mosaic understanding: {first_caption or 'N/A'}\n"
        f"- Second mosaic understanding: {second_caption or 'N/A'}\n"
        f"- Final video understanding: {final_summary or 'N/A'}\n\n"
        "Explicitly check for indicators of theft/shoplifting, arson/fire-setting, vandalism/property damage, "
        "trespassing, assault, and weapon-related threats.\n"
        "Scoring guide:\n"
        "- 0 to 20: clearly benign routine activity\n"
        "- 21 to 50: suspicious behavior or possible pre-incident indicators\n"
        "- 51 to 80: likely criminal or dangerous behavior (e.g., theft, vandalism, attempted arson)\n"
        "- 81 to 100: active high-risk threat (e.g., confirmed arson attempt, violent assault, weapon threat)\n\n"
        "When uncertain between two ranges, choose the higher range if suspicious indicators are present.\n"
        "Return strict JSON only."
    )
    raw = query_llamacpp_with_images(images, user_prompt, max_tokens=160, system_prompt=system_prompt)
    return parse_threat_assessment(raw)


def run_video_understanding(
    video_path: Path,
    n: int,
    prompt: str,
    conf: float,
    llm_max_batch_requests: int,
    progress_cb: callable | None = None,
) -> tuple[list[dict[str, str]], str, str, str, int | None, str | None]:
    n = max(1, int(n))
    if progress_cb is not None:
        progress_cb(2, "Starting video understanding")
    person_frames = collect_person_frames_for_mosaic(video_path, conf=conf, progress_cb=progress_cb)
    if not person_frames:
        raise RuntimeError("No YOLO person-detected frames were found for video understanding.")
    frames_per_mosaic = n * n
    scaled_frames_per_mosaic = frames_per_mosaic * MOSAIC_SCALE_DIVISOR
    dynamic_t = max(1, (len(person_frames) + scaled_frames_per_mosaic - 1) // scaled_frames_per_mosaic)
    t = min(dynamic_t, MAX_DYNAMIC_MOSAICS)
    mosaics = build_temporal_mosaics(str(video_path), n=n, t=t)
    if progress_cb is not None:
        progress_cb(55, "Built temporal staggered mosaics")

    user_prompt = (
        prompt.strip()
        or "Analyze this surveillance video for suspicious activity and potential threats, "
        "including theft, arson, vandalism, trespassing, assault, and weapon-related behavior."
    )
    max_workers = max(1, int(llm_max_batch_requests))
    responses: list[str] = [""] * len(mosaics)
    response_texts: list[str] = [""] * len(mosaics)
    total_mosaics = len(mosaics)
    completed = 0

    def caption_mosaic(idx: int, mosaic: Image.Image) -> tuple[int, str]:
        constrained_prompt = (
            f"{user_prompt}\n\n"
            f"You are viewing temporal mosaic {idx} of {total_mosaics} from one video. "
            "Prioritize suspicious actions, threat indicators, and victim/property risk. "
            "Return only one short sentence. Do not include reasoning."
        )
        try:
            return idx, query_llamacpp(mosaic, constrained_prompt, max_tokens=256)
        except Exception as exc:
            return idx, f"Model inference failed for mosaic {idx}: {exc}"

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(caption_mosaic, i, mosaic) for i, mosaic in enumerate(mosaics, start=1)]
        for future in concurrent.futures.as_completed(futures):
            idx, caption = future.result()
            response_texts[idx - 1] = caption
            responses[idx - 1] = f"Mosaic {idx}: {caption}"
            completed += 1
            if progress_cb is not None:
                llm_pct = 55 + int((completed / max(total_mosaics, 1)) * 40)
                progress_cb(min(95, llm_pct), f"Understanding mosaics ({completed}/{total_mosaics})")

    all_captions = "\n".join(responses)
    try:
        final_summary = summarize_mosaic_answers(mosaics[0] if mosaics else None, all_captions)
    except Exception as exc:
        final_summary = f"Final summary failed: {exc}"
    threat_score: int | None = None
    threat_assessment: str | None = None
    try:
        first_caption = response_texts[0] if len(response_texts) > 0 else ""
        second_caption = response_texts[1] if len(response_texts) > 1 else ""
        second_mosaic = mosaics[1] if len(mosaics) > 1 else None
        threat_score, threat_assessment = generate_threat_assessment(
            first_mosaic=mosaics[0] if mosaics else None,
            second_mosaic=second_mosaic,
            first_caption=first_caption,
            second_caption=second_caption,
            final_summary=final_summary,
        )
    except Exception as exc:
        threat_assessment = f"Threat assessment failed: {exc}"
    if progress_cb is not None:
        progress_cb(100, "Video understanding complete")

    gallery = [
        {"label": f"Mosaic {idx + 1}/{len(mosaics)}", "src": pil_to_data_url(img)} for idx, img in enumerate(mosaics)
    ]
    stats = (
        f"Detected person-frames: {len(person_frames)} | "
        f"Mosaic size: {n}x{n} ({frames_per_mosaic} frames/mosaic) | "
        f"Mosaics generated: {t} (scale divisor: {MOSAIC_SCALE_DIVISOR}) | "
        f"LLM max batch requests: {max_workers}"
    )
    return gallery, all_captions, final_summary, stats, threat_score, threat_assessment


def run_full_analysis(
    upload_path: Path,
    conf: float,
    n: int,
    prompt: str,
    llm_max_batch_requests: int,
    source_video_url: str,
    job_id: str | None = None,
) -> dict[str, object]:
    started = time.perf_counter()

    def detect_progress(pct: int, message: str) -> None:
        if job_id is not None:
            update_job(
                job_id,
                stage="detection",
                stage_message=message,
                detect_progress=max(0, min(100, int(pct))),
            )

    def understanding_progress(pct: int, message: str) -> None:
        if job_id is not None:
            update_job(
                job_id,
                stage="understanding",
                stage_message=message,
                understanding_progress=max(0, min(100, int(pct))),
            )

    detect_error: str | None = None
    detect_processing_seconds: float | None = None
    result_note: str | None = None
    clip_results: list[dict[str, object]] = []
    try:
        detect_started = time.perf_counter()
        _result_url, result_note, clip_results = run_person_detection(
            upload_path, conf, "video", progress_cb=detect_progress
        )
        detect_processing_seconds = round(time.perf_counter() - detect_started, 2)
    except Exception as exc:
        detect_error = str(exc)
        if job_id is not None:
            update_job(job_id, detect_progress=100)

    understanding_error: str | None = None
    understanding_processing_seconds: float | None = None
    gallery: list[dict[str, str]] = []
    captions: str | None = None
    summary: str | None = None
    understanding_stats: str | None = None
    threat_score: int | None = None
    threat_assessment: str | None = None
    try:
        understanding_started = time.perf_counter()
        gallery, captions, summary, understanding_stats, threat_score, threat_assessment = run_video_understanding(
            upload_path,
            n=n,
            prompt=prompt,
            conf=conf,
            llm_max_batch_requests=llm_max_batch_requests,
            progress_cb=understanding_progress,
        )
        understanding_processing_seconds = round(time.perf_counter() - understanding_started, 2)
    except Exception as exc:
        understanding_error = str(exc)
        if job_id is not None:
            update_job(job_id, understanding_progress=100)

    overall_processing_seconds = round(time.perf_counter() - started, 2)
    overall_error = None
    if detect_error and understanding_error:
        overall_error = "Both analysis stages failed."

    return {
        "overall_error": overall_error,
        "source_video_url": source_video_url,
        "detect_result_note": result_note,
        "detect_error": detect_error,
        "detect_processing_seconds": detect_processing_seconds,
        "clip_results": clip_results,
        "understanding_error": understanding_error,
        "understanding_gallery": gallery,
        "understanding_captions": captions,
        "understanding_summary": summary,
        "understanding_stats": understanding_stats,
        "threat_score": threat_score,
        "threat_assessment": threat_assessment,
        "understanding_processing_seconds": understanding_processing_seconds,
        "overall_processing_seconds": overall_processing_seconds,
    }


def render_page(**overrides: object):
    base = {
        "runtime": runtime_status(),
        "overall_error": None,
        "overall_processing_seconds": None,
        "source_video_url": None,
        "detect_error": None,
        "detect_result_note": None,
        "detect_processing_seconds": None,
        "clip_results": [],
        "understanding_error": None,
        "understanding_gallery": [],
        "understanding_captions": None,
        "understanding_summary": None,
        "understanding_stats": None,
        "threat_score": None,
        "threat_assessment": None,
        "understanding_processing_seconds": None,
        "llamacpp_endpoint": LLAMACPP_BASE_URL,
    }
    base.update(overrides)
    return render_template("index.html", **base)


@app.get("/")
def index():
    try:
        get_model()
    except Exception:
        pass
    return render_page()


@app.post("/analyze")
def analyze_route():
    upload = flask_request.files.get("video")
    if upload is None or not upload.filename:
        return render_page(overall_error="Please choose a video file."), 400

    upload_path: Path | None = None
    try:
        allowed_media(upload.filename, include_images=False, include_videos=True)
        conf = float(flask_request.form.get("conf", "0.25"))
        if conf < 0.01 or conf > 0.99:
            raise ValueError("Confidence must be between 0.01 and 0.99.")
        n = int(flask_request.form.get("n", "4"))
        llm_max_batch_requests = int(flask_request.form.get("llm_max_batch_requests", str(LLM_MAX_BATCH_REQUESTS)))
        if llm_max_batch_requests < 1:
            raise ValueError("LLM max batch requests must be at least 1.")
        prompt = flask_request.form.get(
            "prompt",
            "Analyze this surveillance video for suspicious activity and potential threats, including theft, arson, vandalism, trespassing, assault, and weapon-related behavior.",
        )

        ext = Path(upload.filename).suffix.lower()
        upload_path = UPLOAD_DIR / unique_name("video", ext)
        upload.save(upload_path)
        display_name = unique_name("source", ext)
        display_path = OUTPUT_DIR / display_name
        shutil.copy2(upload_path, display_path)
        source_video_url = f"/files/outputs/{display_name}"
        context = run_full_analysis(
            upload_path,
            conf=conf,
            n=n,
            prompt=prompt,
            llm_max_batch_requests=llm_max_batch_requests,
            source_video_url=source_video_url,
        )
        return render_page(**context)
    except Exception as exc:
        return render_page(overall_error=str(exc)), 400
    finally:
        if upload_path is not None and upload_path.exists():
            upload_path.unlink()


def process_analysis_job(
    job_id: str,
    upload_path: Path,
    conf: float,
    n: int,
    prompt: str,
    llm_max_batch_requests: int,
    source_video_url: str,
) -> None:
    try:
        update_job(
            job_id,
            state="processing",
            stage="detection",
            stage_message="Starting person detection",
            detect_progress=0,
            understanding_progress=0,
        )
        context = run_full_analysis(
            upload_path=upload_path,
            conf=conf,
            n=n,
            prompt=prompt,
            llm_max_batch_requests=llm_max_batch_requests,
            source_video_url=source_video_url,
            job_id=job_id,
        )
        update_job(
            job_id,
            state="done",
            stage="complete",
            stage_message="Analysis complete",
            detect_progress=100,
            understanding_progress=100,
            result_context=context,
            error=None,
        )
    except Exception as exc:
        update_job(
            job_id,
            state="error",
            stage="error",
            stage_message=str(exc),
            error=str(exc),
            detect_progress=100,
            understanding_progress=100,
        )
    finally:
        if upload_path.exists():
            upload_path.unlink()


@app.post("/analyze/start")
def analyze_start():
    upload = flask_request.files.get("video")
    if upload is None or not upload.filename:
        return jsonify({"error": "Please choose a video file."}), 400

    upload_path: Path | None = None
    try:
        allowed_media(upload.filename, include_images=False, include_videos=True)
        conf = float(flask_request.form.get("conf", "0.25"))
        if conf < 0.01 or conf > 0.99:
            raise ValueError("Confidence must be between 0.01 and 0.99.")
        n = int(flask_request.form.get("n", "4"))
        llm_max_batch_requests = int(flask_request.form.get("llm_max_batch_requests", str(LLM_MAX_BATCH_REQUESTS)))
        if llm_max_batch_requests < 1:
            raise ValueError("LLM max batch requests must be at least 1.")
        prompt = flask_request.form.get(
            "prompt",
            "Analyze this surveillance video for suspicious activity and potential threats, including theft, arson, vandalism, trespassing, assault, and weapon-related behavior.",
        )

        ext = Path(upload.filename).suffix.lower()
        upload_path = UPLOAD_DIR / unique_name("video", ext)
        upload.save(upload_path)
        display_name = unique_name("source", ext)
        display_path = OUTPUT_DIR / display_name
        shutil.copy2(upload_path, display_path)
        source_video_url = f"/files/outputs/{display_name}"

        job_id = create_job(
            {
                "state": "queued",
                "stage": "queued",
                "stage_message": "Queued",
                "detect_progress": 0,
                "understanding_progress": 0,
                "result_context": None,
                "error": None,
                "created_at_ms": int(time.time() * 1000),
            }
        )

        worker = threading.Thread(
            target=process_analysis_job,
            args=(job_id, upload_path, conf, n, prompt, llm_max_batch_requests, source_video_url),
            daemon=True,
        )
        worker.start()
        return jsonify({"job_id": job_id})
    except Exception as exc:
        if upload_path is not None and upload_path.exists():
            upload_path.unlink()
        return jsonify({"error": str(exc)}), 400


@app.get("/analyze/status/<job_id>")
def analyze_status(job_id: str):
    job = get_job(job_id)
    if job is None:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(
        {
            "state": job.get("state", "queued"),
            "stage": job.get("stage", "queued"),
            "stage_message": job.get("stage_message", ""),
            "detect_progress": job.get("detect_progress", 0),
            "understanding_progress": job.get("understanding_progress", 0),
            "error": job.get("error"),
        }
    )


@app.get("/analyze/result/<job_id>")
def analyze_result(job_id: str):
    job = get_job(job_id)
    if job is None:
        return render_page(overall_error="Job not found."), 404
    state = str(job.get("state", "queued"))
    if state not in {"done", "error"}:
        return render_page(overall_error="Analysis is still running."), 409

    result_context = job.get("result_context")
    if isinstance(result_context, dict):
        return render_page(**result_context)
    return render_page(overall_error=str(job.get("error", "Analysis failed."))), 500


@app.get("/files/<path:filename>")
def files(filename: str):
    return send_from_directory(RUNTIME_DIR, filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
