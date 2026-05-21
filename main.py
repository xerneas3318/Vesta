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
from datetime import datetime
from io import BytesIO
from pathlib import Path
from urllib import error, request
from urllib.parse import quote, unquote, urlsplit, urlunsplit

import cv2
import numpy as np
import torch
from flask import Flask, Response, jsonify, render_template, request as flask_request, send_from_directory
from PIL import Image
from ultralytics import YOLO


ROOT_DIR = Path(__file__).resolve().parent
PERSON_DIR = ROOT_DIR / "person-detect"
RUNTIME_DIR = ROOT_DIR / "runtime"
UPLOAD_DIR = RUNTIME_DIR / "uploads"
OUTPUT_DIR = RUNTIME_DIR / "outputs"
CACHE_DIR = RUNTIME_DIR / "cache"
CLIPS_CACHE_DIR = CACHE_DIR / "video_clips"
RECORDINGS_DIR = RUNTIME_DIR / "recordings"

PT_MODEL_PATH = PERSON_DIR / "yolo26s.pt"
ONNX_MODEL_PATH = PERSON_DIR / "yolo26s.onnx"

LLAMACPP_BASE_URL = os.getenv("LLAMACPP_BASE_URL", "http://127.0.0.1:8078")
LLAMACPP_MODEL = os.getenv("LLAMACPP_MODEL", "local-model")
LIVE_RTSP_DEFAULT = os.getenv(
    "LIVE_RTSP_URL",
    "rtsp://user:robotics3800@172.16.100.188:554/cam/realmonitor?channel=1&subtype=1",
)
LIVE_RTSP_DISCOVER_USER = os.getenv("LIVE_RTSP_DISCOVER_USER", "user")
LIVE_RTSP_DISCOVER_PASSWORD = os.getenv("LIVE_RTSP_DISCOVER_PASSWORD", "robotics3800")
LIVE_DETECT_CONF = min(0.99, max(0.01, float(os.getenv("LIVE_DETECT_CONF", "0.25"))))
LIVE_DISCOVERY_PROBE_TIMEOUT_S = max(2, int(os.getenv("LIVE_DISCOVERY_PROBE_TIMEOUT_S", "5")))
LIVE_DISCOVERY_MAX_CANDIDATES = max(1, int(os.getenv("LIVE_DISCOVERY_MAX_CANDIDATES", "24")))
LIVE_RTSP_READ_FAILS_BEFORE_RECONNECT = max(1, int(os.getenv("LIVE_RTSP_READ_FAILS_BEFORE_RECONNECT", "20")))
LIVE_RTSP_RECONNECT_BASE_S = max(0.2, float(os.getenv("LIVE_RTSP_RECONNECT_BASE_S", "1.5")))
LIVE_RTSP_RECONNECT_MAX_S = max(1.0, float(os.getenv("LIVE_RTSP_RECONNECT_MAX_S", "12.0")))
LIVE_YOLO_EVERY_N_FRAMES = max(1, int(os.getenv("LIVE_YOLO_EVERY_N_FRAMES", "8")))
LIVE_YOLO_IMGSZ = max(320, int(os.getenv("LIVE_YOLO_IMGSZ", "640")))
LIVE_MODEL_RETRY_S = max(1.0, float(os.getenv("LIVE_MODEL_RETRY_S", "4.0")))
REQUIRE_GPU = os.getenv("REQUIRE_GPU", "1").strip().lower() not in {
    "0", "false", "no"}
VIDEO_BATCH_SIZE = max(1, int(os.getenv("VIDEO_BATCH_SIZE", "32")))
YOLO_FRAME_WORKERS = max(1, int(os.getenv("YOLO_FRAME_WORKERS", "4")))
MAX_DYNAMIC_MOSAICS = max(1, int(os.getenv("MAX_DYNAMIC_MOSAICS", "24")))
MOSAIC_SCALE_DIVISOR = max(1, int(os.getenv("MOSAIC_SCALE_DIVISOR", "8")))
LLM_MAX_BATCH_REQUESTS = max(1, int(os.getenv("LLM_MAX_BATCH_REQUESTS", "16")))
AUTONOMOUS_TRIGGER_HITS = max(1, int(os.getenv("AUTONOMOUS_TRIGGER_HITS", "3")))
AUTONOMOUS_TRIGGER_MISSES = max(1, int(os.getenv("AUTONOMOUS_TRIGGER_MISSES", "12")))
AUTONOMOUS_MAX_CLIP_S = max(5.0, float(os.getenv("AUTONOMOUS_MAX_CLIP_S", "90.0")))
AUTONOMOUS_PROMPT = os.getenv(
    "AUTONOMOUS_PROMPT",
    "Analyze this surveillance clip for suspicious activity or potential threats (theft, arson, vandalism, trespassing, assault, weapons).",
)
AUTONOMOUS_MAX_EVENTS = max(20, int(os.getenv("AUTONOMOUS_MAX_EVENTS", "200")))
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;500000",
)

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CLIPS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)

model: YOLO | None = None
loaded_model_path: Path | None = None
inference_device: str = "cpu"
inference_half: bool = False
JOBS: dict[str, dict[str, object]] = {}
JOBS_LOCK = threading.Lock()
LIVE_LOCK = threading.Lock()
LIVE_THREAD: threading.Thread | None = None
RTSP_LAST_GOOD: dict[str, str] = {}
LIVE_STATE: dict[str, object] = {
    "running": False,
    "rtsp_url": LIVE_RTSP_DEFAULT,
    "last_jpeg": None,
    "error": None,
    "recording_active": False,
    "recording_writer": None,
    "recording_path": None,
    "recording_started_at_ms": None,
    "recording_kind": None,
    "recording_event_id": None,
    "discovered_streams": [],
}

AUTONOMOUS_STATE: dict[str, object] = {
    "enabled": False,
    "last_event_at_ms": None,
}
EVENTS_PATH = RUNTIME_DIR / "events.json"
EVENTS_LOCK = threading.Lock()
AUTONOMOUS_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="auto-analyze")

COMMON_RTSP_PATHS = [
    "",
    "/Streaming/Channels/101",
    "/Streaming/Channels/102",
    "/cam/realmonitor?channel=1&subtype=0",
    "/cam/realmonitor?channel=1&subtype=1",
    "/h264Preview_01_main",
    "/h264Preview_01_sub",
    "/live/ch00_0",
    "/live/ch00_1",
    "/stream1",
    "/stream2",
    "/11",
    "/12",
    "/axis-media/media.amp",
    "/profile1/media.smp",
    "/profile2/media.smp",
]


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


def list_recorded_videos() -> list[dict[str, object]]:
    videos: list[dict[str, object]] = []
    for path in sorted(RECORDINGS_DIR.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True):
        stat = path.stat()
        videos.append(
            {
                "name": path.name,
                "url": f"/files/recordings/{path.name}",
                "created_at_ms": int(stat.st_mtime * 1000),
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
            }
        )
    return videos


def load_events() -> list[dict[str, object]]:
    if not EVENTS_PATH.exists():
        return []
    try:
        with EVENTS_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        events = data.get("events", [])
        return list(events) if isinstance(events, list) else []
    except Exception:
        return []


def _save_events_unlocked(events: list[dict[str, object]]) -> None:
    trimmed = events[-AUTONOMOUS_MAX_EVENTS:]
    tmp = EVENTS_PATH.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump({"events": trimmed}, f, indent=2)
    tmp.replace(EVENTS_PATH)


def append_event(event: dict[str, object]) -> None:
    with EVENTS_LOCK:
        events = load_events()
        events.append(event)
        _save_events_unlocked(events)


def update_event(event_id: str, **fields: object) -> None:
    with EVENTS_LOCK:
        events = load_events()
        for ev in events:
            if ev.get("id") == event_id:
                ev.update(fields)
                break
        _save_events_unlocked(events)


def _autonomous_status_payload() -> dict[str, object]:
    with LIVE_LOCK:
        recording_kind = LIVE_STATE.get("recording_kind")
        recording_event_id = LIVE_STATE.get("recording_event_id")
    return {
        "enabled": bool(AUTONOMOUS_STATE.get("enabled", False)),
        "trigger_hits": AUTONOMOUS_TRIGGER_HITS,
        "trigger_misses": AUTONOMOUS_TRIGGER_MISSES,
        "max_clip_s": AUTONOMOUS_MAX_CLIP_S,
        "recording_kind": recording_kind,
        "recording_event_id": recording_event_id,
        "last_event_at_ms": AUTONOMOUS_STATE.get("last_event_at_ms"),
    }


def analyze_autonomous_clip(event_id: str, clip_path: Path) -> None:
    try:
        if not clip_path.exists():
            update_event(event_id, state="error", error="Clip file missing.")
            return
        try:
            if not is_browser_friendly_video(clip_path):
                transcode_to_browser_mp4(clip_path)
        except Exception:
            pass
        gallery, captions, summary, stats, threat_score, threat_assessment = run_video_understanding(
            clip_path,
            n=4,
            prompt=AUTONOMOUS_PROMPT,
            conf=LIVE_DETECT_CONF,
            llm_max_batch_requests=LLM_MAX_BATCH_REQUESTS,
            use_yolo_filter=False,
        )
        update_event(
            event_id,
            state="done",
            summary=summary,
            captions=captions,
            threat_score=threat_score,
            threat_assessment=threat_assessment,
            stats=stats,
            analyzed_at_ms=int(time.time() * 1000),
        )
    except Exception as exc:
        update_event(event_id, state="error", error=str(exc),
                     analyzed_at_ms=int(time.time() * 1000))


def _close_recording_writer_unlocked() -> None:
    writer = LIVE_STATE.get("recording_writer")
    if isinstance(writer, cv2.VideoWriter):
        writer.release()
    LIVE_STATE["recording_writer"] = None
    LIVE_STATE["recording_active"] = False
    LIVE_STATE["recording_started_at_ms"] = None


def _reset_live_unlocked(reset_rtsp: bool = False) -> None:
    LIVE_STATE["running"] = False
    LIVE_STATE["last_jpeg"] = None
    LIVE_STATE["error"] = None
    _close_recording_writer_unlocked()
    LIVE_STATE["recording_path"] = None
    LIVE_STATE["discovered_streams"] = []
    if reset_rtsp:
        LIVE_STATE["rtsp_url"] = LIVE_RTSP_DEFAULT


def _open_live_capture(rtsp_url: str) -> cv2.VideoCapture:
    attempts: list[tuple[str, cv2.VideoCapture]] = [
        ("ffmpeg", cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)),
        ("default", cv2.VideoCapture(rtsp_url)),
    ]
    for backend_name, capture in attempts:
        try:
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        if not capture.isOpened():
            capture.release()
            continue

        # Ensure we can actually read frames, not just open a socket.
        warmup_deadline = time.time() + 6.0
        while time.time() < warmup_deadline:
            ok, frame = capture.read()
            if ok and frame is not None and frame.size > 0:
                return capture
            time.sleep(0.05)
        capture.release()

    auth_hint = ""
    if "@" in rtsp_url:
        auth_hint = " Check credentials and encode special chars in password (e.g. @ -> %40)."
    raise RuntimeError(f"Could not open RTSP stream: {rtsp_url}.{auth_hint}")


def _assemble_rtsp_url(host: str, port: int, username: str | None, password: str | None, path: str) -> str:
    safe_host = host.strip()
    # Normalize first so already-encoded credentials (e.g. %40) do not get
    # repeatedly encoded into %2540 across reconnect attempts.
    safe_user = unquote((username or "").strip())
    safe_pass = unquote((password or "").strip())
    auth = ""
    if safe_user:
        auth = quote(safe_user, safe="")
        if safe_pass:
            auth += f":{quote(safe_pass, safe='')}"
        auth += "@"
    path_part = path or ""
    if path_part and not path_part.startswith("/"):
        path_part = f"/{path_part}"
    return f"rtsp://{auth}{safe_host}:{int(port)}{path_part}"


def _probe_rtsp_url(rtsp_url: str, timeout_s: int = 6) -> bool:
    cmd = [
        "ffprobe",
        "-rtsp_transport",
        "tcp",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=index",
        "-of",
        "json",
        rtsp_url,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        return proc.returncode == 0
    except Exception:
        return False


def _rtsp_cache_key(host: str, port: int, username: str | None) -> str:
    return f"{(username or '').strip().lower()}@{host.strip().lower()}:{int(port)}"


def _cache_rtsp_stream(rtsp_url: str) -> None:
    parsed = urlsplit(rtsp_url)
    if parsed.scheme != "rtsp" or not parsed.hostname:
        return
    key = _rtsp_cache_key(parsed.hostname, int(parsed.port or 554), parsed.username)
    RTSP_LAST_GOOD[key] = rtsp_url


def _dedupe_urls(urls: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for url in urls:
        if url in seen:
            continue
        seen.add(url)
        deduped.append(url)
    return deduped


def discover_rtsp_streams(
    host: str,
    port: int,
    username: str | None = None,
    password: str | None = None,
    max_results: int = 4,
    preferred_urls: list[str] | None = None,
) -> list[str]:
    probe_user = username if username is not None else LIVE_RTSP_DISCOVER_USER
    probe_pass = password if password is not None else LIVE_RTSP_DISCOVER_PASSWORD
    path_candidates = [
        _assemble_rtsp_url(host, port, probe_user, probe_pass, path)
        for path in COMMON_RTSP_PATHS
    ]
    candidates = _dedupe_urls((preferred_urls or []) + path_candidates)[:LIVE_DISCOVERY_MAX_CANDIDATES]
    found: list[str] = []
    for url in candidates:
        if _probe_rtsp_url(url, timeout_s=LIVE_DISCOVERY_PROBE_TIMEOUT_S):
            found.append(url)
            _cache_rtsp_stream(url)
            if len(found) >= max_results:
                break
        time.sleep(0.08)
    return found


def resolve_live_rtsp_url(raw_rtsp_url: str | None) -> tuple[str, list[str]]:
    target = (raw_rtsp_url or LIVE_RTSP_DEFAULT).strip()
    if not target:
        target = LIVE_RTSP_DEFAULT
    if not target.startswith("rtsp://"):
        target = f"rtsp://{target}"

    parsed = urlsplit(target)
    if parsed.scheme != "rtsp":
        raise ValueError("RTSP URL must start with rtsp://")
    if not parsed.hostname:
        raise ValueError("Missing RTSP host.")

    host = parsed.hostname
    port = int(parsed.port or 554)
    username = parsed.username or LIVE_RTSP_DISCOVER_USER
    password = parsed.password or LIVE_RTSP_DISCOVER_PASSWORD
    cache_key = _rtsp_cache_key(host, port, username)
    cached_url = RTSP_LAST_GOOD.get(cache_key)
    has_specific_path = bool(parsed.path and parsed.path not in {"", "/"}) or bool(parsed.query)

    normalized = _assemble_rtsp_url(
        host=host,
        port=port,
        username=username,
        password=password,
        path=f"{parsed.path or ''}{('?' + parsed.query) if parsed.query else ''}",
    )

    if has_specific_path:
        # For explicit stream URLs, skip ffprobe preflight to avoid double auth/connect
        # patterns that can trigger camera lockouts on strict firmware.
        return normalized, [normalized]

    preferred = [cached_url] if cached_url else []
    discovered = discover_rtsp_streams(
        host=host,
        port=port,
        username=username,
        password=password,
        preferred_urls=preferred,
    )
    if discovered:
        return discovered[0], discovered
    raise RuntimeError(
        "No RTSP stream path discovered automatically. Try a full URL path like "
        "rtsp://user:pass@host:554/Streaming/Channels/101"
    )


def _live_loop() -> None:
    capture: cv2.VideoCapture | None = None
    detector: YOLO | None = None
    person_class_ids: list[int] = [0]
    last_rtsp_url: str | None = None
    consecutive_read_failures = 0
    reconnect_sleep_s = LIVE_RTSP_RECONNECT_BASE_S
    frame_counter = 0
    last_annotated: np.ndarray | None = None
    next_model_retry_at = 0.0
    auto_hits = 0
    auto_misses = 0
    auto_event_id: str | None = None
    auto_event_started_at_ms: int | None = None
    try:
        while True:
            with LIVE_LOCK:
                running = bool(LIVE_STATE.get("running", False))
                rtsp_url = str(LIVE_STATE.get("rtsp_url", LIVE_RTSP_DEFAULT))
            if not running:
                break

            if capture is None or last_rtsp_url != rtsp_url:
                if capture is not None:
                    capture.release()
                try:
                    capture = _open_live_capture(rtsp_url)
                    _cache_rtsp_stream(rtsp_url)
                    last_rtsp_url = rtsp_url
                    consecutive_read_failures = 0
                    reconnect_sleep_s = LIVE_RTSP_RECONNECT_BASE_S
                    with LIVE_LOCK:
                        LIVE_STATE["error"] = None
                except Exception as exc:
                    with LIVE_LOCK:
                        LIVE_STATE["error"] = f"{exc} | reconnecting in {reconnect_sleep_s:.1f}s"
                    time.sleep(reconnect_sleep_s)
                    reconnect_sleep_s = min(LIVE_RTSP_RECONNECT_MAX_S, reconnect_sleep_s * 1.6)
                    continue

            ok, frame = capture.read()
            if not ok or frame is None:
                consecutive_read_failures += 1
                if consecutive_read_failures < LIVE_RTSP_READ_FAILS_BEFORE_RECONNECT:
                    with LIVE_LOCK:
                        LIVE_STATE["error"] = (
                            f"RTSP read failed ({consecutive_read_failures}/"
                            f"{LIVE_RTSP_READ_FAILS_BEFORE_RECONNECT}); waiting before reconnect"
                        )
                    time.sleep(0.08)
                    continue
                with LIVE_LOCK:
                    LIVE_STATE["error"] = f"RTSP unstable; reconnecting in {reconnect_sleep_s:.1f}s"
                capture.release()
                capture = None
                consecutive_read_failures = 0
                time.sleep(reconnect_sleep_s)
                reconnect_sleep_s = min(LIVE_RTSP_RECONNECT_MAX_S, reconnect_sleep_s * 1.4)
                continue
            consecutive_read_failures = 0
            reconnect_sleep_s = LIVE_RTSP_RECONNECT_BASE_S
            frame_counter += 1

            run_yolo = (frame_counter % LIVE_YOLO_EVERY_N_FRAMES) == 0
            annotated = frame
            if run_yolo:
                now = time.time()
                if detector is None and now >= next_model_retry_at:
                    try:
                        detector = get_model()
                        person_class_ids = get_person_class_ids(detector)
                        with LIVE_LOCK:
                            LIVE_STATE["error"] = None
                    except Exception as exc:
                        next_model_retry_at = now + LIVE_MODEL_RETRY_S
                        with LIVE_LOCK:
                            LIVE_STATE["error"] = f"Model load retrying in {LIVE_MODEL_RETRY_S:.0f}s: {exc}"
                person_present_this_yolo: bool | None = None
                if detector is not None:
                    try:
                        prediction = safe_yolo_predict(
                            detector,
                            source=frame,
                            conf=LIVE_DETECT_CONF,
                            classes=person_class_ids,
                            device=inference_device,
                            half=inference_half,
                            stream=False,
                            save=False,
                            verbose=False,
                            imgsz=LIVE_YOLO_IMGSZ,
                        )
                        annotated = prediction[0].plot() if prediction else frame
                        last_annotated = annotated
                        if prediction:
                            boxes = getattr(prediction[0], "boxes", None)
                            person_present_this_yolo = bool(boxes is not None and len(boxes) > 0)
                    except Exception as exc:
                        annotated = last_annotated if last_annotated is not None else frame
                        with LIVE_LOCK:
                            LIVE_STATE["error"] = f"Live YOLO failed: {exc}"
                elif last_annotated is not None:
                    annotated = last_annotated

                # ---------- autonomous trigger ----------
                if person_present_this_yolo is not None:
                    with LIVE_LOCK:
                        autonomous_enabled = bool(AUTONOMOUS_STATE.get("enabled", False))
                        recording_kind = LIVE_STATE.get("recording_kind")
                    if autonomous_enabled and recording_kind in (None, "auto"):
                        if person_present_this_yolo:
                            auto_hits += 1
                            auto_misses = 0
                        else:
                            auto_misses += 1
                            auto_hits = 0

                        if auto_event_id is None and auto_hits >= AUTONOMOUS_TRIGGER_HITS:
                            now_ms = int(time.time() * 1000)
                            filename = f"auto_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.mp4"
                            candidate = RECORDINGS_DIR / filename
                            if candidate.exists():
                                filename = f"auto_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.mp4"
                                candidate = RECORDINGS_DIR / filename
                            event_id = "evt_" + uuid.uuid4().hex[:10]
                            with LIVE_LOCK:
                                LIVE_STATE["recording_path"] = candidate
                                LIVE_STATE["recording_active"] = True
                                LIVE_STATE["recording_kind"] = "auto"
                                LIVE_STATE["recording_event_id"] = event_id
                                LIVE_STATE["recording_started_at_ms"] = now_ms
                                AUTONOMOUS_STATE["last_event_at_ms"] = now_ms
                            append_event({
                                "id": event_id,
                                "started_at_ms": now_ms,
                                "ended_at_ms": None,
                                "clip_filename": filename,
                                "clip_url": f"/files/recordings/{filename}",
                                "state": "recording",
                                "threat_score": None,
                                "summary": None,
                                "trigger": f"yolo_dwell hits={AUTONOMOUS_TRIGGER_HITS}",
                            })
                            auto_event_id = event_id
                            auto_event_started_at_ms = now_ms
                            auto_hits = 0
                            auto_misses = 0

                        elif auto_event_id is not None:
                            now_ms = int(time.time() * 1000)
                            clip_age_s = (now_ms - (auto_event_started_at_ms or now_ms)) / 1000.0
                            stop_for_misses = auto_misses >= AUTONOMOUS_TRIGGER_MISSES
                            stop_for_cap = clip_age_s >= AUTONOMOUS_MAX_CLIP_S
                            if stop_for_misses or stop_for_cap:
                                ending_event_id = auto_event_id
                                with LIVE_LOCK:
                                    recording_path_to_analyze = LIVE_STATE.get("recording_path")
                                    _close_recording_writer_unlocked()
                                    LIVE_STATE["recording_path"] = None
                                    LIVE_STATE["recording_kind"] = None
                                    LIVE_STATE["recording_event_id"] = None
                                update_event(
                                    ending_event_id,
                                    state="analyzing",
                                    ended_at_ms=now_ms,
                                    duration_s=round(clip_age_s, 2),
                                    stop_reason="cap" if stop_for_cap else "dwell-off",
                                )
                                if isinstance(recording_path_to_analyze, Path):
                                    AUTONOMOUS_EXECUTOR.submit(
                                        analyze_autonomous_clip,
                                        ending_event_id,
                                        recording_path_to_analyze,
                                    )
                                auto_event_id = None
                                auto_event_started_at_ms = None
                                auto_hits = 0
                                auto_misses = 0
                    else:
                        auto_hits = 0
                        auto_misses = 0
                        if auto_event_id is not None:
                            now_ms = int(time.time() * 1000)
                            ending_event_id = auto_event_id
                            with LIVE_LOCK:
                                recording_path_to_analyze = LIVE_STATE.get("recording_path")
                                _close_recording_writer_unlocked()
                                LIVE_STATE["recording_path"] = None
                                LIVE_STATE["recording_kind"] = None
                                LIVE_STATE["recording_event_id"] = None
                            update_event(
                                ending_event_id,
                                state="analyzing",
                                ended_at_ms=now_ms,
                                duration_s=round((now_ms - (auto_event_started_at_ms or now_ms)) / 1000.0, 2),
                                stop_reason="autonomous-disabled",
                            )
                            if isinstance(recording_path_to_analyze, Path):
                                AUTONOMOUS_EXECUTOR.submit(
                                    analyze_autonomous_clip,
                                    ending_event_id,
                                    recording_path_to_analyze,
                                )
                            auto_event_id = None
                            auto_event_started_at_ms = None
            elif last_annotated is not None:
                annotated = last_annotated

            with LIVE_LOCK:
                recording_active = bool(LIVE_STATE.get("recording_active", False))
                recording_writer = LIVE_STATE.get("recording_writer")
                recording_path = LIVE_STATE.get("recording_path")

            if recording_active and isinstance(recording_path, Path):
                if not isinstance(recording_writer, cv2.VideoWriter):
                    fps = capture.get(cv2.CAP_PROP_FPS)
                    if not isinstance(fps, float) or fps <= 1:
                        fps = 20.0
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(str(recording_path), fourcc, fps, (frame.shape[1], frame.shape[0]))
                    with LIVE_LOCK:
                        LIVE_STATE["recording_writer"] = writer
                        recording_writer = writer
                if isinstance(recording_writer, cv2.VideoWriter):
                    recording_writer.write(frame)

            encode_ok, encoded = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 82])
            if not encode_ok:
                continue
            with LIVE_LOCK:
                LIVE_STATE["last_jpeg"] = encoded.tobytes()
                current_error = str(LIVE_STATE.get("error") or "")
                if not current_error.startswith("Live YOLO failed"):
                    LIVE_STATE["error"] = None
    finally:
        if capture is not None:
            capture.release()
        with LIVE_LOCK:
            _close_recording_writer_unlocked()
            LIVE_STATE["running"] = False


def _ensure_live_thread_running() -> None:
    global LIVE_THREAD
    with LIVE_LOCK:
        if LIVE_THREAD is not None and LIVE_THREAD.is_alive():
            return
        LIVE_THREAD = threading.Thread(target=_live_loop, daemon=True)
        LIVE_THREAD.start()


def start_live_stream(rtsp_url: str | None = None) -> dict[str, object]:
    target_url, discovered = resolve_live_rtsp_url(rtsp_url or LIVE_RTSP_DEFAULT)
    with LIVE_LOCK:
        LIVE_STATE["rtsp_url"] = target_url
        LIVE_STATE["discovered_streams"] = discovered
        LIVE_STATE["running"] = True
        LIVE_STATE["error"] = None
    _ensure_live_thread_running()
    return get_live_status_payload()


def stop_live_stream() -> dict[str, object]:
    with LIVE_LOCK:
        _reset_live_unlocked(reset_rtsp=False)
    return get_live_status_payload()


def start_recording(rtsp_url: str | None = None) -> dict[str, object]:
    start_live_stream(rtsp_url=rtsp_url)
    filename = f"recording_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.mp4"
    recording_path = RECORDINGS_DIR / filename
    if recording_path.exists():
        filename = f"recording_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.mp4"
        recording_path = RECORDINGS_DIR / filename
    with LIVE_LOCK:
        LIVE_STATE["recording_kind"] = "manual"
        LIVE_STATE["recording_event_id"] = None
    with LIVE_LOCK:
        LIVE_STATE["recording_path"] = recording_path
        LIVE_STATE["recording_active"] = True
        LIVE_STATE["recording_started_at_ms"] = int(time.time() * 1000)
    status = get_live_status_payload()
    status["recording_file"] = filename
    return status


def stop_recording() -> dict[str, object]:
    with LIVE_LOCK:
        recording_path = LIVE_STATE.get("recording_path")
        _close_recording_writer_unlocked()
        LIVE_STATE["recording_path"] = None
        LIVE_STATE["recording_kind"] = None
        LIVE_STATE["recording_event_id"] = None
    recording_error: str | None = None
    recording_url: str | None = None
    payload = get_live_status_payload()
    if isinstance(recording_path, Path):
        try:
            transcode_to_browser_mp4(recording_path)
        except Exception as exc:
            recording_error = str(exc)
        recording_url = f"/files/recordings/{recording_path.name}"
        payload["saved_recording"] = {
            "name": recording_path.name,
            "url": recording_url,
        }
    if recording_error:
        payload["error"] = recording_error
    return payload


def get_live_status_payload() -> dict[str, object]:
    with LIVE_LOCK:
        recording_path = LIVE_STATE.get("recording_path")
        payload = {
            "running": bool(LIVE_STATE.get("running", False)),
            "rtsp_url": str(LIVE_STATE.get("rtsp_url", LIVE_RTSP_DEFAULT)),
            "error": LIVE_STATE.get("error"),
            "recording_active": bool(LIVE_STATE.get("recording_active", False)),
            "recording_started_at_ms": LIVE_STATE.get("recording_started_at_ms"),
            "recording_file": recording_path.name if isinstance(recording_path, Path) else None,
            "discovered_streams": list(LIVE_STATE.get("discovered_streams", [])),
        }
    payload["recorded_videos"] = list_recorded_videos()
    return payload


def resolve_analysis_video() -> tuple[Path, str]:
    analysis_source = (flask_request.form.get("analysis_source", "upload") or "upload").strip().lower()
    if analysis_source == "recorded":
        selected_name = Path(flask_request.form.get("recorded_video", "")).name
        if not selected_name:
            raise ValueError("Choose a recorded video.")
        recorded_path = RECORDINGS_DIR / selected_name
        if not recorded_path.exists():
            raise ValueError("Selected recorded video does not exist anymore.")
        if not is_browser_friendly_video(recorded_path):
            transcode_to_browser_mp4(recorded_path)
        ext = recorded_path.suffix.lower() or ".mp4"
        upload_path = UPLOAD_DIR / unique_name("video", ext)
        shutil.copy2(recorded_path, upload_path)
        return upload_path, f"/files/recordings/{selected_name}"

    upload = flask_request.files.get("video")
    if upload is None or not upload.filename:
        raise ValueError("Please choose a video file.")
    allowed_media(upload.filename, include_images=False, include_videos=True)
    ext = Path(upload.filename).suffix.lower()
    upload_path = UPLOAD_DIR / unique_name("video", ext)
    upload.save(upload_path)
    display_name = unique_name("source", ext)
    display_path = OUTPUT_DIR / display_name
    shutil.copy2(upload_path, display_path)
    return upload_path, f"/files/outputs/{display_name}"


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
        raise RuntimeError(f"No model found at {
                           PT_MODEL_PATH} or {ONNX_MODEL_PATH}.")

    if REQUIRE_GPU:
        raise RuntimeError(
            "GPU mode is required but CUDA is unavailable. Set REQUIRE_GPU=0 for CPU fallback.")

    if ONNX_MODEL_PATH.exists():
        return ONNX_MODEL_PATH, "cpu", False
    if PT_MODEL_PATH.exists():
        return PT_MODEL_PATH, "cpu", False
    raise RuntimeError(f"No model found at {
                       PT_MODEL_PATH} or {ONNX_MODEL_PATH}.")


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
        "yolo_frame_workers": str(YOLO_FRAME_WORKERS),
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


def is_cuda_oom_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "cuda error" in text


def safe_yolo_predict(detector: YOLO, **kwargs):
    try:
        return detector.predict(**kwargs)
    except Exception as exc:
        preferred_device = str(kwargs.get("device", "cpu"))
        if preferred_device.startswith("cuda") and is_cuda_oom_error(exc):
            retry_kwargs = dict(kwargs)
            retry_kwargs["device"] = "cpu"
            retry_kwargs["half"] = False
            return detector.predict(**retry_kwargs)
        raise


def run_cmd(cmd: list[str], err_prefix: str) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        stderr_tail = "\n".join(proc.stderr.splitlines()[-12:])
        raise RuntimeError(f"{err_prefix}\n{stderr_tail}")


def transcode_to_browser_mp4(src_path: Path) -> Path:
    if not src_path.exists():
        raise RuntimeError(f"Source video not found for transcode: {src_path}")
    tmp_path = src_path.with_suffix(".browser.mp4")
    cmd = [
        "ffmpeg",
        "-y",
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
        str(tmp_path),
    ]
    run_cmd(cmd, "Failed to transcode recording to browser-compatible MP4.")
    tmp_path.replace(src_path)
    return src_path


def ffprobe_streams(path: Path) -> list[dict[str, object]]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_streams",
        "-of",
        "json",
        str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return []
    try:
        data = json.loads(proc.stdout)
    except Exception:
        return []
    streams = data.get("streams", [])
    return streams if isinstance(streams, list) else []


def has_video_stream(path: Path) -> bool:
    for stream in ffprobe_streams(path):
        if str(stream.get("codec_type", "")).lower() == "video":
            return True
    return False


def is_browser_friendly_video(path: Path) -> bool:
    if path.suffix.lower() != ".mp4":
        return False
    for stream in ffprobe_streams(path):
        if str(stream.get("codec_type", "")).lower() != "video":
            continue
        codec = str(stream.get("codec_name", "")).lower()
        if codec == "h264":
            return True
    return False


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
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)
                       ) if capture.isOpened() else 0
    capture.release()

    active_start: int | None = None
    frame_segments: list[tuple[int, int]] = []
    processed_frames = 0

    for batch_start_idx, frame_batch in iter_video_frame_batches(src_path, batch_size=VIDEO_BATCH_SIZE):
        batch_results = safe_yolo_predict(
            detector,
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
        if progress_cb is not None and total_frames > 0:
            scan_pct = min(
                75, max(5, int((processed_frames / total_frames) * 75)))
            progress_cb(scan_pct, "Scanning video for person detections")

    if active_start is not None:
        frame_segments.append(
            (active_start, max(processed_frames - 1, active_start)))

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
    if end_s <= start_s:
        raise RuntimeError("Invalid segment range for cutting.")
    duration_s = end_s - start_s
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src_path),
        "-ss",
        f"{start_s:.3f}",
        "-t",
        f"{duration_s:.3f}",
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
    safe_yolo_predict(
        detector,
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
    if default_out.exists() and has_video_stream(default_out):
        return default_out

    candidates = sorted((p for p in ann_root.iterdir()
                        if p.is_file()), key=lambda p: p.stat().st_mtime)
    valid_video_candidates = [p for p in candidates if has_video_stream(p)]
    if valid_video_candidates:
        return valid_video_candidates[-1]
    if not candidates:
        raise RuntimeError("Failed to produce annotated video.")
    raise RuntimeError("Failed to produce annotated video stream.")


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
    cache_key = f"{file_hash[:16]}_{model_name}_{
        runtime_name}_c{int(conf * 1000):03d}"
    clip_dir = CLIPS_CACHE_DIR / cache_key
    manifest_path = clip_dir / "manifest.json"

    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            clips = manifest.get("clips", [])
            if clips:
                all_valid = True
                for clip in clips:
                    clip_url = str(clip.get("url", ""))
                    clip_rel = clip_url.removeprefix("/files/")
                    clip_path = RUNTIME_DIR / clip_rel
                    if not clip_path.exists() or not has_video_stream(clip_path):
                        all_valid = False
                        break
                if not all_valid:
                    clips = []
                    manifest_path.unlink(missing_ok=True)
                else:
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
    clip_note_suffix = ""
    try:
        annotated_video = create_annotated_source_video(
            src_path, clip_dir, detector, person_class_ids, conf
        )
    except Exception:
        annotated_video = src_path
        clip_note_suffix = " Annotated overlay unavailable; used source video for clipping."
    _, clip_source_duration = probe_video_metadata(annotated_video)
    clips: list[dict[str, object]] = []
    for idx, (start_s, end_s) in enumerate(segments):
        end_s = min(end_s, clip_source_duration)
        if end_s <= start_s:
            continue
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
            progress_cb(min(100, clip_pct),
                        f"Cutting person clips ({idx + 1}/{len(segments)})")
    if not clips:
        raise RuntimeError("Person segments were found, but no playable clips could be cut.")

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
    return clips, f"Created {len(clips)} person clip(s) with bounding boxes and cached them.{clip_note_suffix}"


def run_person_detection(
    src_path: Path,
    conf: float,
    media_kind: str,
    progress_cb: callable | None = None,
) -> tuple[str | None, str | None, list[dict[str, object]]]:
    if media_kind == "video":
        clips, note = run_video_clipper(
            src_path, conf, progress_cb=progress_cb)
        return None, note, clips

    detector = get_model()
    person_class_ids = get_person_class_ids(detector)
    run_dir_name = unique_name("result", "")
    safe_yolo_predict(
        detector,
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
        return f"/files/outputs/{run_dir_name}/{default_out.name}", None, []

    output_files = sorted((p for p in out_dir.iterdir()
                          if p.is_file()), key=lambda p: p.stat().st_mtime)
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
    indices = np.linspace(0, max(frame_count - 1, 0),
                          num=sample_count, dtype=int)

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


def make_mosaic(
    frames: list[np.ndarray],
    n: int,
    cell_size: int = 224,
    source_color: str = "rgb",
) -> Image.Image:
    if not frames:
        raise RuntimeError("Cannot build a mosaic with zero frames.")
    n = max(1, n)
    cells = n * n
    tile_size = max(1, int(cell_size))
    canvas = np.zeros((n * tile_size, n * tile_size, 3), dtype=np.uint8)
    frame_count = len(frames)
    for i in range(cells):
        frame = frames[i % frame_count]
        if frame is None or frame.size == 0:
            continue
        interpolation = cv2.INTER_AREA if frame.shape[
            0] > tile_size or frame.shape[1] > tile_size else cv2.INTER_LINEAR
        resized = cv2.resize(frame, (tile_size, tile_size),
                             interpolation=interpolation)
        r, c = divmod(i, n)
        y0 = r * tile_size
        x0 = c * tile_size
        canvas[y0: y0 + tile_size, x0: x0 + tile_size] = resized

    if source_color.lower() == "bgr":
        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    return Image.fromarray(canvas, mode="RGB")


def build_temporal_mosaics(video_path: str, n: int, t: int, cell_size: int = 224) -> list[Image.Image]:
    n = max(1, int(n))
    t = max(1, int(t))
    cells = n * n
    sampled = sample_frames(video_path, cells * t)
    sampled_count = len(sampled)
    if sampled_count == 0:
        raise RuntimeError("No frames were sampled from the video.")

    def build_one_mosaic(offset: int) -> tuple[int, Image.Image]:
        # Interleave across the full sampled timeline so each mosaic spans
        # the entire video while staying temporally staggered from others.
        frame_group = sampled[offset::t]
        if not frame_group:
            frame_group = [sampled[offset % sampled_count]]
        frame_group = frame_group[:cells]
        return offset, make_mosaic(frame_group, n=n, cell_size=cell_size, source_color="rgb")

    mosaics: list[Image.Image | None] = [None] * t
    mosaic_workers = min(t, max(1, os.cpu_count() or 1))
    if mosaic_workers == 1:
        for offset in range(t):
            idx, mosaic = build_one_mosaic(offset)
            mosaics[idx] = mosaic
        return [m for m in mosaics if m is not None]

    with concurrent.futures.ThreadPoolExecutor(max_workers=mosaic_workers) as executor:
        for idx, mosaic in executor.map(build_one_mosaic, range(t)):
            mosaics[idx] = mosaic
    return [m for m in mosaics if m is not None]


def collect_person_frames_for_mosaic(
    video_path: Path,
    conf: float,
    progress_cb: callable | None = None,
) -> list[np.ndarray]:
    detector = get_model()
    person_class_ids = get_person_class_ids(detector)
    selected: list[np.ndarray] = []
    capture = cv2.VideoCapture(str(video_path))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)
                       ) if capture.isOpened() else 0
    capture.release()
    processed_frames = 0

    for batch_start_idx, frame_batch in iter_video_frame_batches(video_path, batch_size=VIDEO_BATCH_SIZE):
        batch_results = safe_yolo_predict(
            detector,
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
            has_person = result.boxes is not None and len(result.boxes) > 0
            if not has_person:
                continue
            selected.append(frame_batch[idx_in_batch])
        processed_frames = batch_start_idx + len(frame_batch)
        if progress_cb is not None and total_frames > 0:
            scan_pct = min(
                45, max(5, int((processed_frames / total_frames) * 45)))
            progress_cb(scan_pct, "Collecting person-detected frames")

    return selected


def build_temporal_mosaics_from_frames(person_frames: list[np.ndarray], n: int, t: int, cell_size: int = 224) -> list[Image.Image]:
    if not person_frames:
        raise RuntimeError("No person-detected frames available for mosaics.")
    n = max(1, int(n))
    t = max(1, int(t))
    cells = n * n
    frame_count = len(person_frames)

    def build_one_mosaic(offset: int) -> tuple[int, Image.Image]:
        # Interleave detected frames to produce staggered mosaics.
        frame_group = person_frames[offset::t]
        if not frame_group:
            frame_group = [person_frames[offset % frame_count]]
        frame_group = frame_group[:cells]
        return offset, make_mosaic(frame_group, n=n, cell_size=cell_size, source_color="bgr")

    mosaics: list[Image.Image | None] = [None] * t
    mosaic_workers = min(t, max(1, os.cpu_count() or 1))
    if mosaic_workers == 1:
        for offset in range(t):
            idx, mosaic = build_one_mosaic(offset)
            mosaics[idx] = mosaic
        return [m for m in mosaics if m is not None]

    with concurrent.futures.ThreadPoolExecutor(max_workers=mosaic_workers) as executor:
        for idx, mosaic in executor.map(build_one_mosaic, range(t)):
            mosaics[idx] = mosaic
    return [m for m in mosaics if m is not None]


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
        raise RuntimeError(f"Unexpected llama.cpp response: {
                           response_body[:400]}")

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

    raise RuntimeError(f"llama.cpp returned no assistant text. Raw excerpt: {
                       response_body[:400]}")


def _build_image_content(images: list[Image.Image]) -> list[dict[str, object]]:
    return [{"type": "image_url", "image_url": {"url": pil_to_data_url(img)}} for img in images]


def query_llamacpp_with_images(
    images: list[Image.Image],
    prompt: str,
    max_tokens: int = 192,
    system_prompt: str | None = None,
) -> str:
    if not images:
        raise RuntimeError(
            "At least one image is required for llama.cpp query.")

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
        raise RuntimeError(f"llama.cpp server error ({exc.code}): {
                           detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Could not reach llama.cpp at {
                           endpoint}.") from exc

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
    raw = query_llamacpp_with_images(
        images, user_prompt, max_tokens=160, system_prompt=system_prompt)
    return parse_threat_assessment(raw)


def run_video_understanding(
    video_path: Path,
    n: int,
    prompt: str,
    conf: float,
    llm_max_batch_requests: int,
    use_yolo_filter: bool = True,
    progress_cb: callable | None = None,
) -> tuple[list[dict[str, str]], str, str, str, int | None, str | None]:
    n = max(1, int(n))
    if progress_cb is not None:
        progress_cb(2, "Starting video understanding")
    frames_per_mosaic = n * n
    scaled_frames_per_mosaic = frames_per_mosaic * MOSAIC_SCALE_DIVISOR
    person_frame_count: int | None = None
    if use_yolo_filter:
        person_frames = collect_person_frames_for_mosaic(
            video_path, conf=conf, progress_cb=progress_cb)
        if not person_frames:
            raise RuntimeError(
                "No YOLO person-detected frames were found for video understanding.")
        person_frame_count = len(person_frames)
        dynamic_t = max(1, (person_frame_count +
                        scaled_frames_per_mosaic - 1) // scaled_frames_per_mosaic)
        t = min(dynamic_t, MAX_DYNAMIC_MOSAICS)
        mosaics = build_temporal_mosaics_from_frames(person_frames, n=n, t=t)
    else:
        try:
            fps, duration = probe_video_metadata(video_path)
            total_frames = max(1, int(fps * duration))
        except Exception:
            total_frames = scaled_frames_per_mosaic * MAX_DYNAMIC_MOSAICS
        dynamic_t = max(1, (total_frames +
                        scaled_frames_per_mosaic - 1) // scaled_frames_per_mosaic)
        t = min(dynamic_t, MAX_DYNAMIC_MOSAICS)
        if progress_cb is not None:
            progress_cb(30, "Sampling frames evenly (YOLO filter off)")
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
            f"You are viewing temporal mosaic {idx} of {
                total_mosaics} from one video. "
            "Prioritize suspicious actions, threat indicators, and victim/property risk. "
            "Return only one short sentence. Do not include reasoning."
        )
        try:
            return idx, query_llamacpp(mosaic, constrained_prompt, max_tokens=256)
        except Exception as exc:
            return idx, f"Model inference failed for mosaic {idx}: {exc}"

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(caption_mosaic, i, mosaic)
                   for i, mosaic in enumerate(mosaics, start=1)]
        for future in concurrent.futures.as_completed(futures):
            idx, caption = future.result()
            response_texts[idx - 1] = caption
            responses[idx - 1] = f"Mosaic {idx}: {caption}"
            completed += 1
            if progress_cb is not None:
                llm_pct = 55 + int((completed / max(total_mosaics, 1)) * 40)
                progress_cb(min(95, llm_pct), f"Understanding mosaics ({
                            completed}/{total_mosaics})")

    all_captions = "\n".join(responses)
    try:
        final_summary = summarize_mosaic_answers(
            mosaics[0] if mosaics else None, all_captions)
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
    frame_source = (
        f"Person-detected frames: {person_frame_count}"
        if use_yolo_filter and person_frame_count is not None
        else "All frames (YOLO filter off)"
    )
    stats = (
        f"{frame_source} | "
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
    use_yolo_filter: bool = True,
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
        detect_processing_seconds = round(
            time.perf_counter() - detect_started, 2)
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
            use_yolo_filter=use_yolo_filter,
            progress_cb=understanding_progress,
        )
        understanding_processing_seconds = round(
            time.perf_counter() - understanding_started, 2)
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
        "live_rtsp_default": LIVE_RTSP_DEFAULT,
        "recorded_videos": list_recorded_videos(),
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
    upload_path: Path | None = None
    try:
        if get_live_status_payload().get("running"):
            stop_live_stream()
        conf = float(flask_request.form.get("conf", "0.25"))
        if conf < 0.01 or conf > 0.99:
            raise ValueError("Confidence must be between 0.01 and 0.99.")
        n = int(flask_request.form.get("n", "4"))
        llm_max_batch_requests = int(flask_request.form.get(
            "llm_max_batch_requests", str(LLM_MAX_BATCH_REQUESTS)))
        if llm_max_batch_requests < 1:
            raise ValueError("LLM max batch requests must be at least 1.")
        prompt = flask_request.form.get(
            "prompt",
            "Analyze this surveillance video for suspicious activity and potential threats, including theft, arson, vandalism, trespassing, assault, and weapon-related behavior.",
        )
        use_yolo_filter = flask_request.form.get("use_yolo_filter", "").lower() not in {"", "0", "false", "off", "no"}

        upload_path, source_video_url = resolve_analysis_video()
        context = run_full_analysis(
            upload_path,
            conf=conf,
            n=n,
            prompt=prompt,
            llm_max_batch_requests=llm_max_batch_requests,
            source_video_url=source_video_url,
            use_yolo_filter=use_yolo_filter,
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
    use_yolo_filter: bool = True,
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
            use_yolo_filter=use_yolo_filter,
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
    upload_path: Path | None = None
    try:
        if get_live_status_payload().get("running"):
            stop_live_stream()
        conf = float(flask_request.form.get("conf", "0.25"))
        if conf < 0.01 or conf > 0.99:
            raise ValueError("Confidence must be between 0.01 and 0.99.")
        n = int(flask_request.form.get("n", "4"))
        llm_max_batch_requests = int(flask_request.form.get(
            "llm_max_batch_requests", str(LLM_MAX_BATCH_REQUESTS)))
        if llm_max_batch_requests < 1:
            raise ValueError("LLM max batch requests must be at least 1.")
        prompt = flask_request.form.get(
            "prompt",
            "Analyze this surveillance video for suspicious activity and potential threats, including theft, arson, vandalism, trespassing, assault, and weapon-related behavior.",
        )
        use_yolo_filter = flask_request.form.get("use_yolo_filter", "").lower() not in {"", "0", "false", "off", "no"}

        upload_path, source_video_url = resolve_analysis_video()

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
            args=(job_id, upload_path, conf, n, prompt,
                  llm_max_batch_requests, source_video_url, use_yolo_filter),
            daemon=True,
        )
        worker.start()
        return jsonify({"job_id": job_id})
    except Exception as exc:
        if upload_path is not None and upload_path.exists():
            upload_path.unlink()
        return jsonify({"error": str(exc)}), 400


@app.post("/live/start")
def live_start():
    payload = flask_request.get_json(silent=True) or flask_request.form
    rtsp_url = payload.get("rtsp_url") if isinstance(payload, dict) else None
    try:
        return jsonify(start_live_stream(str(rtsp_url) if rtsp_url else None))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/live/discover")
def live_discover():
    payload = flask_request.get_json(silent=True) or flask_request.form
    rtsp_url = str(payload.get("rtsp_url", LIVE_RTSP_DEFAULT)) if isinstance(payload, dict) else LIVE_RTSP_DEFAULT
    try:
        parsed = urlsplit(rtsp_url if rtsp_url.startswith("rtsp://") else f"rtsp://{rtsp_url}")
        host = parsed.hostname
        if not host:
            raise ValueError("Missing RTSP host.")
        port = int(parsed.port or 554)
        username = parsed.username or LIVE_RTSP_DISCOVER_USER
        password = parsed.password or LIVE_RTSP_DISCOVER_PASSWORD
        found = discover_rtsp_streams(host=host, port=port, username=username, password=password)
        return jsonify({"streams": found, "count": len(found)})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/live/stop")
def live_stop():
    return jsonify(stop_live_stream())


@app.get("/live/status")
def live_status():
    return jsonify(get_live_status_payload())


@app.get("/live/feed")
def live_feed():
    def generate():
        boundary = b"--frame\r\n"
        while True:
            with LIVE_LOCK:
                frame_jpeg = LIVE_STATE.get("last_jpeg")
                running = bool(LIVE_STATE.get("running", False))
            if isinstance(frame_jpeg, bytes):
                yield (
                    boundary
                    + b"Content-Type: image/jpeg\r\n\r\n"
                    + frame_jpeg
                    + b"\r\n"
                )
            elif not running:
                time.sleep(0.2)
            time.sleep(0.08)

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.post("/recordings/start")
def recordings_start():
    payload = flask_request.get_json(silent=True) or flask_request.form
    rtsp_url = payload.get("rtsp_url") if isinstance(payload, dict) else None
    try:
        return jsonify(start_recording(str(rtsp_url) if rtsp_url else None))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/recordings/stop")
def recordings_stop():
    try:
        return jsonify(stop_recording())
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.get("/recordings/list")
def recordings_list():
    return jsonify({"recorded_videos": list_recorded_videos()})


@app.post("/autonomous/start")
def autonomous_start():
    payload = flask_request.get_json(silent=True) or flask_request.form
    rtsp_url = payload.get("rtsp_url") if isinstance(payload, dict) else None
    try:
        start_live_stream(str(rtsp_url) if rtsp_url else None)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400
    AUTONOMOUS_STATE["enabled"] = True
    return jsonify(_autonomous_status_payload())


@app.post("/autonomous/stop")
def autonomous_stop_route():
    AUTONOMOUS_STATE["enabled"] = False
    return jsonify(_autonomous_status_payload())


@app.get("/autonomous/status")
def autonomous_status_route():
    return jsonify(_autonomous_status_payload())


@app.get("/events")
def events_list_route():
    events = list(reversed(load_events()))
    return jsonify({"events": events})


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
