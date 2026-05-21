"""Reproduce the analysis pipeline failure against the most recent recording.

Mirrors what the /analyze/start endpoint does for a "recorded" source:
  1. resolve_analysis_video equivalent (copy recording into UPLOAD_DIR)
  2. run_full_analysis(...) with the same arguments the form would send
  3. print detect_error / understanding_error / overall_error and full tracebacks

Run with: uv run python test_recorded_pipeline.py
"""
from __future__ import annotations

import shutil
import sys
import time
import traceback
from pathlib import Path

import main as M


def newest_recording() -> Path:
    candidates = sorted(
        M.RECORDINGS_DIR.glob("*.mp4"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise SystemExit("No recordings found in " + str(M.RECORDINGS_DIR))
    return candidates[0]


def newest_source_or_recording() -> Path:
    """Pick the most recent video the user actually analyzed: either a recording
    in RECORDINGS_DIR or a staged upload copy in OUTPUT_DIR (source_*.mp4)."""
    candidates: list[Path] = list(M.RECORDINGS_DIR.glob("*.mp4")) + list(M.OUTPUT_DIR.glob("source_*.mp4"))
    if not candidates:
        raise SystemExit("No analyzable video files found.")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main() -> int:
    recording = newest_source_or_recording() if "--newest" in sys.argv else newest_recording()
    print(f"[test] using recording: {recording.name} ({recording.stat().st_size} bytes)")

    if not M.is_browser_friendly_video(recording):
        print("[test] transcoding to browser-friendly mp4 (matches resolve_analysis_video)")
        M.transcode_to_browser_mp4(recording)

    upload_path = M.UPLOAD_DIR / M.unique_name("video", recording.suffix or ".mp4")
    shutil.copy2(recording, upload_path)
    print(f"[test] staged upload at: {upload_path}")

    conf = 0.25
    n = 4
    llm_max_batch_requests = M.LLM_MAX_BATCH_REQUESTS
    prompt = (
        "Analyze this surveillance video for suspicious activity and potential threats, "
        "including theft, arson, vandalism, trespassing, assault, and weapon-related behavior."
    )

    use_yolo_filter = "--no-yolo" not in sys.argv
    print(f"[test] use_yolo_filter={use_yolo_filter}")

    started = time.perf_counter()
    try:
        ctx = M.run_full_analysis(
            upload_path=upload_path,
            conf=conf,
            n=n,
            prompt=prompt,
            llm_max_batch_requests=llm_max_batch_requests,
            source_video_url=f"/files/recordings/{recording.name}",
            job_id=None,
            use_yolo_filter=use_yolo_filter,
        )
    except Exception:
        print("[test] run_full_analysis raised unexpectedly:")
        traceback.print_exc()
        return 2
    finally:
        if upload_path.exists():
            upload_path.unlink()

    elapsed = time.perf_counter() - started
    print(f"[test] run_full_analysis finished in {elapsed:.2f}s")

    print()
    print("=== summary ===")
    for key in (
        "overall_error",
        "detect_error",
        "detect_result_note",
        "detect_processing_seconds",
        "understanding_error",
        "understanding_stats",
        "understanding_processing_seconds",
        "threat_score",
    ):
        print(f"  {key}: {ctx.get(key)!r}")

    print()
    print("=== detection clips ===")
    for clip in (ctx.get("clip_results") or []):
        print(f"  {clip}")

    if ctx.get("understanding_summary"):
        print()
        print("=== understanding summary ===")
        print(ctx["understanding_summary"])

    if ctx.get("overall_error"):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
