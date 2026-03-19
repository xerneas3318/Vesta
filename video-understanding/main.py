from __future__ import annotations

import base64
import json
import os
from io import BytesIO
from urllib import error, request

import cv2
import gradio as gr
import numpy as np
from PIL import Image, ImageOps

LLAMACPP_BASE_URL = os.getenv("LLAMACPP_BASE_URL", "http://127.0.0.1:8078")
LLAMACPP_MODEL = os.getenv("LLAMACPP_MODEL", "local-model")


def sample_frames(video_path: str, sample_count: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video.")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        raise ValueError("Could not read video frame count.")

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
        raise ValueError("No frames could be extracted.")
    return frames


def make_mosaic(frames: list[np.ndarray], rows: int, cols: int, cell_size: int = 224) -> Image.Image:
    rows = max(1, rows)
    cols = max(1, cols)
    cells = rows * cols
    canvas = Image.new("RGB", (cols * cell_size, rows * cell_size), color=(0, 0, 0))

    for i in range(cells):
        if i >= len(frames):
            break
        frame = Image.fromarray(frames[i])
        fitted = ImageOps.fit(frame, (cell_size, cell_size), method=Image.Resampling.LANCZOS)
        r, c = divmod(i, cols)
        canvas.paste(fitted, (c * cell_size, r * cell_size))

    return canvas


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

        # Some reasoning-capable models return this separate field.
        reasoning = message.get("reasoning_content")
        if isinstance(reasoning, str) and reasoning.strip():
            return reasoning.strip()

    # Fallbacks used by some OpenAI-compatible servers.
    text = first.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip()

    delta = first.get("delta")
    if isinstance(delta, dict):
        d_content = delta.get("content")
        d_text = extract_message_content(d_content).strip()
        if d_text:
            return d_text

    raise RuntimeError(
        "llama.cpp returned a response but no assistant text was found. "
        f"Raw excerpt: {response_body[:400]}"
    )


def query_llamacpp(mosaic: Image.Image, prompt: str, max_tokens: int = 192) -> str:
    image_data_url = pil_to_data_url(mosaic)
    payload = {
        "model": LLAMACPP_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            }
        ],
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
        raise RuntimeError(
            f"Could not reach llama.cpp at {endpoint}. Ensure server is running on port 8078."
        ) from exc

    return parse_llamacpp_caption(body)


def describe_video(
    video_path: str,
    total_frames: int,
    rows: int,
    cols: int,
    prompt: str,
):
    if not video_path:
        raise gr.Error("Please upload a video.")

    cells = max(1, rows * cols)
    used_frames = max(1, total_frames)
    if used_frames != cells:
        used_frames = cells

    frames = sample_frames(video_path, used_frames)
    mosaic = make_mosaic(frames, rows=rows, cols=cols)

    user_prompt = prompt.strip() or "Describe the overall action taking place in this video."
    try:
        constrained_prompt = (
            f"{user_prompt}\n\n"
            "Return only the final caption in one short sentence. Do not include reasoning."
        )
        caption = query_llamacpp(mosaic, constrained_prompt, max_tokens=512)
    except Exception as exc:  # noqa: BLE001
        caption = f"Model inference failed: {exc}"

    return mosaic, caption


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Video Understanding with llama.cpp") as demo:
        gr.Markdown(
            "## Video Understanding with llama.cpp\n"
            "Upload a video, sample evenly spaced frames, build a mosaic, and caption the overall action.\n"
            f"Endpoint: `{LLAMACPP_BASE_URL}`"
        )

        with gr.Row():
            video = gr.Video(label="Upload Video")
            mosaic_out = gr.Image(label="Frame Mosaic", type="pil")

        with gr.Row():
            total_frames = gr.Slider(minimum=1, maximum=120, value=60, step=1, label="A: Total sampled frames")
            rows = gr.Slider(minimum=1, maximum=20, value=6, step=1, label="B: Mosaic rows")
            cols = gr.Slider(minimum=1, maximum=20, value=10, step=1, label="C: Mosaic columns")

        prompt = gr.Textbox(
            label="Caption prompt",
            value="Caption the overall action taking place across these sampled frames from a single video.",
            lines=3,
        )
        caption_out = gr.Textbox(label="Model caption", lines=8)
        run_btn = gr.Button("Analyze Video")

        run_btn.click(
            fn=describe_video,
            inputs=[video, total_frames, rows, cols, prompt],
            outputs=[mosaic_out, caption_out],
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="0.0.0.0")
