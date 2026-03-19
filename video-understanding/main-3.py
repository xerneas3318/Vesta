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


def make_mosaic(frames: list[np.ndarray], n: int, cell_size: int = 224) -> Image.Image:
    n = max(1, n)
    cells = n * n
    canvas = Image.new("RGB", (n * cell_size, n * cell_size), color=(0, 0, 0))

    for i in range(min(cells, len(frames))):
        frame = Image.fromarray(frames[i])
        fitted = ImageOps.fit(frame, (cell_size, cell_size), method=Image.Resampling.LANCZOS)
        r, c = divmod(i, n)
        canvas.paste(fitted, (c * cell_size, r * cell_size))

    return canvas


def build_temporal_mosaics(video_path: str, n: int, t: int, cell_size: int = 224) -> list[Image.Image]:
    n = max(1, int(n))
    t = max(1, int(t))
    cells = n * n
    total_samples = cells * t

    sampled = sample_frames(video_path, total_samples)

    mosaics: list[Image.Image] = []
    for offset in range(t):
        # Interleave sampled frames so each mosaic gets "every next frame" in sequence.
        frame_group = sampled[offset::t]
        mosaics.append(make_mosaic(frame_group[:cells], n=n, cell_size=cell_size))
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

    raise RuntimeError(
        "llama.cpp returned a response but no assistant text was found. "
        f"Raw excerpt: {response_body[:400]}"
    )


def query_llamacpp(
    mosaic: Image.Image, prompt: str, max_tokens: int = 192, system_prompt: str | None = None
) -> str:
    image_data_url = pil_to_data_url(mosaic)
    messages: list[dict] = []
    if system_prompt and system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt})
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ],
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
        raise RuntimeError(
            f"Could not reach llama.cpp at {endpoint}. Ensure server is running on port 8078."
        ) from exc

    return parse_llamacpp_caption(body)


def summarize_mosaic_answers(first_mosaic: Image.Image | None, mosaic_answers: str) -> str:
    if first_mosaic is None:
        raise gr.Error("No first mosaic available.")
    if not mosaic_answers or not mosaic_answers.strip():
        raise gr.Error("No mosaic answers available.")

    system_prompt = (
        "You are summarizing per-mosaic findings from a video. "
        "The provided image is only the first temporal mosaic and may miss information "
        "that appears in the mosaic answers. Prefer the full mosaic answers when conflicts appear."
    )
    user_prompt = (
        "Create one final understanding of the full video using ONLY:\n"
        "1) the first mosaic image\n"
        "2) the mosaic answers text below\n\n"
        f"Mosaic answers:\n{mosaic_answers}\n\n"
        "Return only one short paragraph."
    )
    try:
        return query_llamacpp(
            first_mosaic,
            user_prompt,
            max_tokens=384,
            system_prompt=system_prompt,
        )
    except Exception as exc:  # noqa: BLE001
        return f"Final summary failed: {exc}"


def describe_video(video_path: str, n: int, t: int, prompt: str):
    if not video_path:
        raise gr.Error("Please upload a video.")

    mosaics = build_temporal_mosaics(video_path, n=n, t=t)

    user_prompt = prompt.strip() or "Describe the overall action taking place in this video."
    responses: list[str] = []
    for i, mosaic in enumerate(mosaics, start=1):
        constrained_prompt = (
            f"{user_prompt}\n\n"
            f"You are viewing temporal mosaic {i} of {len(mosaics)} from one video. "
            "Return only one short sentence. Do not include reasoning."
        )
        try:
            caption = query_llamacpp(mosaic, constrained_prompt, max_tokens=256)
        except Exception as exc:  # noqa: BLE001
            caption = f"Model inference failed for mosaic {i}: {exc}"
        responses.append(f"Mosaic {i}: {caption}")

    gallery = [(img, f"Mosaic {i + 1}/{len(mosaics)}") for i, img in enumerate(mosaics)]
    all_captions = "\n".join(responses)
    first_mosaic = mosaics[0] if mosaics else None
    final_summary = summarize_mosaic_answers(first_mosaic, all_captions)
    return gallery, all_captions, final_summary


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Video Understanding with Temporal n x n Mosaics") as demo:
        gr.Markdown(
            "## Temporal n x n Mosaic Video Understanding\n"
            "Given `n` and `t`, this app samples `n*n*t` evenly spaced frames.\n"
            "It then creates `t` mosaics, each of size `n x n`, using interleaved frames.\n"
            f"Endpoint: `{LLAMACPP_BASE_URL}`"
        )

        with gr.Row():
            video = gr.Video(label="Upload Video")

        with gr.Row():
            n = gr.Slider(minimum=1, maximum=20, value=4, step=1, label="n (n x n per mosaic)")
            t = gr.Slider(minimum=1, maximum=24, value=4, step=1, label="t (number of mosaics)")

        prompt = gr.Textbox(
            label="Caption prompt",
            value="Caption the overall action taking place across these temporal mosaics from a single video.",
            lines=3,
        )
        run_btn = gr.Button("Analyze Video")

        mosaic_gallery = gr.Gallery(label="Temporal Mosaics", columns=4, height=420)
        caption_out = gr.Textbox(label="Collected mosaic captions", lines=12)
        final_summary_out = gr.Textbox(label="Final understanding", lines=6)

        run_btn.click(
            fn=describe_video,
            inputs=[video, n, t, prompt],
            outputs=[mosaic_gallery, caption_out, final_summary_out],
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="0.0.0.0")
