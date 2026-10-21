"""Captioning a clip for `soto expand`'s prompt, when --prompt isn't given:
Qwen3.5-4B watches a few frames from across the clip and describes the scene
and what plausibly lies around it, which is what FLUX.1 Fill and VACE have
to paint. It's asked to leave colours out (they pull Flux and VACE off the
footage's own hues; the colour match after VACE is what sets them), and any
that slip through are dropped.

It's run once per clip and kept in the work dir, keyed on this module's own
source, so the model is only loaded the first time."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch

from soto import config, video

# How many frames, spread evenly over the clip, the model watches, and the
# long side they're scaled to.
_FRAMES = 6
_SIZE = 448
_MAX_TOKENS = 160
_INSTRUCTION = (
    "These frames are from one video shot. Write a prompt for an image generator that will extend the "
    "picture beyond its edges: one paragraph, under 70 words, describing the place, the main subject and "
    "what it is doing, the objects, the lighting, the weather and time of day, the camera framing, and "
    "what plausibly surrounds the frame. Don't use any colour words. Don't name people. Don't mention "
    "that these are frames or a video. Reply with the prompt only."
)
_COLOUR = (
    r"(?:(?:light|dark|pale|deep|bright)[- ])?(?:red|reddish|orange|yellow|yellowish|green|greenish|"
    r"blue|bluish|purple|violet|pink|pinkish|magenta|cyan|teal|turquoise|brown|brownish|beige|tan|"
    r"grey|gray|greyish|grayish|black|white|golden|gold|silver|crimson|scarlet|maroon|navy|olive|"
    r"lavender|amber|azure)"
)
# A run of colours ("blue and white", "black-and-tan"), so dropping it leaves
# no dangling "and".
_COLOURS = re.compile(rf"\b{_COLOUR}(?:(?:,\s*|\s+and\s+|\s+or\s+|-and-|-){_COLOUR})*\b,?\s*",
                      re.IGNORECASE)


def _strip_colours(text: str) -> str:
    text = _COLOURS.sub("", text)
    text = re.sub(r"\s+([,.;])", r"\1", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def _sample_frames(clip: Path, info: video.ClipInfo) -> list[np.ndarray]:
    s = _SIZE / max(info.width, info.height)
    w, h = video.even_size(round(info.width * s), round(info.height * s))
    picks = set(np.linspace(0, info.n_frames - 1, min(_FRAMES, info.n_frames)).round().astype(int).tolist())
    return [f for i, f in enumerate(video.iter_frames(clip, w, h)) if i in picks]


def caption_clip(clip: Path, info: video.ClipInfo) -> str:
    """A one-paragraph description of the clip's scene and its surroundings,
    with no colour words, for the start of `expand`'s prompt."""
    from PIL import Image
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    if not (config.CAPTION_MODEL_DIR / "config.json").is_file():
        raise FileNotFoundError(f"no caption model in {config.CAPTION_MODEL_DIR}: run `soto setup`, "
                                "or pass --prompt")
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    frames = _sample_frames(clip, info)
    processor = AutoProcessor.from_pretrained(config.CAPTION_MODEL_DIR)
    model = AutoModelForMultimodalLM.from_pretrained(config.CAPTION_MODEL_DIR, dtype=torch.bfloat16).to(dev).eval()
    try:
        content = [{"type": "image"} for _ in frames] + [{"type": "text", "text": _INSTRUCTION}]
        text = processor.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True,
                                             tokenize=False, enable_thinking=False)
        inputs = processor(text=[text], images=[Image.fromarray(f) for f in frames], return_tensors="pt").to(dev)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=_MAX_TOKENS, do_sample=False)
        reply = processor.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    finally:
        del model
        if dev.type == "mps":
            torch.mps.empty_cache()
    reply = re.sub(r"<think>.*?</think>", "", reply, flags=re.DOTALL).strip().strip('"')
    return _strip_colours(reply).rstrip(".")
