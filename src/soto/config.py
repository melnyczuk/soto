"""Paths and constants for the soto CLI."""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"
VENDOR_DIR = REPO_ROOT / "vendor"
COMFYUI_DIR = VENDOR_DIR / "ComfyUI"
COMFYUI_GGUF_DIR = COMFYUI_DIR / "custom_nodes" / "ComfyUI-GGUF"

COMFYUI_REPO_URL = "https://github.com/comfyanonymous/ComfyUI.git"
COMFYUI_GGUF_REPO_URL = "https://github.com/city96/ComfyUI-GGUF.git"

# Pinned commits, bumped deliberately via `soto setup --update`.
COMFYUI_COMMIT = "2eb609766a749e3104485979615e062e401bab97"
COMFYUI_GGUF_COMMIT = "6ea2651e7df66d7585f6ffee804b20e92fb38b8a"

# The 14B GGUF checkpoints were originally ruled out: every test of them
# predates the torch.inference_mode() fix in comfy_engine.py (see README),
# which turned out to be the actual cause of every OOM observed that day --
# including, in hindsight, probably most of what looked like a 14B-specific
# ~42GB ceiling. They're wired up as a selectable option again for
# `--model 14b-q4`/`14b-q5` so that can actually be tested properly; expect
# them to need a smaller working resolution/frame count than 1.3B at the
# same memory budget, since 14B's wider layers cost more per token even with
# the retention bug gone.
UNET_REPO = "Comfy-Org/Wan_2.1_ComfyUI_repackaged"


class ModelSpec:
    def __init__(self, filename: str, loader: str, remote_path: str | None = None):
        self.filename = filename
        self.loader = loader  # "UNETLoader" (standard) or "UnetLoaderGGUF"
        self.remote_path = (
            remote_path  # None if already present (e.g. pre-downloaded GGUF files)
        )


MODELS: dict[str, ModelSpec] = {
    "1.3b": ModelSpec(
        filename="wan2.1_vace_1.3B_fp16.safetensors",
        loader="UNETLoader",
        remote_path="split_files/diffusion_models/wan2.1_vace_1.3B_fp16.safetensors",
    ),
    "14b-q4": ModelSpec(
        filename="Wan2.1_14B_VACE-Q4_K_M.gguf", loader="UnetLoaderGGUF"
    ),
    "14b-q5": ModelSpec(
        filename="Wan2.1_14B_VACE-Q5_K_M.gguf", loader="UnetLoaderGGUF"
    ),
}
DEFAULT_MODEL = "1.3b"

# Small pieces soto setup downloads.
VAE_REPO = "Comfy-Org/Wan_2.1_ComfyUI_repackaged"
VAE_REMOTE_PATH = "split_files/vae/wan_2.1_vae.safetensors"
VAE_FILENAME = "wan_2.1_vae.safetensors"

TEXT_ENCODER_REPO = "city96/umt5-xxl-encoder-gguf"
TEXT_ENCODER_REMOTE_PATH = "umt5-xxl-encoder-Q5_K_M.gguf"
TEXT_ENCODER_FILENAME = "umt5-xxl-encoder-Q5_K_M.gguf"

VAE_DIR = MODELS_DIR / "vae"
TEXT_ENCODERS_DIR = MODELS_DIR / "text_encoders"

# Where Wan's VAE encodes and decodes, "gpu" (MPS) or "cpu", and at what
# precision on the GPU: "bf16", "fp16" or "fp32" (the CPU always uses fp32).
# The VAE used to be forced onto the CPU after it seemed to leak memory on
# MPS (24GB at 9 frames, OOM at 17), but that was measured before the
# torch.inference_mode() fix in comfy_engine.py and was the same autograd
# retention. Measured since at 832x480: GPU peak 5.3GB at 17 frames, 9.6GB
# at 81 (the CPU held 15-21GB of RAM doing the same). In a real 33-frame
# run, encoding went 296s -> 148s and decoding 390s -> 118s on the GPU. At
# --preview fast's 480x272 the two are about even.
# fp16 ran no faster than bf16 and matched the CPU's fp32 output more
# closely (72dB vs 53dB PSNR, both invisible), but bf16 is kept as the
# default for its fp32-sized range: fp16 is the one that can overflow into
# NaN/black frames on some VAEs. Override per run with SOTO_VAE_DEVICE /
# SOTO_VAE_DTYPE.
VAE_DEVICE = os.environ.get("SOTO_VAE_DEVICE", "gpu")
VAE_DTYPE = os.environ.get("SOTO_VAE_DTYPE", "bf16")
VAE_DEVICES = ("gpu", "cpu")
VAE_DTYPES = ("bf16", "fp16", "fp32")

# pix2pixHD next-frame generators for `soto dream`, as `<name>.pth`. Not
# downloaded by `soto setup`: they're trained with `soto train` (which also
# keeps its frame cache, resumable state and previews in `<name>.train/`),
# or copied in by hand from the older soto-pix2pixHD project (its
# checkpoints/<name>/latest_net_G.pth).
PIX2PIXHD_DIR = MODELS_DIR / "pix2pixhd"
# Depth Anything V2 Small, for finding the subject in `expand`
# (Apache-2.0; ~95MB, fetched by `soto setup`).
DEPTH_MODEL_REPO = "depth-anything/Depth-Anything-V2-Small-hf"
DEPTH_MODEL_DIR = MODELS_DIR / "depth" / "Depth-Anything-V2-Small-hf"
# SlimSAM (Segment Anything, pruned), for cutting the subject out whole
# (Apache-2.0; ~110MB, fetched by `soto setup`).
SEGMENT_MODEL_REPO = "nielsr/slimsam-50-uniform"
SEGMENT_MODEL_DIR = MODELS_DIR / "segment" / "slimsam-50-uniform"

# FLUX.1 Fill [dev], an image model made for in- and outpainting, for
# `soto expand` (FLUX.1 [dev] Non-Commercial License;
# ~14GB in all, fetched by `soto setup --flux`): the model as GGUF, its two
# text encoders (T5 as GGUF too) and its VAE (Comfy-Org's copy, ungated).
FLUX_FILL_REPO = "YarvixPA/FLUX.1-Fill-dev-GGUF"
FLUX_FILL_FILENAME = "flux1-fill-dev-Q5_K_S.gguf"
FLUX_T5_REPO = "city96/t5-v1_1-xxl-encoder-gguf"
FLUX_T5_FILENAME = "t5-v1_1-xxl-encoder-Q8_0.gguf"
FLUX_CLIP_REPO = "comfyanonymous/flux_text_encoders"
FLUX_CLIP_FILENAME = "clip_l.safetensors"
FLUX_VAE_REPO = "Comfy-Org/Lumina_Image_2.0_Repackaged"
FLUX_VAE_REMOTE_PATH = "split_files/vae/ae.safetensors"
FLUX_VAE_FILENAME = "flux_ae.safetensors"
# How it's run: working area (at the output's aspect), steps, guidance (Fill
# wants it high), sampler; a keyframe every this many seconds, each after the
# first started from the last one's fill and remade this much of the way, so
# the scene keeps its layout; and how far into the clip it may redraw, at
# working size, to blend its edge: wide, so the clip fades into Flux's own
# redraw of it over a long way rather than into a different picture.
FLUX_AREA = 1024 * 1024
FLUX_STEPS = 24
FLUX_GUIDANCE = 30.0
FLUX_SAMPLER = "euler"
FLUX_SCHEDULER = "normal"
FLUX_KEY_SECONDS = 2.0
FLUX_REDENOISE = 0.6
FLUX_SEAM_PX = 96

# `soto train` defaults. pix2pixHD's own: Adam at 2e-4 with beta1 0.5, held
# for the first half of training and decayed linearly to 0 over the second,
# and feature-matching and VGG losses weighted 10 each against the GAN loss.
TRAIN_LR = 2e-4
TRAIN_BETAS = (0.5, 0.999)
TRAIN_LAMBDA_FEAT = 10.0
TRAIN_LAMBDA_VGG = 10.0
TRAIN_DEFAULT_EPOCHS = 10
TRAIN_DEFAULT_CROP = 512
TRAIN_DEFAULT_UNROLL = 4
TRAIN_DEFAULT_NOISE = 0.02
# UNET goes directly in MODELS_DIR (flat), matching where the unused legacy
# GGUF checkpoints already sit -- both map to extra_model_paths.yaml's
# `diffusion_models: models/` entry, no separate subfolder needed.

# Sampling defaults, taken from Comfy-Org's own bundled Wan2.1-VACE FLF2V
# template (workflow_templates/templates/video_wan_vace_flf2v.json), which is
# the closest official example to this task (single-frame context generalized
# here to multi-frame context on each side).
CLIP_TYPE = "wan"
MODEL_SHIFT = 8.0
SAMPLER_NAME = "uni_pc"
SCHEDULER = "simple"
CFG = 6.0
DENOISE = 1.0
DEFAULT_STEPS = 20
DEFAULT_SEED = None  # None -> random per run

DEFAULT_NEGATIVE_PROMPT = (
    "过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走,过曝，"
)
DEFAULT_PROMPT = (
    "smooth continuous motion, consistent lighting, seamless natural transition, "
    "high quality, coherent scene"
)

DEFAULT_DURATION_S = None  # required CLI flag, no sensible default
DEFAULT_CONTEXT_S = 1.0

# 832x480 is the preferred working resolution: it's VACE's native training
# resolution, so the model generates at the scale it was trained on rather
# than extrapolating to a smaller one. This is a target *area* --
# cli._fit_working_resolution keeps the source clip's aspect ratio at about
# this many pixels, rounded to multiples of 16.
#
# Memory is the constraint: MPS has no flash-attention/xformers, so ComfyUI
# falls back to full, non-windowed self-attention, whose cost scales roughly
# quadratically with token count, and longer --duration/--context frame
# counts add tokens the same way resolution does. 480x272 was the earlier
# default, chosen when 640x368 looked like a hard limit; 832x480 now runs
# successfully. If a long run OOMs, drop back towards 480x272.
DEFAULT_WIDTH = 832
DEFAULT_HEIGHT = 480

# --preview tiers: (working-area override or None to keep the default, steps).
# "draft" keeps the working resolution, so the same seed reproduces its
# layout in a full run (the initial noise depends on the latent shape, not on
# the step count); "fast" drops to the old 480x272 area, so it doesn't.
PREVIEW_TIERS: dict[str, tuple[int | None, int]] = {
    "fast": (480 * 272, 2),
    "draft": (None, 3),
}

# `expand --stills`: each still is VACE run on this many frames centred on
# it (a valid VACE length), not the whole window, and only its middle kept.
STILL_WINDOW = 9
# At most this many latent frames tiled into each per-step progress PNG.
STEP_PREVIEW_FRAMES = 4

# Wan's causal VAE has temporal stride 4 (+1 leading frame); WanVaceToVideo's
# `length` should satisfy (length - 1) % 4 == 0 for a clean latent frame count.
VACE_LENGTH_STRIDE = 4

# `soto expand` (outpainting). Each VACE run covers EXPAND_WINDOW frames, 81
# being VACE's native clip length; every run after the first is given the
# previous run's last EXPAND_OVERLAP generated frames, whole, so the new
# borders carry on from the old ones instead of being reinvented each window.
EXPAND_WINDOW = 81
EXPAND_OVERLAP = 8
# The anamorphic look's defaults (see anamorphic.py): blur strength
# multiplier; how far in from the clip's edges the blur reaches full
# strength, and how far from the subject the clip's background takes to
# blur (with `--focus heuristic`), both as fractions of the output's short
# side; highlight-streak opacity; with depth, how far in depth from the
# subject the background takes to blur (as a fraction of the frame's depth
# range); how the subject's found; and how much the new edges bend and drift.
ANAMORPHIC_STRENGTH = 1.0
ANAMORPHIC_FEATHER = 0.35
ANAMORPHIC_FALLOFF = 0.3
ANAMORPHIC_DEPTH_FALLOFF = 0.3
ANAMORPHIC_FOCUS_MODES = ("segment", "depth", "heuristic")
ANAMORPHIC_STREAK = 0.8
ANAMORPHIC_DEFORM = 1.0
# Drift is scaled by how much the camera moves: none for a held shot.
ANAMORPHIC_WOBBLE = 0.5
# How far in from the clip's edges the subject itself starts to blur, as a
# fraction of the output's short side: it's otherwise kept sharp.
ANAMORPHIC_SUBJECT_EDGE = 0.2
# Blur levels (see anamorphic.py; 1 = 5% of a 16:9 clip's width, x --anamorphic-strength):
# the clip's own background away from the subject, the clip's edges, and the
# far corners of the output. Then how much wider than tall the blur is.
ANAMORPHIC_BACKGROUND = 0.25
ANAMORPHIC_EDGE = 0.25
ANAMORPHIC_FAR = 0.9
ANAMORPHIC_SQUEEZE = 2.5
EXPAND_FITS = ("contain", "native")
# VACE starts from the anamorphic look over Flux's outpainting, masked up to
# EXPAND_ANAMORPHIC_MASK (<1) as the blur comes in, so the picture shows through
# for VACE to restyle and the subject is left alone.
EXPAND_ANAMORPHIC_MASK = 0.8
DEFAULT_ANAMORPHIC_PROMPT = (
    "dreamy out-of-focus edges of the frame as shot through an anamorphic lens, soft horizontal light "
    "streaks, glowing bokeh, heavily blurred, smooth gradients, soft diffusion"
)
# What Flux is asked to paint outside the clip; describing the whole scene helps.
DEFAULT_EXPAND_PROMPT = (
    "the scene continuing naturally beyond the edges of the frame, consistent lighting and perspective, "
    "coherent scene, smooth continuous motion, high quality"
)
