"""Paths and constants for the soto CLI."""

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
        self.remote_path = remote_path  # None if already present (e.g. pre-downloaded GGUF files)


MODELS: dict[str, ModelSpec] = {
    "1.3b": ModelSpec(
        filename="wan2.1_vace_1.3B_fp16.safetensors",
        loader="UNETLoader",
        remote_path="split_files/diffusion_models/wan2.1_vace_1.3B_fp16.safetensors",
    ),
    "14b-q4": ModelSpec(filename="Wan2.1_14B_VACE-Q4_K_M.gguf", loader="UnetLoaderGGUF"),
    "14b-q5": ModelSpec(filename="Wan2.1_14B_VACE-Q5_K_M.gguf", loader="UnetLoaderGGUF"),
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

# VACE's native training resolution (832x480) is too large for this backend:
# MPS has no flash-attention/xformers, so ComfyUI falls back to full,
# non-windowed self-attention, whose memory cost scales roughly quadratically
# with token count. Measured directly at length=9 with the 1.3B model: 320x176
# through 576x320 all ran in under a second at ~4.3GB; 640x368 was already
# past the edge (didn't complete in 150s, vs <1s for everything at or below
# 576x320). 480x272 keeps a comfortable margin below that cliff, with room
# left for longer --duration/--context frame counts (which add to token count
# the same way resolution does).
DEFAULT_WIDTH = 480
DEFAULT_HEIGHT = 272

# Wan's causal VAE has temporal stride 4 (+1 leading frame); WanVaceToVideo's
# `length` should satisfy (length - 1) % 4 == 0 for a clean latent frame count.
VACE_LENGTH_STRIDE = 4
