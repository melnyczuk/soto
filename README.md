# soto

A local CLI that generates an AI-blended transition between two video clips,
running on Apple Silicon (MPS) with a 32GB memory budget in mind.

It uses [Wan2.1-VACE](https://github.com/Wan-Video/Wan2.1) (1.3B by default,
14B GGUF-quantized as an option -- see `--model` below) via a vendored copy of
[ComfyUI](https://github.com/comfyanonymous/ComfyUI) and
[ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) (for the GGUF-quantized
text encoder, and the 14B UNet when selected), driven **in-process as a
library** (no HTTP server, no subprocess) -- see
[`src/soto/comfy_engine.py`](src/soto/comfy_engine.py). The transition is
conditioned on several frames of real context from the tail of clip A and the
head of clip B (not just a single first/last frame), via VACE's masked
temporal video-to-video inpainting.

## Setup

Requires `uv` and `ffmpeg`.

```bash
uv sync
uv run soto setup    # vendors ComfyUI + ComfyUI-GGUF, downloads model weights
uv run soto doctor    # verifies the environment
```

## Usage

```bash
uv run soto transition clip_a.mp4 clip_b.mp4 -o output.mp4 --duration 1.5
```

Key flags: `--duration` (seconds of *generated* transition, required),
`--context` (seconds of real context taken from each clip's edge, default
1.0), `--model` (`1.3b` default, `14b-q4`, `14b-q5` -- see
[`config.py`](src/soto/config.py)'s `MODELS` dict), `--prompt`, `--steps`,
`--seed`.

To continue a clip past its last frame instead:

```bash
uv run soto extend clip.mp4 -o output.mp4 --frames 32 --iterations 3
```

`--frames` is how many frames each pass generates (exact; the context length
is nudged to satisfy VACE's frame-count stride instead). `--iterations`
repeats the extension, each pass conditioned on the tail of the previous
one's output, so the clip grows by `frames * iterations` in total. Each pass
uses `--seed + i`. `--context`, `--model`, `--prompt` and `--steps` behave as
above.

## Dreaming with pix2pixHD

`dream` also continues a clip past its last frame, but with a pix2pixHD
generator from the older soto-pix2pixHD project instead of VACE: a GAN
trained on one video to predict each frame from the one before, fed its own
output frame after frame, ported here for inference only
([`pix2pixhd.py`](src/soto/pix2pixhd.py)). It takes seconds per second of
footage rather than VACE's hours (~0.3s/frame at 1920x1088 on MPS), but it
only knows the video its checkpoint was trained on. On that video it carries
on from the last real frame and then drifts off into its own GAN-flavoured
version of it. On anything else it jumps straight there.

```bash
uv run soto dream clip.mp4 -o output.mp4 --checkpoint nasa --frames 250
```

`--checkpoint` is the name of a generator in `models/pix2pixhd/` or a path to
one. These aren't downloaded by `soto setup`; copy them in from the old
project's `checkpoints/<name>/latest_net_G.pth` (only the generator is needed,
not the discriminators or the per-epoch snapshots):

```bash
mkdir -p models/pix2pixhd
cp ../soto-pix2pixHD/checkpoints/nasa/latest_net_G.pth models/pix2pixhd/nasa.pth
```

The architecture is read off the weights, so any pix2pixHD GlobalGenerator
checkpoint with RGB in and out and instance norm will load. The clip is
processed at its own resolution, rounded to multiples of 16, the same way
the checkpoints were trained.

The old feedback-loop camera moves come along too. `--zoom N` crops N pixels
off each edge of every frame before feeding it back, for a steady push in.
`--heat-seeking N` pans N pixels per frame toward the brightest quadrant,
smoothed over 30 frames, plus N pixels of zoom so the pan never exposes the
edges. They now run as a single resample on the GPU. The original went through
uint8 and PIL for each move, and its truncation darkened every frame by about
half a level per move.

## Softening the seams

`transition`, `bridge` and `extend` all take `--crossfade N`: instead of
cutting straight from real footage to generated frames, the last/first `N`
real frames at each seam are linearly blended into the model's own decoded
version of those same frames (VACE regenerates its context frames too). The
two line up frame-for-frame, so the blend doesn't ghost; it just eases the
change in resolution, colour and texture. `N` comes out of the context, so it
can't exceed it (in `bridge`, two neighbouring bridges also split the real
footage between them). Output length is unchanged.

To do the fade by hand instead, `--export-generated PATH` also writes every
frame the model produced -- including its regeneration of the real context
frames -- as one clip on the same timeline as the main output (same length,
native resolution and frame rate), padded with the original footage wherever
the model produced nothing. E.g. for `extend`, if the model's frames start at
frame 210, the export is the clip's first 210 frames followed by all of them.
Lay it over the output in an editor and fade between the two. Where two
`bridge` runs' context overlaps, the later run's frames win.

## Model choice: 1.3B vs the 14B GGUF checkpoints

`--model 1.3b` (default) is fast: ~2.9-3.6s/step at small scale, and a real
1280x720 10s-transition/20-step run on real footage took ~1h18m total.
`--model 14b-q4`/`14b-q5` select the two pre-downloaded GGUF-quantized 14B
VACE checkpoints in `models/` instead -- these were originally ruled out
entirely (measured needing ~42GB regardless of vram-mode flags, attention
backend, or resolution/frame count), but that conclusion predated finding the
real bug below; every 14B test that produced that ~42GB figure was run
*before* the `torch.inference_mode()` fix. Retested after the fix at small
scale (480x272, 9 frames, 4 steps), 14B completed successfully at ~40s/step --
roughly 10-14x slower than 1.3B at the same scale, consistent with it being
~10x more parameters. Whether a full-length 14B run (matching the 1.3B
1h18m/10s benchmark above) is practical hasn't been tested -- extrapolating
that per-step multiplier to the same 300-frame scale suggests it could run
into many hours, but per-step scaling behavior isn't guaranteed to hold
constant across sequence lengths, so this needs an actual measurement rather
than extrapolation before committing to a long run.

## The real memory bug: a missing `torch.inference_mode()`

The 1.3B model *also* hit the same ~42GB ceiling at first, at any resolution
or attention backend -- until it turned out to have nothing to do with MPS,
model size, or GGUF at all. `comfy_engine.py` calls ComfyUI's node classes
directly rather than going through its graph executor (`execution.py`), and
that executor always runs node execution under `torch.inference_mode()`.
Without it, every VAE causal-conv chunk and every `KSampler` denoising step
was building *and retaining* an autograd graph -- nothing ever called
`.backward()`, but the graph kept every intermediate activation alive, so
memory grew across the run instead of being freed between iterations. Wrapping
the pipeline in `torch.inference_mode()` (see `run_vace_transition()`) fixed
it outright.

This was diagnosed the hard way: by measuring memory directly at each pipeline
stage, the failure point turned out to be *identical* (~42GB) across five
different configurations that should have behaved very differently --
different model sizes, different vram-mode flags, different attention
backends -- which is what eventually pointed away from "real GPU memory
limit" and toward "something retaining references it shouldn't."

`--lowvram`, `--cpu-vae`, and `--use-pytorch-cross-attention` are still forced
in the bootstrap; they're no longer required to avoid OOM but are cheap
insurance and were left in.

## Working resolution

`DEFAULT_WIDTH`/`DEFAULT_HEIGHT` are 832x480, the preferred working
resolution because it matches VACE's native training resolution. It's
treated as a target area: each clip is processed at its own aspect ratio at
about that many pixels, rounded to multiples of 16, and the generated frames
are scaled back to the clip's native resolution on output.

MPS has no flash-attention/xformers, so ComfyUI falls back to full,
non-windowed self-attention, and higher resolutions and longer
`--duration`/`--context` frame counts both mean more tokens and more memory.
The earlier default was 480x272; if a long run runs out of memory, drop back
towards that.
