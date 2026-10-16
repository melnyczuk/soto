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

Making a video takes two steps. A generation command (`transition`,
`bridge`, `extend` or `dream`) writes only what the model made. That's a
lossless `.mkv` holding the generated frames plus the model's own version of
the real context frames on either side, at working resolution. Then
`soto crossfade` splices those frames into the original footage, fading
from each real seam into the model's version of it. That takes seconds,
so you can try as many fade lengths as you like without re-running the
model:

```bash
uv run soto transition clip_a.mp4 clip_b.mp4 -o gen.mkv --duration 1.5
uv run soto crossfade gen.mkv -o output.mp4 --frames 8
```

Transition's key flags: `--duration` (seconds of *generated* transition, required),
`--context` (seconds of real context taken from each clip's edge, default
1.0), `--model` (`1.3b` default, `14b-q4`, `14b-q5` -- see
[`config.py`](src/soto/config.py)'s `MODELS` dict), `--prompt`, `--steps`,
`--seed`.

To continue a clip past its last frame instead:

```bash
uv run soto extend clip.mp4 -o gen.mkv --frames 32 --iterations 3
```

`--frames` is how many frames each pass generates (exact; the context length
is nudged to satisfy VACE's frame-count stride instead). `--iterations`
repeats the extension, each pass conditioned on the tail of the previous
one's output, so the clip grows by `frames * iterations` in total. Each pass
uses `--seed + i`. `--context`, `--model`, `--prompt` and `--steps` behave as
above.

`--side` picks which end of the clip grows. `end` is the default. `start`
generates frames leading *into* the first frame, conditioned on the clip's
head, and `bookend` does both:

```bash
uv run soto extend clip.mp4 -o gen.mkv --frames 32 --side bookend
```

`--frames` and `--iterations` apply per side, so `bookend` adds twice as many
frames. Both sides use the same per-pass seeds, so `--side start --seed S`
reproduces the start of a `--side bookend --seed S` run. When `crossfade`
adds frames at the start, it delays the original audio to stay in sync with
the real footage, and the new frames are silent.

## Previews

`transition`, `bridge` and `extend` take `--preview fast` or `--preview
draft` for a quick, rough look before committing to a full run:

- `fast` runs at about 480x272 instead of 832x480 and takes 2 steps instead
  of 20. That's ~3x fewer tokens, and attention cost grows roughly with the
  square of that, plus 10x fewer steps. It's fine for judging the prompt,
  the motion and the seams. But the initial noise depends on the resolution,
  so the same seed won't give the same picture at full size.
- `draft` keeps the working resolution and takes 3 steps, about 7x faster.
  Frames come out soft and smeared, but the noise is the same as a full run's,
  so the same seed gives the same layout and motion.

Every run ends by printing its seed. To render a draft properly, re-run it
without `--preview` and with that seed:

```bash
uv run soto transition a.mp4 b.mp4 -o draft.mkv --duration 1.5 --preview draft
uv run soto transition a.mp4 b.mp4 -o final.mkv --duration 1.5 --seed 123456
```

Passing `--steps` explicitly overrides the preview's step count. The tiers
live in `config.PREVIEW_TIERS`. `crossfade` still scales a preview back to
the clip's native size.

## Run metadata

Every command writes a sidecar next to its output (`gen.mkv` gets
`gen.json`). The sidecar records:

- `settings`: what was actually used. That's the model and checkpoint,
  prompt and negative prompt, steps, seed (and whether it was given or
  random), working resolution, CFG, sampler, scheduler, shift, and context.
  For `dream`, it's the checkpoint, its architecture, the device, context,
  zoom and heat-seeking. For `crossfade`, it's the fade length.
- `adjustments`: every value soto changed from what you asked for, and why.
  For example: a gap rounded up to VACE's length stride, context nudged or
  cut short by the footage available, crossfades shared between neighbouring
  bridges, or steps lowered by `--preview`.
- `inputs`: each clip's probe info and any conforming (frame rate, size),
  with the target in `conform_to`, so `crossfade` can conform it the same
  way.
- `runs`: one entry per model call, with its seed, steps, frame counts,
  where its context came from, and how long it took.
- `segments`: what each stretch of the output is, in frames, seconds and
  `HH:MM:SS:FF` timecode. In a generation command's `.mkv`, that's
  `generated` frames and `context`, the model's version of real frames,
  with the input and frames each one stands in for. In `crossfade`'s
  output, it's `real` footage, `crossfade`s and `generated` frames.
- `timeline` and `audio` (generation commands): how `crossfade` lays the
  generated frames out among the real footage, and whose audio it carries
  over, at what offset.
- `generated_from` (`crossfade`): the `.mkv` it spliced and the command,
  seed and model that made it.
- `blank_intervals` (`bridge` only): each detected blank and whether it was
  bridged or skipped.
- The git commit (and whether the tree was dirty), the command line, and
  start and finish times.

The segments are also embedded in the output as chapter markers, without
re-encoding. They show up in QuickTime, VLC
and mpv. Most NLEs (Resolve, Premiere) don't import mp4 chapters, so use the
timecodes in the JSON there. Pass `--no-metadata` to skip both.

If a run crashes or is interrupted, everything logged up to that point goes
to `out.failed.json` instead. That includes the settings, the seed, the
adjustments, and every model call so far, with the one that failed marked
`"failed": true`. It also gets a `status` (`failed` or `interrupted`) and
the error with its traceback. The record is kept separate so a crash never
overwrites `out.json` from an earlier successful run. The next successful
run to that output deletes it. Bad options (e.g. an `-o` that isn't a
`.mkv`) are rejected before anything runs, so they don't leave a record.

`crossfade` needs the generation's sidecar, so `--no-metadata` on a
generation command means its output can't be spliced.

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
uv run soto dream clip.mp4 -o gen.mkv --checkpoint nasa --frames 250
uv run soto crossfade gen.mkv -o output.mp4 --frames 8
```

For `crossfade` to blend into, it first runs the generator over the last
`--context` seconds of the clip (1.0 by default), feeding it the real frames
and predicting each from the one before. That's the generator's own version
of those frames, lined up with the footage, at a cost of one extra pass per
frame. The dream carries on from the clip's last frame, exactly as before.
The `.mkv` is the clip's resolution rounded to multiples of 16, and lossless,
so it's big (about 3MB per frame at 1920x1088).

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

## Expanding the frame

`expand` grows a clip to a bigger output resolution, with the original
centred and new picture generated around it from what's in the frame
(outpainting):

```bash
uv run soto expand clip.mp4 -o output.mp4 --size 1920x1080
```

`--fit contain` (the default) scales the clip as large as it fits in
`--size`, so only the edges along the mismatched axis are filled. For
example, 4:3 into 16:9 fills the sides. `--fit native` keeps the clip at its
own size, so it gets a border on every side.

The default `--fill vace` uses the same Wan2.1-VACE model as everything
else, with a spatial mask instead of a temporal one. Each frame sits on a
canvas at the output's aspect ratio, and VACE generates everything outside
it, plus a 16px band just inside. It works on whole runs of frames at once,
so the new edges move with the footage instead of flickering frame to frame.
Runs are `--window` frames long (81 by default). Each one after the first is
given the previous run's last `--overlap` generated frames (8 by default),
so the edges carry on across runs. Run `i` uses `--seed + i`. The model
works at the usual ~832x480 working resolution. The result is scaled up to
`--size`, and the original footage is laid back over the middle at full
quality and feathered into the generated edges across the 16px band. So
only the new edges are soft. `--model`, `--prompt`, `--steps`, `--preview`
and `--seed` work as in the other commands. `--export-generated` writes the
model's raw canvases, before the original is laid back over them.

`--fill blur` uses no model: the clip goes over a blurred, slightly darkened
copy of itself scaled to fill the frame. It renders in seconds.

## Softening the seams

VACE regenerates its context frames along with the new ones, and `dream`
regenerates them from the real frames before them. A generation command keeps
all of it in its `.mkv`. `soto crossfade GEN.mkv -o OUT.mp4 --frames N` then
cuts the generated frames into the original footage. Instead of cutting
straight from real footage to generated frames, the last/first `N` real
frames at each seam are linearly blended into the model's version of those
same frames. The two line up frame-for-frame, so the blend doesn't ghost; it
just eases the change in resolution, colour and texture. `--frames 0` is a
hard cut.

A fade comes out of its seam's context, so a longer `N` is cut down to it.
Two fades over the same stretch of real footage split it between them: two
bridges close together in `bridge`, or both ends of a short clip in
`extend --side bookend`. Each cut is printed and logged in the output's
`adjustments`. Output length doesn't depend on `N`.

`crossfade` re-reads the original inputs from the paths in the sidecar,
conforms them the same way the generation did, and refuses if any has
changed since. It also refuses an `-o` whose sidecar would overwrite the
generation's, such as `gen.mp4` next to `gen.mkv`.

The `.mkv` is FFV1, so it plays in VLC and mpv but not QuickTime. Reading it
back gives exactly the frames the model produced, so splicing it is the same
as if the generation had spliced them itself. At 832x480 it's about 0.5MB a
frame.

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

`--lowvram` and `--use-pytorch-cross-attention` are still forced in the
bootstrap. They're no longer required to avoid OOM, but they're cheap
insurance and were left in.

`--cpu-vae` was forced too, after Wan's VAE seemed to leak memory on MPS
(24GB to encode 9 frames at 832x480, OOM at 17). That was the same missing
`inference_mode()`. Measured since, the VAE on MPS peaks at 5.3GB for 17
frames and 9.6GB for 81. Its output matches the CPU's to 53dB PSNR. The
CPU, oddly, held 15-21GB of RAM doing the same work. So the VAE now runs on
the GPU in bf16. Each VACE run encodes its whole control video twice and
decodes the result once. In a real 33-frame run at 832x480, encoding went
from 296s to 148s and decoding from 390s to 118s: 13 minutes down to under
6 at one step. That's around 17 minutes saved per 81-frame run. At
`--preview fast`'s 480x272, the GPU and CPU are about even. Set
`SOTO_VAE_DEVICE=cpu` to go back, or `SOTO_VAE_DTYPE=fp16`/`fp32` to change
the GPU precision.

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
