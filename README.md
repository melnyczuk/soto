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

For a look as you go, rather than at the end, see
[Looking before a long run](#looking-before-a-long-run).

## Looking before a long run

Every VACE and FLUX run keeps a work dir next to its output (`gen.mkv` gets
`gen.work/`). That's `transition`, `bridge`, `extend` and `expand`. The
work dir gives you three ways to see where a run is going before you commit
hours to it.

**Progress PNGs.** While it runs, `progress/step.png` shows the model's
current guess at the picture after every sampling step. It's a rough
picture made straight from the latents, without the VAE, so it costs
nothing. Every step's guess is also kept as `<run>-step-NN.png`, so you can
see roughly when the picture settles. That's not quite what a run with
that many steps would give, though, because a shorter run takes bigger
steps. When a run finishes, its first, middle and last generated frames go
to `progress/<run>.png` and `progress/latest.png`. A run here is one
`extend` iteration, `bridge` gap or `expand` window, or one FLUX keyframe.
Leave `latest.png` open and kill the job if it's going wrong.

**Kept results.** Every model call's result is kept in `runs/` (lossless)
or `plates/` (FLUX keyframes, as PNGs). Each is filed under a hash of
everything that went into the call: the control frames themselves, seed,
steps, prompt, model and resolution. Running the same command again reuses
whatever's already there. So:

- A run that crashes or is stopped with Ctrl-C carries on from the last
  finished call when you re-run it.
- A run that asks for more of the same builds on what's there. `extend
  --iterations 6` after `--iterations 3` with the same `--seed` only makes
  the last three. An `expand` carries on from the FLUX keyframes and VACE
  windows already made.
- Anything that changes a call's inputs makes that call again, and every
  call after it that it fed into: a new `--steps`, `--prompt` or `--seed`,
  say. Calls before it are still reused.
- FLUX keyframes can be **edited by hand**. Paint over
  `plates/key-002-frame000120-*.png` and re-run: keyframe 2 is used as you
  left it. Each keyframe is made on its own, so none of the others change.
- `expand`'s subject analysis (depth and SAM) is kept in `analysis/` too,
  keyed on the clip and on `focus.py` itself. Trying out `--deform`,
  `--streak`, `--feather` and the rest only re-renders it.

The work dir can get big: about 0.5MB per generated frame at working size.
Delete it once you're happy. `--work DIR` puts it somewhere else, and
`--no-work` turns all of this off.

**Stills.** `expand --stills` renders just a few frames, through the whole
pipeline at the output's size, with the final composite and anamorphic look,
and writes them as PNGs to `<work>/stills/`, plus a `sheet.png` of them all,
instead of the video:

```bash
uv run soto expand clip.mp4 -o out.mp4 --size 1080x1350 --stills 3
uv run soto expand clip.mp4 -o out.mp4 --size 1080x1350 --stills 0,4.5s
```

A number is that many frames spread through the clip. A list is frames,
or times with an `s`, so a single frame needs a trailing comma: `120,`.
Each still is generated by VACE on a short window of
`config.STILL_WINDOW` (9) frames around it, rather than the whole clip, so
it takes minutes, not hours. That's a
different shape of noise from the full run's windows, so the same `--seed`
won't lay the edges out the same way. Stills are for judging the prompt,
step count, look and how far the fill reaches. `--preview draft` is still
the way to see the real layout. FLUX makes the keyframes up to the last
still and keeps them, so the full run carries on from them. The temporal
commands have no stills, since each of their frames depends on the whole
run. Watch their progress PNGs instead.

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
generator instead of VACE: a GAN trained on one video to predict each frame
from the one before, fed its own output frame after frame
([`pix2pixhd.py`](src/soto/pix2pixhd.py)). It takes seconds per second of
footage rather than VACE's hours (~0.3s/frame at 1920x1088 on MPS), but it
only knows the footage its checkpoint was trained on. On that footage it
carries on from the last real frame and then drifts off into its own
GAN-flavoured version of it. On anything else it jumps straight there. Train
one on your own footage with `soto train`
([below](#training-your-own-dream-model)).

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
one. These aren't downloaded by `soto setup`. Train them with `soto train`, or
copy them in from the older soto-pix2pixHD project's
`checkpoints/<name>/latest_net_G.pth` (only the generator is needed,
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

## Training your own dream model

`train` trains a generator for `dream` on your own footage
([`pix2pixhd_train.py`](src/soto/pix2pixhd_train.py)):

```bash
uv run soto train clip.mp4 --name mine
uv run soto dream clip.mp4 -o gen.mkv --checkpoint mine --frames 250
```

The networks and losses are pix2pixHD's: the same generator, a two-scale
PatchGAN discriminator, LSGAN, discriminator feature matching and VGG19
perceptual loss. Two things differ from how the old generators were trained:

- **It trains on its own output.** The old generators only ever saw real
  frames, so in a dream, fed their own output, small errors compounded until
  the picture melted. Here the generator first runs on its own for up to
  `--unroll` frames (4 by default) from a real frame, without gradients, then
  is trained to predict the next real frame from where it got to. The depth
  is random per step and its ceiling grows over the first half of training,
  so it learns both to follow the footage and to pull its own drift back
  toward it.
- **It trains at full resolution.** Frames stay at the clip's own size
  (rounded to 16) and are trained on in random `--crop`-pixel squares (512 by
  default) instead of being downscaled. The generator is fully convolutional,
  so it dreams whole frames at the scale it learned detail at. `--scale 0.5`
  trains on half-size frames instead, for footage you'll dream at that size.

New generators are pix2pixHD's default width, `--ngf 64`, four times the
parameters of the old `nasa` and `balloon` (`--ngf 32`). `--init balloon`
fine-tunes an existing generator instead of starting from scratch.

Several clips train one generator on all of them (`soto train a.mp4 b.mp4
--name mixed`), never across the join between two. That gives a generator
that has seen more kinds of picture, but it's still a next-frame GAN: on
footage unlike anything it was trained on, it jumps straight into its own
version of what it knows.

Training length is `--epochs` (10 by default) passes over every frame, or
`--steps`. On an M2 Pro, a step takes about 3s at `--ngf 64` and 1.6s at
`--ngf 32` with 512px crops, so 10 epochs of a one-minute 25fps clip (15,000
steps) is about 12 hours, or 7 at `--ngf 32`. Crop size matters less than
you'd expect, because most of each step is a fixed cost: 256px crops are only
about twice as fast as 512px.

Everything is saved under `models/pix2pixhd/`:

- `<name>.pth`: the generator, every `--save-every` steps (500), so `dream`
  can use it while training carries on.
- `<name>.train/`: the decoded frames (cached as raw RGB, about 6MB per 1080p
  frame, and reused by later runs on the same clip), the full training state,
  and a preview dream every `--preview-every` steps (2000). Each preview runs
  `--preview-seconds` (5) from the middle frame of the first clip, so
  successive previews show the dreams holding together for longer.
- `<name>.json`: the settings, inputs and timings.

Ctrl-C (or `kill`) saves and stops. Re-run the same command with `--resume`
to carry on from where it stopped. Delete `<name>.train/` once you're done to
reclaim the frame cache.

## Expanding the frame

`expand` grows a clip to a bigger output resolution, with the original
centred and new picture generated around it from what's in the frame
(outpainting):

```bash
uv run soto expand clip.mp4 --size 1920x1080 --prompt "a black cow lying in an alpine meadow, steep mountainsides, cloudy sky"
```

It writes `outputs/<clip name>-expanded.mp4` unless you give `-o`. It needs
`soto setup --flux` (FLUX.1 Fill, ~14GB) as well as the usual `soto setup`.
`--fit contain` (the default) scales the clip as large as it fits in
`--size`, so only the edges along the mismatched axis are filled. For
example, 4:3 into 16:9 fills the sides. `--fit native` keeps the clip at its
own size, so it gets a border on every side.

It's one pipeline, in four steps:

1. **Subject.** The clip's subject is found (below), so it can be kept sharp.
2. **FLUX.1 Fill outpaints keyframes.** See below.
3. **VACE bridges between the keyframes.** The clip is laid over Flux's
   edges, and Wan2.1-VACE starts from that instead of a blank border, masked
   over the new picture (and the band where the clip fades into it), so it
   leaves the clip alone. The mask moves with time. At a Flux keyframe it
   only opens to 0.3, holding that frame as an anchor; midway between two
   keyframes it opens to 0.8, where VACE invents the picture and carries it
   from one anchor to the next. That's what keeps the new edges continuous
   even though the keyframes are made independently of each other. It works
   on whole runs of `--window` frames (81 by default), each after the first
   given the previous run's last `--overlap` generated frames (8 by
   default), so the edges move with the footage and carry on across runs.
   Run `i` uses `--seed + i`. It works at the usual ~832x480 working
   resolution (`--model`, `--steps` and `--preview` work as in the other
   commands), and paints what `--prompt` describes.
4. **Blend** (`blend.py`). The generated sky comes out of VACE bluer and more
   saturated than the footage's, more so the further it is from the clip. So
   the new edges are matched to the footage first, place by place: their
   local chroma and hue (clouds and other near-greys, and the subject, left
   out of the measure) are set to the footage's, carried out from the clip
   and steadied over time. Brightness is left as VACE made it. Then VACE's
   canvas is scaled up to `--size`, and the clip's fine detail (the clip
   less the clip as VACE saw it) is added back over it, fading out toward
   the edges that border new picture. VACE's canvas already runs the clip
   into the new edges without a line, so this keeps that, at full
   sharpness. `config.EXPAND_COLOUR_MATCH` sets how strongly the colour is
   matched. `--export-generated` writes VACE's raw canvases, before any of
   that.
5. **The anamorphic look** (`anamorphic.py`) goes over the whole frame at
   `--size`, the clip and the new edges alike, as if one anamorphic lens had
   shot it all (below).

`--prompt` is what FLUX.1 Fill and VACE paint outside the clip: describing
the whole scene helps. [FLUX.1 Fill
[dev]](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev)
(non-commercial license; as a Q5 GGUF with its text encoders and VAE, ~14GB,
fetched by `soto setup --flux`) outpaints a keyframe every two seconds
of the clip to the whole canvas, starting from the clip smeared outward
rather than a blank border. Each keyframe is made independently of the
others, on the same seed, and the frames between crossfade from one to the
next while VACE bridges them (step 3 above). Chaining them
instead -- each starting from the last one's picture -- flattened the new
edges a little further every time, and over a clip of any length they washed
out to flat grey. Flux also redraws a wide band just inside the clip's edge, and
the clip fades into that over the band, so it fades into Flux's own version
of itself rather than straight into a different picture. And since the
generated picture is held from one keyframe while the footage's light and
colour drift, it's matched to each frame broadly, by how the two differ
along that band, carried on out and fading away from the clip.
`--flux-steps` (24 by default) and `--seed` set how it's made.

The look is a lens over the whole output frame, centred on the clip.
Everything but the subject softens more the further it is off the lens's
axis, in a circle round the centre: not at all near the middle, then
building steadily out to its strongest in the output's corners. So a clip
that's a small part of the output sits in the sharp middle of the lens and
gets little of it, and one that nearly fills the output softens toward its
own corners. The new picture is always at least lightly softened, just
enough to hide VACE's picture being scaled up, easing off over a band just
inside the clip, so there's no jump from the real picture to the new edges.
The blur is two and a half times wider than it is tall, and highlights smear
into long soft horizontal streaks, as through an anamorphic lens. Where the
subject runs off the clip's edge, it softens into the fill as itself.

The subject is found with two small models, both fetched by `soto setup`:
[Depth Anything V2
Small](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf)
(Apache-2.0, ~95MB, in `models/depth/`) and
[SlimSAM](https://huggingface.co/nielsr/slimsam-50-uniform), a pruned
Segment Anything (Apache-2.0, ~110MB, in `models/segment/`). SAM is
prompted all over every third frame, so every object in it is a candidate,
and each is scored as a focus of the frame: how much it stands out in
colour and motion, whether it's in front of what's around it in depth (an
object is; the ground it stands on isn't), how cleanly it sits at one depth,
a pull toward the centre, whether it runs along the frame's edge (a table
top or a hillside does; a subject mostly doesn't), and whether it was the
focus the frame before. The best is kept if it scores well enough at all,
and any others nearly as good, so several subjects all stay sharp, and a
landscape with no subject has none picked out of it. The frames between are
blended from those either side.

Everything is then blurred by how far it is in depth from the subject, as
through a lens focused on it, so the ground the subject stands on stays
sharp while the distance and the near foreground soften. That depth blur is
in the clip's own units, as through the same lens whatever's around it. And
it bends the picture, as an anamorphic lens does off its axis: stretched
outward from the centre, more sideways than up and down, swirled round
it, and with red and blue drawn apart into colour fringes. The bend comes in
very softly with the same lens, so the middle stays put, and it's strongest
at the output's corners; it's more the further each part is in depth from the
subject, and never on the subject itself. `--deform` scales it (0 turns it
off). The bent parts also drift slowly, like heat haze, as much as the
camera moves: the whole frame's shift is measured as it goes, so a shot on
a tripod doesn't ripple and a pan or a handheld shot ripples some.
`--wobble` scales that (0 turns it off). A held shot doesn't drift at all.
Finding the subject takes about 10s per second of footage.

`--focus depth` skips the segmentation and takes the subject to be what's
near the colour-and-motion guess at about its depth. That's quicker, but
depth can't tell a subject from the ground right where they meet, so on a
long subject (a cow lying down) the far end of it can be taken for
background, and a piece of it can turn up in the fill.

Footage with no distance in it (a close-up of a dog nosing the grass) has
nothing to blur away into, so the new edges there come out looking pasted
on, whatever the settings.

`--focus heuristic` skips the depth model and uses the guess alone, blurring
by distance from it across the frame instead. That guess never takes more
than about a fifth of the frame, so footage that's all texture (smoke,
water) blurs out toward the edges as a whole.

`--depth-falloff` is how far in depth from the subject the background takes
to blur and bend (a fraction of the frame's depth range: lower is a shallower depth
of field), and `--focus-falloff` the same across the frame for `--focus
heuristic`. `--feather` is how far off the lens's axis its own softness
starts (a fraction of half the output's longer side: lower softens more of
the clip), `--anamorphic-strength` how wide the blur is, and `--streak` how
bright the streaks are. Either falloff at 0 leaves the whole clip sharp but
for the lens's own softness. If a batch of frames
comes back from the GPU corrupted (it happens now and then, silently), it's
noticed and redone.

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
