# framegen-rocm

**Open-source neural frame generation (frame interpolation) on AMD GPUs via ROCm — with CPU/ONNX fallbacks so the same model runs on Windows, Linux, and anywhere.**

Built on the [RIFE](https://arxiv.org/abs/2109.14908) architecture (ECCV 2022, MIT license), ported and re-structured for this project.

> [![CI](https://github.com/KieranB2004/framegen-rocm/actions/workflows/CI/badge.svg)](https://github.com/KieranB2004/framegen-rocm/actions/workflows/CI)

---

## What it does

Given a video, it **inserts frames between adjacent ones** (e.g. 24→48, 30→90 fps) using a learned neural network that estimates an *intermediate flow* and warps + blends the two input frames into a smooth in-between frame. This is exactly the job of “frame generation” (a.k.a. frame interpolation / temporal upscaling).

```
  input:  f0   f1   f2   f3          (30 fps)
  model  ──>  [generates f0.5, f1.5, f2.5] between each pair
  output: f0 f0.5 f1 f1.5 f2 f2.5 f3   (60 fps)
```

---

## Why RIFE, and not FSR 4.0?

Be aware of the landscape (important for interviews too):

| | FSR 4.0 (ML) | FSR 3 (analytic) | **RIFE (this repo)** |
|---|---|---|---|
| Open source? | ❌ closed weights | ✅ FFX SDK (MIT) | ✅ MIT |
| Hardware | **RDNA4 only** (RX 9000) | RDNA2+ | RDNA2+ (incl. **6700 XT**) |
| Standalone app? | ❌ game SDK (needs depth+motion) | ❌ game SDK (needs depth+motion) | ✅ works on any video |

**You cannot run FSR 4.0 ML on a 6700 XT** (wrong GPU generation, closed weights). The open, shippable “frame generator like FSR4” that runs on AMD GPUs is **RIFE** — which is what this repo is.

---

## Hardware reality (read this first)

- The **RX 6700 XT (gfx1031, RDNA2) is officially supported on Linux ROCm.** ✅
- It is **NOT officially supported on Windows** ROCm (AMD dropped RDNA2 from the Windows matrix). On Windows the supported GPU path is **ONNX Runtime + DirectML/OpenCL** (or CPU). ✅ (this repo handles it)
- So: **Linux + ROCm = the real GPU path.** Windows = ONNX fallback (DirectML for AMD GPU, or CPU).

---

## Quickstart

### 🐧 Linux + ROCm (recommended, full GPU)

```bash
# 0. you need ROCm + its driver installed (you said this works on your box)
git clone https://github.com/KieranB2004/framegen-rocm
cd framegen-rocm
python -m venv .venv && source .venv/bin/activate
pip install -e .                                    # base deps
cd .. && pip install torch --index-url https://download.pytorch.org/whl/rocm6.4 && cd -
#          ↑ use the index that matches YOUR rocM (see setup.sh / docs/rocm-guide.md)

python -m framegen.cli --list-backends              # confirm a GPU EP is present
python scripts/make_demo.py                         # → demo.mp4
cp demo.mp4 ../   # or run from repo: python scripts/make_demo
python -m framegen.cli demo.mp4 --mult 2            # 2× fps on torch-ROCm
python -m framegen.cli my_video.mp4 --mult 4 --fp16 --tta
```

### 🪟 Windows (ONNX → DirectML/OpenCL/CPU)

```bash
pip install -r requirements-win.txt
python -m framegen.cli --list-backends
python -m framegen.cli my_video.mp4 --onnx models/rife49.onnx --backend dml   # AMD GPU (DirectML)
python -m framegen.cli my_video.mp4 --onnx models/rife49.onnx --backend cpu    # CPU (always works)
# the ONNX auto-downloads to ./models on first run
```

### 🖥 CPU (any OS)

```bash
pip install -e . && pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m framegen.cli demo.mp4 --mult 2 --device cpu
```

---

## How the model works (RIFE)

RIFE takes two consecutive frames `I0, I1` and a timestep `t` (here always 0.5) and outputs the in-between frame `It`:

```
I0 ──┬──────────────────────────────┐
     │   IFNet (3× IFBlock)          │  → intermediate flow field F (per-pixel, both dirs)
I1 ──┴──────────────────────────────┴──>
        │  warp(I0, F0)      warp(I1, F1)   (backward sample)
        │   I0_warp          I1_warp
        └────────►  UNet + ContextNet (detail refine)  ──>
                        mask σ (per-pixel blend weight)
        It = I0_warp·σ  +  I1_warp·(1−σ)
```

1. **Intermediate flow** — a per-pixel offset that points to where “the pixel at time t” came from (estimated directly, not from a full optical-flow pair).
2. **Backward warp** — `F.grid_sample` samples each input frame along the flow → two warped images.
3. **Mask (σ)** — learned per-pixel weight deciding how much each warped frame contributes (handles occlusions).
4. **Blend** — the in-between frame is a weighted sum.
5. **UNet refine** — a U-Net + context nets add high-frequency detail.

**Training is student/teacher (distillation):** the *teacher* network is supervised by the real intermediate frame (GT); the *student* learns to match the teacher. At inference **only the student** is deployed — that’s the single `.pkl` / `.onnx` you run.

### “Why recursive halving?”

To generate `n` frames between `I0` and `I1`, we don’t interpolate `n` pairs independently (that drifts). Instead we compute the **midpoint**, then **recursively** interpolate each half — keeping flow estimation local and stable:

```
make(I0, I1, 3) = [ mid(I0,I1) , ... ]
make(I0,I1,3): mid = inf(I0,I1); first=make(I0,mid,1); second=make(mid,I1,1)
              = [first, mid, second]      # 3 frames
```

---

## Why ONNX for the Windows / cross-platform path?

One RIFE graph, many **Execution Providers (EPs)** via ONNX Runtime:

| OS | EP | GPU? |
|---|---|---|
| Linux + ROCm | **MIGraphX EP** (or native torch-ROCm) | ✅ AMD |
| Windows + AMD | **DirectML EP** (or OpenCL) | ✅ AMD |
| Windows + CPU | CPU EP | ✅ (slow) |
| Linux + NVIDIA | CUDA EP | ✅ NVIDIA |

So the *model* is written once; the *runtime* picks the right EP per platform. That’s the whole cross-platform story.

---

## Project structure

```
framegen-rocm/
├── src/framegen/
│   ├── model/
│   │   ├── __init__.py      # re-exports warp, IFNet, IFBlock, Model, losses
│   │   ├── ifnet.py         # RIFE IFNet + IFBlock (the core network)
│   │   ├── refine.py        # ContextNet, UNet, Conv2, conv/deconv
│   │   ├── warp.py          # backward warp (grid_sample) + Warp nn.Module
│   │   ├── rife.py          # Model wrapper: inference / make_inference / update (train)
│   │   ├── laplacian.py     # Laplacian L1 loss
│   │   └── loss.py          # EPE, SOBEL losses
│   ├── backend.py           # ROCm / CUDA / CPU detection + device selection
│   ├── videoio.py           # decode/encode, BGR<->tensor, audio re-mux
│   ├── onnx.py              # export_model + OrtFrameGen (EP fallback)
│   ├── weights.py           # stable HF mirrors (torch .pkl + onnx)
│   └── cli.py               # the app you run
├── scripts/                 # make_demo.py · export_onnx.py · train_finetune.py
├── tests/                   # CPU smoke tests
├── docs/                    # rocm-guide.md · interview-prep.md
├── .github/workflows/ci.yml
├── setup.sh · requirements-*.txt · pyproject.toml
└── LICENSE
```

---

## Training / fine-tuning

The `Model.update(imgs, gt)` method implements the RIFE training step (student+teacher+distillation). To fine-tune on your own data, feed it pairs of frames + a ground-truth middle frame (see [scripts/train_finetune.py](scripts/train_finetune.py)). Full pre-training of RIFE needs the HD/UCF datasets and multi-GPU — overkill for a 6700 XT — so **fine-tuning a pretrained checkpoint** is the realistic path here.

---

## Tests

```bash
python -m pytest -v      # CPU-only smoke tests (run in CI)
python -m framegen.cli --list-backends
```

---

## Learning docs (this is the “teach me” part)

- **[docs/rocm-guide.md](docs/rocm-guide.md)** — ROCm/HIP architecture, install, verification, debugging, profiling, and the exact gotchas for a 6700 XT.
- **[docs/interview-prep.md](docs/interview-prep.md)** — ML-systems, GPU programming, and ROCm-specific interview questions with model answers.

---

## License

MIT. RIFE model code is a faithful port of [megvii-research/ECCV2022-RIFE](https://github.com/megvii-research/ECCV2022-RIFE) (MIT). Model weights are MIT-licensed mirrors.
