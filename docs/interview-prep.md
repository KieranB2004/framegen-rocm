# ML / GPU / ROCm Interview Prep

Curated for someone building (and defending) a real GPU inference project like this one. Grouped by theme, with model answers and follow-ups.

## A. Frame interpolation / temporal ML

**Q: Explain how a neural frame interpolator works. Why is naive per-pixel blending bad?**
- Naive blending `α·I0+(1-α)·I1` blurs motion (every moving pixel becomes a smeared average). It has no notion of *where the motion went*, so fast movement = ghosting + blur.
- A proper model **estimates motion** (flow) and **warp + blend**: sample I0 and I1 along the per-pixel offset to a common “time t”, then blend with a learned mask. RIFE goes further by estimating an *intermediate* flow (toward t, not the full I0→I1 displacement), which is more stable for large displacements.

**Q: Why is RIFE’s flow “intermediate” rather than full optical flow?**
- Full optical flow (I0→I1) over large displacements is hard and multi-modal; averaging it is what causes blur. RIFE predicts the *partial* displacement from each frame to the middle frame (I0→It and I1→It), which is shorter and more local, so warps are cleaner. It’s a trick that dramatically reduces motion blur. (Know this — it’s the core of the paper.)

**Q: How do you train a model that only has 2 inputs but must produce a 3rd frame?**
- **Student/teacher distillation.** The teacher network has access to the true middle frame (GT) and learns a good flow; the student sees only I0, I1 (+t) and is trained to match the teacher (plus L1 to GT). At inference the student alone runs. This is why the deployed model is a single forward pass with no GT.

**Q: How do you generate many frames between I0 and I1 without drift?**
- **Recursive halving:** compute the midpoint, then interpolate each half independently. This keeps each flow estimation local (short displacement) so errors don’t accumulate. (It’s what `make_inference` / the CLI does.)

**Q: What metrics do you use? SSIM/PSNR vs. LPIPS vs. human eval?**
- **PSNR/SSIM** are cheap, per-pixel, good for tracking regression but can be gamed (a blurry average scores “well” on some metrics). **LPIPS** is perceptual (CNN-based) and correlates better with perceived quality. In the end, **human A/B eval** is ground truth. Know that no single metric captures “looks right”; use a suite.

## B. PyTorch internals

**Q: What happens when you call `loss.backward()`?**
- Autograd has been building a computation graph since you created the tensors. `.backward()` does a reverse-mode AD pass: it walks the graph from the loss, computes dloss/dparam for every differentiable leaf using the chain rule, and accumulates into `.grad`. `.step()` then applies AdamW (lr, β1, β2, weight decay) to update params. It’s O(graph), not O(model) — sparsity is preserved.

**Q: What’s a CUDA/HIP stream and why does it matter?**
- A **stream** is an ordered queue of GPU operations. You can overlap ops across streams (e.g. copy while compute) to hide latency. PyTorch’s default is one stream; `.pin_memory()` + `non_blocking=True` copies make H2D transfers overlap with compute. On a 12 GB GPU, overlap = free throughput.

**Q: What’s `.pin_memory()` / `non_blocking=True` and when do you use them?**
- `.pin_memory()` locks host RAM into page-locked memory so the H2D copy is DMA-efficient. `non_blocking=True` starts the copy in a background stream so compute can proceed. Together they let the next iteration overlap with the previous transfer. Use them when your data pipeline is otherwise on the CPU (data loader). (Know: it’s a bandwidth/latency overlap technique.)

**Q: What’s mixed precision and when is it safe?**
- Run ops in FP16 (fast, half the memory/bandwidth, uses Tensor Cores) and keep some ops (reductions, attention, normalization) in FP32 for accuracy. PyTorch: `torch.autocast(device, dtype=torch.float16)`. On RDNA2 it’s generally safe for inference; verify numerics. (Know the risk: loss of precision in large sums.)

## C. ROCm / GPU systems

**Q: What is ROCm, in one sentence, and what’s its role vs. CUDA?**
- “ROCm is AMD’s open software stack (HIP runtime + HSA ROCr + tuned libraries like rocBLAS/MIOpen) that makes AMD GPUs programmable, with a CUDA-compatible API layer (HIP) so frameworks port with minimal changes.” (Know the layered view: hardware → driver → HSA/ROCn → HIP → libraries → PyTorch.)

**Q: Is ROCm just “CUDA for AMD”?**
- Mostly, for Python users (torch.cuda API is the same). But under the hood it’s **HIP + HSA + LLVM-Clang**, not a CUDA shim, and the tuned libraries (MIOpen, rocBLAS via Tensile) are AMD’s own. The portability story is HIP↔CUDA, not a 1:1 binary drop-in. (This nuance is worth stating.)

**Q: How do you pick a backend/EP at runtime?**
- Probe the environment (e.g. `torch.cuda.is_available()`, `torch.version.rocm`, ONNX Runtime’s `get_available_providers()`), rank candidates by preference (GPU-EP first, CPU fallback), and construct the first working session/model. That’s exactly what `backend.py` and `OrtFrameGen` do — a **capability-based dispatch**. (Frame it as “feature detection, not hard-coding.”)

**Q: Why export to ONNX for cross-platform?**
- One graph, many runtimes. The model is backend-agnostic; the EP (MIGraphX/ROCm on Linux, DirectML/OpenCL/CPU on Windows, CUDA on NVIDIA) is the variable. It decouples *model* from *device* and is the standard way to ship the same inference graph to multiple OSes/GPU families.

## D. System design (video pipeline)

**Q: Design a 24→96 fps frame generator that stays real-time on a 12 GB GPU.**
- (This is a great whiteboard answer. Structure it:) 
  1. **Budget:** 96 fps means each frame must be produced in ≤10.4 ms end-to-end (including decode/encode). At 1080p, a single RIFE inference is ~15–40 ms on a 6700 XT in FP32 — too slow at 96 fps full-res. So: **downscale to 720p for the hot path** (or tile), use **FP16**, and **batch/overlap**.
  2. **Pipeline:** decode (hardware, e.g. VA-API/AMD AMF or ffmpeg hwaccel) → preprocess (normalize, pad) → GPU inference (stream, overlap H2D/D2H) → postprocess → encode (hwaccel). **Overlap** decode of frame N+2 while encoding frame N (pipelining). Use `non_blocking` pins.
  3. **Memory:** 12 GB budget. Keep frames in fp16 on GPU. Downscale = big win. Avoid holding the full video in RAM — stream.
  4. **Audio:** decode/encode side-band, re-mux at the end (lossless copy).
  5. **Degradation:** if a frame exceeds budget, drop to 2× fps or lower res rather than stalling — keep perceived smoothness.
  6. **Measurement:** profile with `rocprof`/`rocm-smi`; measure per-stage ms; fix the bottleneck (usually one stage dominates).

- **Why this is a good answer:** it shows budget-driven reasoning, overlap/pipelining, memory management, and profiling — the things real GPU teams live in. It directly mirrors what this repo does (and where you’d optimize next: FP16, 720p, overlap, hardware accel).

**Q: How do you debug a slow GPU pipeline?**
- (a) `rocm-smi` — is the GPU busy or idle? (b) profile with `rocprof` to find the dominant kernel/stage. (c) Check for implicit syncs (`.cpu()`, `.numpy()`, `print(tensor)`, `if tensor.item()`). (d) Check data-loading (is decode the bottleneck? use hwaccel). (e) Check memory (OOM or thrash). Walk top-down: wall-clock → per-stage → per-kernel.

## E. “You built a RIFE inference engine on ROCm” — likely follow-ups

- **Why not use a TFLite/ONNX-only design from the start?** → You need the framework (PyTorch) for the ROCm-native training/fine-tuning and for full op coverage on GPU; ONNX is the *shipping* format. Best of both. (Defensible trade-off.)
- **Why not train from scratch on a 6700 XT?** → Full RIFE needs the HD/UCF datasets + multi-GPU; 12 GB is a constraint. Realistic path = fine-tune a pretrained checkpoint on a small, task-relevant dataset. (Honest, and it shows you understand compute budgets.)
- **How do you keep the open-source claim clean?** → The model architecture/weights are MIT; we only *port* code under a compatible license and don’t bundle any closed FSR assets. (Know this — it’s legally and technically sound.)

## F. Quick-fire (have these ready)

- `grid_sample` = differentiable sampling of a tensor at arbitrary coords (used for warping).
- **Roofline** = max performance bounded by either memory bandwidth or FLOPs (arithmetic intensity). Useful for “is my kernel memory-bound or compute-bound?”
- **Occupancy** = fraction of GPU SMs (WGs) active; high occupancy hides latency (more work in flight).
- **HSA** = the hardware abstraction the ROCr runtime sits on (queues, memory, work distribution).
- **Tensile** = AMD’s GEMM/linear-algebra backend (what powers rocBLAS/hipBLASLt speed).

---

### How to use this doc
1. Read the **A–C** sections (they’re the core and map directly to what you built).
2. Rehearse the **D** whiteboard answer out loud — it’s the differentiator.
3. For any section, go *deep* on the one that matches the job you’re applying for (e.g. if it’s a GPU/HPC role, drill C + the roofline/occupancy items).
