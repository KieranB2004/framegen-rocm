# ROCm Guide (for a 6700 XT)

This is the “how to actually use ROCm” reference. Read it top-to-bottom; the last sections are the ones that bite people in production.

## 1. What ROCm is (the stack)

ROCm is a layered stack. From bottom up:

```
 L1  Hardware: RX 6700 XT (gfx1031, RDNA2, 12 GB)
 L2  Firmware + kernel driver (amdgpu)
 L3  ROCr Runtime  (HSA — the Heterogeneous System Architecture; manages GPU context, work queues)
 L4  HIP + HIP-Clang  (the C++ programming model, CUDA-like)
 L5  Libraries:   rocBLAS · MIOpen · rocFFT · Composable Kernel · rocPRIM …
 L6  ML frameworks: PyTorch (ROCm build) · JAX · TensorFlow · ONNX Runtime (MIGraphX EP)
 L7  Apps:  your framegen, training, inference
```

Key mental model: **you usually never touch L2–L4 directly.** You write Python, call a PyTorch op, and the framework dispatches it to MIOpen (for NN ops) or rocBLAS (for GEMM) on top of HIP. ROCm is what makes `torch.nn.Conv2d` run on your 6700 XT.

### HIP vs CUDA
HIP is AMD’s portable GPU API. It’s deliberately written to look like CUDA so you can port with minimal changes:
- `cudaMalloc` → `hipMalloc`
- `cudaMemcpy` → `hipMemcpy`
- `kernel<<<grid, block>>>()` → `hipLaunchKernel()`
- `__global__` kernel → `__global__` kernel (same syntax)

For Python users you’ll almost always use the **PyTorch ROCm build**, which hides all of this. The important implication: **on a ROCm build, `torch.cuda.*` still works** — the CUDA API is a thin layer over HIP. There’s no `torch.hip` in Python; it’s all under the hood.

## 2. Verifying your 6700 XT

```bash
# hardware + driver + ROCm version
rocinfo                                  # per-GPU capabilities (gfx1031, compute cap, etc.)
amd-smi version                          # ROCm + driver versions
amdgpu-info                              # (if available)

# Python side
python - <<'PY'
import torch
print('torch', torch.__version__)
print('rocm', getattr(torch, 'version', {}).rocm)
print('cuda_api', getattr(torch, 'version', {}).cuda)
print('cuda_avail', torch.cuda.is_available())
print('gpu', torch.cuda.get_device_name(0))
print('mem', torch.cuda.get_device_properties(0).total_memory / 1e9, 'GB')
PY
```

On your 6700 XT you want: `cuda_avail = True`, `rocm` set (e.g. `6.4.x`), `gpu = AMD Radeon RX 6700 XT`. If `cuda_avail` is False on a ROCm box, the driver/runtime mismatch is almost always the cause (see gotchas).

## 3. Installing the right PyTorch-ROCm

**Do not** `pip install torch` from PyPI — that’s the CUDA/CPU wheel. You must install the ROCm build from the ROCm index:

```
# pick the index tag to match your ROCm version
#   ROCm 6.4.x  → rocm6.4
#   ROCm 6.2.x  → rocm6.2
#   ROCm 7.x    → rocm7.<x>
pip install torch --index-url https://download.pytorch.org/whl/rocm6.4
```

`setup.sh` automates the version detection → index selection → install → verify. The pairing (ROCm version ↔ torch version) is fixed by AMD; if you install a mismatched torch you’ll get subtle errors or no GPU. **Verify after every install.**

## 4. Running your model on it

In this repo, `framegen.backend.get_device()` does:
1. `torch.cuda.is_available()` — true on a ROCm box with a working GPU.
2. Labels the vendor (`rocm`/`cuda`/`cpu`) using `torch.version.rocm` + device name.

```python
from framegen.backend import get_device
dev = get_device()      # torch.device('cuda:0')
model.to(dev)
```

That’s the whole “use ROCm from Python” story for inference: pick a device, move the model, run. Training is identical plus `optimizer.zero_grad()` / `.backward()` / `.step()` and a data loader on the same device.

## 5. Profiling & debugging (interview + real work)

| Tool | What it’s for |
|---|---|
| `rocprof` / `rocprofiler-compute` | GPU kernel profiling (wall-clock, occupancy, roofline) |
| `rocprofiler-systems` | CPU + GPU timing, HSA events |
| `rocprof` HTML report | see which kernels dominate, where to optimize |
| `ROCgdb` (rocgdb) | step into HIP kernels like a C++ debugger |
| `rocm-smi` | GPU utilization / memory / clocks (like nvidia-smi) |
| `rocm_debug_agent` / ROCdbgapi | attach to a process and capture GPU state |

**First three things to check when a ROCm GPU job is slow or broken:**
1. `rocm-smi` while it runs — is the GPU actually busy, or is it waiting on the CPU (data loading, sync)?
2. Is it `OOM`? (check `total_memory` vs batch size; 12 GB is a real budget.)
3. Are you doing a sync on every iteration? (`.cpu()`, `.numpy()`, `print(tensor)` — each forces a GPU→CPU copy and stalls the pipeline.)

## 6. Precision

- **FP32** is safe. **FP16** (`--fp16`) is faster on RDNA2 but some ops (certain reductions, certain conv variants) have more rounding error; RIFE is generally FP16-friendly on 6700 XT. Use it for speed, but verify visual quality on a sample clip.
- The ONNX/MIGraphX path picks precision via the EP; you can set `precision` in the session options.

## 7. Common gotchas (the “I can’t get it to work” list)

1. **`torch.cuda.is_available()` is False on a ROCm box** → driver/ROCm version mismatch. Reinstall ROCm or match versions. Check `rocinfo` shows `gfx1031`.
2. **`ImportError` / `libamdhip64.so` not found** → ROCm libs aren’t on `LD_LIBRARY_PATH`. Source the ROCm env or set `LD_LIBRARY_PATH=/opt/rocm/lib64`.
3. **Model loads but runs slow / on CPU** → you installed the *CPU/CUDA* torch wheel by accident. Confirm `torch.version.rocm` is set.
4. **`grid_sample` / `Upsample` errors on a specific backend** → some ONNX ops have limited EP support; the torch-ROCm path is the most complete. (This is why we default to torch on ROCm and use ONNX only where needed.)
5. **`CUDA_LAUNCH_BLOCKING`-style stalls** → set `torch.backends.cudnn.enabled = True` (already the default) and avoid per-frame `.cpu()` in the hot loop.
6. **12 GB OOM at 1080p** → downscale to 720p for the hot path, use `--fp16`, or process in tiles. (RIFE is resolution-sensitive; memory scales with H×W.)

## 8. The Windows caveat (be precise in interviews)

> “AMD’s ROCm officially supports RDNA2 compute on Linux. On Windows, ROCm support is limited to certain RDNA3+ cards for runtime; RDNA2 (like the 6700 XT) is not in the Windows matrix. So a cross-platform AMD-GPU app either (a) ships a Linux ROCm build for the GPU path, or (b) re-exports to ONNX and uses the DirectML/OpenCL/CPU Execution Provider on Windows. framegen-rocm does both.”

That’s a strong, accurate, interview-ready statement.
