"""ONNX export + ONNX Runtime inference for framegen.

Two roles:

1. **Export** (once, on a GPU/CPU): take the torch RIFE ``Model`` and export
   the ``flownet`` as an ONNX graph. This is the single artifact that any
   backend (ROCm via MIGraphX EP, CUDA via CUDA EP, AMD GPU via DML/OpenCL
   on Windows, or pure CPU) can run.

2. **Inference** via ONNX Runtime: an ``OrtFrameGen`` wrapper that runs a
   given ONNX file with the best available Execution Provider, with a clean
   fallback chain (GPU EP -> CPU).

Why ONNX for the cross-platform story
-------------------------------------
ROCm officially supports the RX 6700 XT (gfx1031) on **Linux only**. On
Windows AMD has dropped RDNA2 from the ROCm matrix, so the only supported
AMD-GPU path on Windows is a non-ROCm backend (DirectML / OpenCL via ONNX
Runtime). Exporting to ONNX once decouples the *model* from the *runtime* so
the same graph runs everywhere.

The RIFE ``warp`` op is ``grid_sample`` (an ONNX op); if export fails on a
given backend the caller can fall back to the torch backend instead. We
export with ``opset_version=14`` and dynamic H/W.
"""

import os
import torch
import torch.nn as nn


class FrameGenONNX(nn.Module):
    """Wrapper so ``flownet`` can be traced/exported standalone.

    The exported graph's inputs are  img0 [1,3,H,W], img1 [1,3,H,W]
    and its output is                 out   [1,3,H,W].
    """

    def __init__(self, model):
        super().__init__()
        self.flownet = model.flownet
        self.flownet.eval()

    def forward(self, img0, img1):
        # 6-channel input (img0+img1) => RIFE sees gt as empty => student-only path.
        x = torch.cat([img0, img1], dim=1)
        flow, mask, merged, flow_tea, merged_tea, loss_distill = self.flownet(
            x, scale_list=[4, 2, 1], timestep=0.5
        )
        return merged[2]


def export_model(model, onnx_path, verbose=True):
    """Export the torch RIFE model to ONNX (dynamic H/W, opset 14)."""
    import onnx

    model.eval()
    for p in model.flownet.parameters():
        p.requires_grad_(False)

    H = W = 720  # static tracing shape; H/W are dynamic below
    img0 = torch.randn(1, 3, H, W)
    img1 = torch.randn(1, 3, H, W)

    wrapper = FrameGenONNX(model)
    onnx_model = onnx.export(
        wrapper,
        (img0, img1),
        export_params=True,
        input_names=["img0", "img1"],
        output_names=["out"],
        dynamic_axes={
            "img0": {2: "H", 3: "W"},
            "img1": {2: "H", 3: "W"},
            "out": {2: "H", 3: "W"},
        },
        opset_version=14,
        do_verification=False,
    )
    onnx.save(onnx_model, onnx_path)
    if verbose:
        print(f"[export] wrote {onnx_path}  ({os.path.getsize(onnx_path) / 1e6:.1f} MB)")
    return onnx_path


class OrtFrameGen:
    """ONNX Runtime inference engine with automatic Execution-Provider fallback.

    backend in {'auto', 'rocm', 'cuda', 'dml', 'opencl', 'cpu'}.
    """

    def __init__(self, onnx_path, backend=None):
        import onnxruntime as ort
        self.ort = ort
        self.onnx_path = onnx_path
        self.session = self._make_session(backend)

    def _candidate_eps(self, backend):
        all_eps = self.ort.get_available_providers()
        if backend is None:
            backend = "auto"
        if backend == "rocm":
            order = ["MIGraphXExecutionProvider", "ROCmExecutionProvider",
                     "CUDAExecutionProvider", "CPUExecutionProvider"]
        elif backend == "cuda":
            order = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif backend == "dml":
            order = ["DmlExecutionProvider", "CPUExecutionProvider"]
        elif backend == "opencl":
            order = ["OpenCLExecutionProvider", "CPUExecutionProvider"]
        elif backend == "cpu":
            order = ["CPUExecutionProvider"]
        else:  # auto
            gpu_eps = [e for e in all_eps if e in (
                "MIGraphXExecutionProvider", "ROCmExecutionProvider",
                "CUDAExecutionProvider", "DmlExecutionProvider",
                "OpenCLExecutionProvider")]
            order = gpu_eps + ["CPUExecutionProvider"]
        seen, out = set(), []
        for e in order:
            if e in all_eps and e not in seen:
                out.append(e)
                seen.add(e)
        return out or ["CPUExecutionProvider"]

    def _make_session(self, backend):
        eps = self._candidate_eps(backend)
        for ep in eps:
            try:
                opts = self.ort.SessionOptions()
                opts.set_log_severity(self.ort.logseverity_INFO)
                session = self.ort.InferenceSession(self.onnx_path, providers=[ep], sess_options=opts)
                print(f"[ort] using Execution Provider: {ep}")
                return session
            except Exception as e:
                print(f"[ort] {ep} unavailable ({e.__class__.__name__}), trying next...")
        raise RuntimeError(f"No ONNX Runtime EP available; tried {eps}")

    def __call__(self, img0_np, img1_np, timestep=0.5):
        """img0_np, img1_np : (H, W, 3) uint8 BGR -> (H, W, 3) uint8 BGR.

        Handles both 2-input graphs (img0, img1) and 3-input graphs
        (img0, img1, timestep) so the same engine works with a freshly
        exported ONNX *and* the pre-built ``edgetools/rife`` ONNX.
        """
        import numpy as np
        t0 = np.ascontiguousarray(img0_np[:, :, ::-1].astype(np.float32) / 255.0)
        t1 = np.ascontiguousarray(img1_np[:, :, ::-1].astype(np.float32) / 255.0)
        t0 = t0.transpose(2, 0, 1)[None, ...]
        t1 = t1.transpose(2, 0, 1)[None, ...]

        inputs = self.session.get_inputs()
        inp = {}
        assigned = 0
        for meta in inputs:
            name = name_lc = meta.name.lower()
            if meta.shape and len(meta.shape) == 4:
                if name_lc in ("img0", "image0", "frame0", "input", "img_a", "a"):
                    inp[name] = t0
                elif name_lc in ("img1", "image1", "frame1", "input_1", "img_b", "b"):
                    inp[name] = t1
                elif assigned == 0:
                    inp[name] = t0
                    assigned += 1
                else:
                    inp[name] = t1
                    assigned += 1
            else:
                # scalar / 1-d => the timestep slot
                inp[name] = np.array([float(timestep)], dtype=np.float32)

        out = self.session.run(None, inp)[0]
        out = np.clip(out[0] * 255.0, 0, 255).astype(np.uint8).transpose(1, 2, 0)
        return out[:, :, ::-1]  # RGB -> BGR

    @property
    def device_name(self):
        return ", ".join(self.session.get_providers())