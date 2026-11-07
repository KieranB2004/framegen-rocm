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

Note: the RIFE ``warp`` op is ``grid_sample`` which is an ONNX op; if the
export fails on a given backend the caller can fall back to the torch backend
instead. We export with ``opset_version=14`` and a dynamic shape on H/W.

"""

import os
import torch
import torch.nn as nn


class FrameGenONNX(nn.Module):
    """Small wrapper so ``flownet`` can be exported standalone.

    The exported graph's inputs:  img0 [1,3,H,W], img1 [1,3,H,W], timestep [1]
    Its output:                   out   [1,3,H,W]

    We wrap the IFNet forward to accept (img0, img1) and internally build
    the 9-channel input used by RIFE, with a dummy zero ``gt`` so the
    teacher branch stays dormant at inference.
    """

    def __init__(self, model):
        super().__init__()
        self.flownet = model.flownet
        self.flownet.eval()

    def forward(self, img0, img1):
        # 9 channels: img0 (3) + img1 (3) + dummy gt (3) = 9. RIFE treats gt
        # as "present" if it has 3 channels; we want the student-only path,
        # so we pass a 6-channel input (img0+img1) instead and let gt be None.
        x = torch.cat([img0, img1], dim=1)
        flow, mask, merged, flow_tea, merged_tea, loss_distill = self.flownet(x, scale_list=[4, 2, 1], timestep=0.5)
        return merged[2]


def export_model(model, onnx_path, model_name="rife", verbose=True):
    """Export the torch RIFE model to ONNX.

    Parameters
    ----------
    model      : framegen.model.Model
    onnx_path  : str
    model_name : str, prefix for tracing input/output names
    """
    import onnx

    model.eval()
    for p in model.flownet.parameters():
        p.requires_grad_(False)

    # Static dummy shape for tracing (RIFE works with any H/W multiple of 8)
    H = W = 720  # will be dynamic below
    img0 = torch.randn(1, 3, H, W)
    img1 = torch.randn(1, 3, H, W)

    wrapper = FrameGenONNX(model)
    inputs = (
        onnx.helper.make_tensor_value_info("img0", onnx.TENSOR_TYPE_FLOAT32, [1, 3, H, W]),
        onnx.helper.make_tensor_value_info("img1", onnx.TENSOR_TYPE_FLOAT32, [1, 3, H, W]),
    )
    outputs = (
        onnx.helper.make_tensor_value_info("out", onnx.TENSOR_TYPE_FLOAT32, [1, 3, H, W]),
    )

    onnx_model = onnx.export(
        wrapper,
        (img0, img1),
        export_params=True,
        input_names=["img0", "img1"],
        output_names=["out"],
        dynamic_axes={"img0": {2: "H", 3: "W"}, "img1": {2: "H", 3: "W"}, "out": {2: "H", 3: "W"}},
        opset_version=14,
        do_verification=False,
    )
    onnx.save(onnx_model, onnx_path)
    if verbose:
        print(f"[export] wrote {onnx_path}  ({os.path.getsize(onnx_path)/1e6:.1f} MB)")
    return onnx_path


class OrtFrameGen:
    """ONNX Runtime inference engine with automatic EP fallback.

    ``backend`` in {\'auto\', \'rocm\', \'cuda\', \'dml\', \'opencl\', \'cpu\'}.
    """

    def __init__(self, onnx_path, backend=None):
        import onnxruntime as ort
        self.ort = ort
        self.onnx_path = onnx_path
        self.session = self._make_session(backend)

    def _candidate_eps(self, backend):
        """Ordered list of Execution Provider names to try, given preference."""
        all_eps = self.ort.get_available_providers()
        if backend is None:
            backend = "auto"
        order = []
        if backend == "rocm":
            order.extend(["MIGraphXExecutionProvider", "ROCmExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"])
        elif backend == "cuda":
            order.extend(["CUDAExecutionProvider", "CPUExecutionProvider"])
        elif backend == "dml":
            order.extend(["DmlExecutionProvider", "CPUExecutionProvider"])
        elif backend == "opencl":
            order.extend(["OpenCLExecutionProvider", "CPUExecutionProvider"])
        elif backend == "cpu":
            order.extend(["CPUExecutionProvider"])
        else:  # auto
            # prefer a GPU-ish EP if present, else CPU
            gpu_eps = [e for e in all_eps if e in ("MIGraphXExecutionProvider", "ROCmExecutionProvider", "CUDAExecutionProvider", "DmlExecutionProvider", "OpenCLExecutionProvider")]
            order.extend(gpu_eps + ["CPUExecutionProvider"])
        # de-duplicate preserving order, drop unknowns
        seen, out = set(), []
        for e in order:
            if e in all_eps and e not in seen:
                out.append(e)
                seen.add(e)
        return out or ["CPUExecutionProvider"]

    def _make_session(self, backend):
        eps = self._candidate_eps(backend)
        for i, ep in enumerate(eps):
            try:
                opts = self.ort.SessionOptions()
                opts.set_log_severity(self.ort.logseverity_INFO)
                session = self.ort.InferenceSession(self.onnx_path, providers=[ep], sess_options=opts)
                print(f"[ort] using Execution Provider: {ep}")
                return session
            except Exception as e:
                print(f"[ort] {ep} unavailable ({e.__class__.__name__}), trying next...")
        raise RuntimeError(f"No ONNX Runtime EP available; tried {eps}")

    def __call__(self, img0_np, img1_np):
        """img0_np, img1_np : (H, W, 3) uint8 BGR -> (H, W, 3) uint8 BGR."""
        import numpy as np
        t0 = np.ascontiguousarray(img0_np[:, :, ::-1].astype(np.float32) / 255.0)
        t1 = np.ascontiguousarray(img1_np[:, :, ::-1].astype(np.float32) / 255.0)
        t0 = t0.transpose(2, 0, 1)[None, ...]
        t1 = t1.transpose(2, 0, 1)[None, ...]
        inp = {self.session.get_inputs()[0].name: t0,
               self.session.get_inputs()[1].name: t1}
        out = self.session.run(None, inp)[0]
        out = np.clip(out[0] * 255.0, 0, 255).astype(np.uint8).transpose(1, 2, 0)
        return out[:, :, ::-1]  # RGB -> BGR

    @property
    def device_name(self):
        eps = self.session.get_providers()
        return ", ".join(eps)
