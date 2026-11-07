"""CLI for framegen: neural frame generation on AMD GPUs (ROCm).

Usage (ROCm Linux, quickstart)
    python -m framegen.cli input.mp4          # 2x fps, auto-choose torch-ROCm
    python -m framegen.cli --mult 4 in.mp4    # 4x fps
    python -m framegen.cli --device rocm in.mp4
    python -m framegen.cli --onnx models/rife49.onnx --backend rocm in.mp4

Usage (Windows, ONNX fallback)
    python -m framegen.cli --onnx models/rife49.onnx --backend dml in.mp4
    python -m framegen.cli --onnx models/rife49.onnx --backend cpu in.mp4

Usage (training/fine-tune)  -- see docs/interview-prep.md
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch

from framegen.backend import get_device, detect
from framegen.model import Model
from framegen.videoio import VideoReader, write_video, re_mux_audio, frame_to_tensor, tensor_to_frame
from framegen.weights import fetch


# ---------------------------------------------------------------------------
# RIFE recursive halving (classic) -- engine-agnostic
# ---------------------------------------------------------------------------
def recursive_interpolate(a, b, n, interp):
    """Return `n` frames between a and b using recursive halving.

    interp(a, b) -> one middle frame (at t=0.5) between a and b.
    """
    if n <= 0:
        return []
    mid = interp(a, b)
    if n == 1:
        return [mid]
    first = recursive_interpolate(a, mid, n // 2, interp)
    second = recursive_interpolate(mid, b, n // 2, interp)
    if n % 2:
        return [*first, mid, *second]
    else:
        return [*first, *second]


# ---------------------------------------------------------------------------
# Engines
# ---------------------------------------------------------------------------
def make_torch_engine(args, device):
    model = Model()
    model.flownet.to(device)
    model.eval()
    use_fp16 = args.fp16 and device.type != "cpu"
    if use_fp16:
        model.flownet.half()

    loaded = False
    if args.model:
        model.load(args.model)
        loaded = True
        kind = f"torch (user checkpoint {args.model})"
    else:
        kind, pkl = fetch(args.model_dir, want="torch", download=args.download, verbose=True)
        d = os.path.dirname(pkl) or ""
        model.load(d)
        loaded = True
        kind = f"torch (downloaded {os.path.basename(pkl)})"
    print(f"[framegen] {kind}  on {device}")

    def interp(a, b):
        ta = frame_to_tensor(a, device)
        tb = frame_to_tensor(b, device)
        if use_fp16:
            ta, tb = ta.half(), tb.half()
        with torch.no_grad():
            mid = model.inference(ta, tb, scale=1.0, TTA=args.tta, timestep=0.5)
        return tensor_to_frame(mid)
    return interp


def make_onnx_engine(args, backend=None):
    from framegen.onnx import OrtFrameGen
    engine = OrtFrameGen(args.onnx, backend=backend or "auto")
    print(f"[framegen] ONNX {args.onnx}  providers={engine.device_name}")
    return lambda a, b: engine(a, b)


def run(args):
    vendor, name, details = detect()
    print(f"[framegen] backend={vendor}  device={name}")
    print(f"[framegen] torch={torch.__version__}  rocm={getattr(torch, 'version', {}).rocm}  cuda={getattr(torch, 'version', {}).cuda}")

    reader = VideoReader(args.input)
    frames = list(reader.frames())
    reader.__exit__(*())
    W, H = reader.width, reader.height
    if len(frames) < 2:
        print("[framegen] video has <2 frames; nothing to interpolate")
        return
    n = max(0, args.mult - 1)
    out_fps = reader.fps * args.mult

    # choose engine
    if args.onnx:
        interp = make_onnx_engine(args, args.backend)
    elif args.model:
        device = get_device(args.device)
        interp = make_torch_engine(args, device)
    else:
        device = get_device(args.device)
        try:
            interp = make_torch_engine(args, device)
        except Exception as e:
            print(f"[framegen] torch path failed: {e}")
            print("[framegen] falling back to ONNX...")
            _, onnx_path = fetch(args.model_dir, want="onnx", download=args.download, verbose=True)
            args.onnx = onnx_path
            interp = make_onnx_engine(args, args.backend)

    out_frames = [frames[0]]
    t0 = time.time()
    for i in range(len(frames) - 1):
        a, b = frames[i], frames[i + 1]
        out_frames.extend(recursive_interpolate(a, b, n, interp))
        out_frames.append(b)
        if (i + 1) % max(1, len(frames) // 30) == 0:
            print(f"[framegen]   frame {i+1}/{len(frames)-1}  ({len(out_frames)} so far)  {time.time()-t0:.1f}s", end="\r")
    print("")
    out_path = args.output or os.path.join(
        os.path.dirname(args.input) or ".",
        os.path.splitext(os.path.basename(args.input))[0] + f"_{args.mult}x.mp4")
    write_video(out_frames, out_path, out_fps, W, H)
    print(f"[framegen] {len(frames)} -> {len(out_frames)} frames  ({reader.fps:.1f} -> {out_fps:.1f} fps)")
    if re_mux_audio(args.input, out_path):
        print("[framegen] audio re-muxed")
    else:
        print("[framegen] no audio or ffmpeg missing -> output has no audio")
    print(f"[framegen] wrote {out_path}")


def list_backends():
    try:
        import onnxruntime as ort
        print("[framegen] ONNX Runtime available providers:")
        for ep in ort.get_available_providers():
            print(f"[framegen]   - {ep}")
    except ImportError:
        print("[framegen] onnxruntime not installed")


def main():
    p = argparse.ArgumentParser(description="framegen - neural frame generation on AMD GPUs (ROCm)")
    p.add_argument("input", help="input video")
    p.add_argument("output", nargs="?", help="output video (default <stem>_<mult>x.mp4)")
    p.add_argument("--mult", type=int, default=2, help="output fps = input fps * mult (default 2)")
    p.add_argument("--device", default="auto", help="rocm|cuda|cpu|auto|cuda:N")
    p.add_argument("--backend", default=None, help="ONNX backend: rocm|cuda|dml|opencl|cpu|auto")
    p.add_argument("--fp16", action="store_true", help="fp16 inference (faster on GPU)")
    p.add_argument("--tta", action="store_true", help="temporal test-time augmentation")
    p.add_argument("--model", default=None, help="dir containing flownet.pkl (torch)")
    p.add_argument("--onnx", default=None, help="path to .onnx")
    p.add_argument("--model-dir", default="./models", help="weight cache dir")
    p.add_argument("--download", action="store_true", help="download weights if missing (default: do)")
    p.add_argument("--no-download", action="store_true")
    p.add_argument("--list-backends", action="store_true", help="print available ONNX providers")
    args = p.parse_args()
    if args.no_download:
        args.download = False
    if args.list_backends:
        list_backends()
        return
    if not os.path.exists(args.input):
        p.error(f"input not found: {args.input}")
    run(args)


if __name__ == "__main__":
    main()
