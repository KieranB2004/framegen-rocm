"""Video I/O for framegen.

We use OpenCV for decode/encode (portable, no ffmpeg dependency for the
core path) and ffmpeg (subprocess) only for re-muxing the audio track
losslessly onto the new, higher-framerate video.

Tensor helpers
--------------
``frame_to_tensor``  : (H, W, 3) uint8 BGR  ->  torch [1, 3, H, W] float [0,1] (RGB)
``tensor_to_frame``  : torch [1, 3, H, W] float [0,1] -> (H, W, 3) uint8 BGR
"""

import os
import subprocess
import cv2
import numpy as np
import torch


class VideoReader:
    """Read video frames with OpenCV.

    Yields (H, W, 3) uint8 BGR frames in original order.
    """

    def __init__(self, path):
        self.path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {path}")
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 30.0)
        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.cap.release()

    def frames(self):
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            yield frame


def write_video(frames, path, fps, width, height, codec="mpeg4"):
    """Write BGR uint8 frames to an MP4/MKV file.

    codec: 'mpeg4' -> mp4 (yuv4x2p-ish via mp4v) / 'x264' would need
    an external encoder; we stick to the portable OpenCV codec.
    """
    fourcc = cv2.VideoWriter_fourcc(*("m", "p", "4", "v"))
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))
    if not out.isOpened():
        raise RuntimeError("cv2.VideoWriter could not open output (codec unsupported).")
    for f in frames:
        out.write(f)
    out.release()
    if os.path.getsize(path) == 0:
        raise RuntimeError(f"Empty output written to {path}")


def frame_to_tensor(frame_bgr, device):
    """(H, W, 3) uint8 BGR -> torch [1, 3, H, W] float [0, 1] (channel order RGB)."""
    rgb = frame_bgr[:, :, ::-1].astype(np.float32) / 255.0
    t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    return t.to(device, non_blocking=True)


def tensor_to_frame(t):
    """torch [1, 3, H, W] float [0, 1] -> (H, W, 3) uint8 BGR."""
    t = t[0].squeeze().cpu().numpy()
    t = np.clip(t, 0.0, 1.0)
    bgr = (t * 255.0).astype(np.uint8)
    bgr = bgr.transpose(1, 2, 0)[:, :, ::-1]  # CHW RGB -> HWC BGR
    return bgr


def re_mux_audio(src_video, dst_video):
    """Copy audio track from src_video to dst_video using ffmpeg (lossless copy).

    Returns True if audio was attached, False if no audio / ffmpeg failed
    (non-fatal)."""
    try:
        import shutil
        if not shutil.which("ffmpeg"):
            return False
    except Exception:
        return False

    cmd = [
        "ffmpeg", "-y",
        "-i", src_video, "-i", dst_video,
        "-c:v", "copy",
        "-c:a", "copy",
        "-map", "1:v",
        "-map", "0:a?",
        "-shortest",
        dst_video,
    ]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if p.returncode != 0:
            return False
        return True
    except Exception:
        return False
