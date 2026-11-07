#!/usr/bin/env python
"""Generate a small synthetic demo video so you always have something to run.

Usage:  python scripts/make_demo.py  ->  demo.mp4
"""
import cv2
import numpy as np


def make_demo(path="demo.mp4", w=640, h=480, fps=30, n=60):
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*("m", "p", "4", "v")), fps, (w, h))
    for i in range(n):
        frame = np.zeros((h, w, 3), np.uint8)
        x = int((i / max(1, n - 1)) * w * 0.5 + w * 0.25)
        cv2.circle(frame, (x, h // 2), 40, (0, 255, 0), -1, 3)
        cv2.putText(frame, f"f={i}", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        out.write(frame)
    out.release()
    print(f"wrote {path} ({n} frames @ {fps} fps)")


if __name__ == "__main__":
    make_demo()
