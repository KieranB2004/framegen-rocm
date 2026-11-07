#!/usr/bin/env python
"""Export the current torch RIFE model to ONNX (run once, on GPU/CPU).

Usage:
    python scripts/export_onnx.py
    -> writes models/rife_exported.onnx

Requires: torch (any), onnx, and a torch checkpoint (downloaded to ./models).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from framegen.model import Model
from framegen.onnx import export_model
from framegen.weights import fetch


def main():
    _, pkl = fetch(want="torch")
    model = Model()
    model.load(os.path.dirname(pkl))
    export_model(model, "models/rife_exported.onnx")


if __name__ == "__main__":
    main()
