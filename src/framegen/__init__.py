"""framegen — neural frame generation / frame interpolation on AMD GPUs (ROCm).

This package provides:

  * A faithful port of the RIFE (ECCV 2022) model for frame interpolation.
  * A device/backend abstraction so the same model runs on:
        - ROCm (AMD GPU, Linux)  -> torch.cuda.* on HIP
        - CUDA (NVIDIA GPU)      -> torch.cuda.*
        - CPU (anywhere)         -> torch.device('cpu')
        - ONNX Runtime (MIGraphX EP / CPU / DML) for cross-platform use
  * A video I/O layer (decode -> interpolate -> re-encode with audio).
  * A CLI.

See ``README.md`` for setup and usage.
"""

__version__ = "0.1.0"
__all__ = ["__version__"]
