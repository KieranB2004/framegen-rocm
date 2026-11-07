"""CPU-only smoke tests (run in CI). No GPU required."""
import numpy as np
import torch

import framegen


def test_detect_backend_label():
    vendor, name, details = framegen.backend.detect()
    assert vendor in ("cpu", "rocm", "cuda")
    assert details["torch_version"]


def test_get_device_cpu():
    from framegen.backend import get_device
    assert str(get_device("cpu")) == "cpu"


def _new_model():
    from framegen.model import Model
    m = Model()
    m.flownet.to("cpu")
    m.eval()
    return m


def test_model_forward_cpu():
    m = _new_model()
    I0 = torch.rand(1, 3, 64, 64) * 0.5
    I1 = torch.rand(1, 3, 64, 64) * 0.5
    with torch.no_grad():
        out = m.inference(I0, I1, scale=1.0)
    assert out.shape == (1, 3, 64, 64)
    assert out.min() >= 0.0 - 1e-5 and out.max() <= 1.0 + 1e-5


def test_make_inference_lengths():
    m = _new_model()
    I0 = torch.rand(1, 3, 64, 64)
    I1 = torch.rand(1, 3, 64, 64)
    for n in (1, 2, 3):
        with torch.no_grad():
            mids = m.make_inference(I0, I1, n)
        assert len(mids) == n, f"n={n}"


def test_recursive_interpolate_length():
    from framegen.cli import recursive_interpolate
    def interp(a, b):
        return np.full((8, 8, 3), 128, dtype=np.uint8)
    a = np.zeros((8, 8, 3), np.uint8)
    b = np.full((8, 8, 3), 255, np.uint8)
    for n in (1, 2, 3, 4):
        assert len(recursive_interpolate(a, b, n, interp)) == n


def test_videoio_roundtrip():
    from framegen.videoio import frame_to_tensor, tensor_to_frame
    a = np.full((8, 8, 3), [200, 100, 50], np.uint8)  # BGR
    t = frame_to_tensor(a, torch.device("cpu"))
    assert t.shape == (1, 3, 8, 8)
    back = tensor_to_frame(t)
    assert back.dtype == np.uint8 and back.shape == a.shape
    assert np.array_equal(back, a)
