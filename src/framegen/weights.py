"""Download model weights from stable Hugging Face mirrors.

  torch : Anson-Saju-George/wms-rifev3/train_log/flownet.pkl   (RIFE IFNet)
  onnx  : edgetools/rife/rife49.onnx                            (RIFE v4.9)

Prefer ``huggingface_hub`` (handles LFS + auth/redirects) and fall back to
plain HTTP on the ``/resolve/main/`` URL.

Returns (kind, path) where kind in {'torch','onnx'} and path is the
flownet.pkl file (torch) or the .onnx file (onnx).
"""

import os
import shutil

TORCH_REPO = "Anson-Saju-George/wms-rifev3"
TORCH_FILE = "train_log/flownet.pkl"
TORCH_NAME = "train_log_flownet.pkl"
ONNX_REPO = "edgetools/rife"
ONNX_FILE = "rife49.onnx"


def _hf_url(repo, filename):
    return f"https://huggingface.co/{repo}/resolve/main/{filename}"


def _download_url(url, dest, retries=3):
    for i in range(retries):
        try:
            if os.path.exists(dest) and os.path.getsize(dest) > 0:
                return dest
            import urllib.request
            urllib.request.urlretrieve(url, dest)
            if os.path.exists(dest) and os.path.getsize(dest) > 0:
                return dest
        except Exception:
            if i == retries - 1:
                raise
    return dest


def _download_hf(repo, filename, cache_dir, dest):
    import huggingface_hub as hfh
    src = hfh.hf_hub_download(
        repo_id=repo, repo_type="model", filename=filename,
        cache_dir=cache_dir,
    )
    if not os.path.exists(dest) or os.path.getsize(dest) == 0:
        shutil.copy(src, dest)
    return dest


def _ensure(path, kind, repo, fn, download, verbose):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return
    if not download:
        raise FileNotFoundError(f"No {kind} weight at {path}")
    if verbose:
        print(f"[weights] downloading {kind} -> {path}")
    try:
        _download_hf(repo, fn, os.path.dirname(path) or ".", dest := path)
    except Exception as e:
        if verbose:
            print(f"[weights] {kind}: huggingface_hub failed ({e.__class__.__name__}); trying HTTP")
        if not (os.path.exists(path) and os.path.getsize(path) > 0):
            _download_url(_hf_url(repo, fn), path)
    if not (os.path.exists(path) and os.path.getsize(path) > 0):
        raise FileNotFoundError(f"Failed to download {kind} weight")


def fetch(cache_dir="./models", want=None, download=True, verbose=True):
    """Return (kind, path) after ensuring weights are present.

    want in {'torch', 'onnx', None}:
      None -> prefer torch, fall back to onnx if torch is unavailable/failed.
    """
    os.makedirs(cache_dir, exist_ok=True)
    torch_path = os.path.join(cache_dir, TORCH_NAME)
    onnx_path = os.path.join(cache_dir, ONNX_FILE)

    if want in (None, "torch"):
        try:
            _ensure(torch_path, "torch", TORCH_REPO, TORCH_FILE, download, verbose)
            return "torch", torch_path
        except Exception as e:
            if want is None and verbose:
                print(f"[weights] torch failed ({e.__class__.__name__}); falling back to onnx")
    if want in (None, "onnx"):
        _ensure(onnx_path, "onnx", ONNX_REPO, ONNX_FILE, download, verbose)
        return "onnx", onnx_path
    raise ValueError(f"unknown want={want!r}")