"""Download model weights from stable Hugging Face mirrors.

  torch : Anson-Saju-George/wms-rifev3/train_log/flownet.pkl   (RIFE IFNet)
  onnx  : edgetools/rife/rife49.onnx                            (RIFE v4.9)

Prefer ``huggingface_hub`` (handles LFS + auth/redirects) and fall back to
plain HTTP on the ``/resolve/main/`` URL.
"""

import os
import shutil

TORCH_REPO = "Anson-Saju-George/wms-rifev3"
TORCH_FILE = "train_log/flownet.pkl"
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


def _download_hf(repo, filename, cache_dir, dest, force=False):
    import huggingface_hub as hfh
    src = hfh.hf_hub_download(
        repo_id=repo, repo_type="model", filename=filename,
        cache_dir=cache_dir, force_download=force,
    )
    if not os.path.exists(dest) or os.path.getsize(dest) == 0:
        shutil.copy(src, dest)
    return dest


def fetch(cache_dir="./models", want=None, download=True, verbose=True):
    """Ensure weights are present and return (kind, path).

    kind in {'torch', 'onnx'}; path is the flownet.pkl file (torch) or the
    .onnx file (onnx).

    want in {'torch', 'onnx', None} (None => torch if available else onnx).
    """
    if not download:
        torch_path = os.path.join(cache_dir, TORCH_FILE.replace('/', '_').split('/')[-1])
        onnx_path = os.path.join(cache_dir, "rife49.onnx")
        if want in (None, "torch") and os.path.exists(torch_path) and os.path.getsize(torch_path) > 0:
            return "torch", torch_path
        if want in (None, "onnx") and os.path.exists(onnx_path) and os.path.getsize(onnx_path) > 0:
            return "kind", onnx_path
        raise FileNotFoundError("No weights found; pass --download or --model/--onnx")

    os.makedirs(cache_dir, exist_ok=True)
    torch_name = TORCH_FILE.replace('/', '_')  # train_log_flownet.pkl
    torch_path = os.path.join(cache_dir, torch_name)
    onnx_path = os.path.join(cache_dir, ONNX_FILE)

    def prefer(kind, path):
        if want in (None, kind) and os.path.exists(path) and os.path.getsize(path) > 0:
            return kind, path
        return None

    # torch first (best on ROCm), fall back to onnx
    if want in (None, "torch"):  # want torch or unspecified
        try:
            if not (os.path.exists(torch_path) and os.path.getsize(torch_path) > 0):
                if verbose:
                    print(f"[weights] downloading torch checkpoint -> {torch_path}")
                try:
                    _download_hf(TORCH_REPO, TORCH_FILE, cache_dir, torch_path)
                except Exception as e:
                    if verbose:
                        print(f"[weights] huggingface_hub failed ({e.__class__.__name__}); falling back to HTTP")
                if not (os.path.exists(torch_path) and os.path.getsize(torch_path) > 0):
                    if verbose:
                        print(f"[weights] HTTP download -> {torch_path}")
                    _download_url(_hf_url(TORCH_REPO, TORCH_FILE), torch_path)
            res = prefer("torch", torch_path)
            if res:
                return res
        except Exception as e:
            if verbose:
                print(f"[weights] torch download failed: {e}")
    if want in (None, "onnx"):
        if not (os.path.exists(onnx_path) and os.path.getsize(onnx_path) > 0):
            if verbose:
                print(f"[weights] downloading ONNX -> {onnx_path}")
            try:
                _download_hf(ONNX_REPO, ONNX_FILE, cache_dir, onnx_path)
            except Exception as e:
                if verbose:
                    print(f"[weights] huggingface_hub failed ({e.__class__.__name__}); falling back to HTTP")
            if not (os.path.exists(onnx_path) and os.path.getsize(onnx_path) > 0):
                if verbose:
                    print(f"[weights] HTTP download -> {onnx_path}")
                _download_url(_hf_url(ONNX_REPO, ONNX_FILE), onnx_path)
        res = prefer("onnx", onnx_path)
        if res:
            return res
    raise FileNotFoundError("Could not download any model weights.")
