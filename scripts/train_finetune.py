#!/usr/bin/env python
"""
Fine-tune the pretrained RIFE checkpoint on your own clip data (template).

Why this, not full training:
  Full RIFE pre-training needs the HD/UCF datasets and multi-GPU. On a single
  6700 XT the realistic path is to take a pretrained checkpoint and
  fine-tune it on task-relevant clips (your own footage / a target domain).

Expected input format (simplest): a folder of .npy triplets.
  Each file is np.save([I0, I1, GT]) where I0, I1, GT are uint8 HxWx3
  (BGR) and GT is the *true* middle frame between I0 and I1 (you can produce
  one by rendering a high-fps clip and taking every-other frame as I0, I1 and
  the skipped one as GT).

Usage:
  python scripts/train_finetune.py --data ./data --epochs 1 --batch 2 --lr 1e-5 --save ./out

Notes:
  * --fp16 uses half precision (faster; verify numerics).
  * This is a TEMPLATE. Adjust the data loader to your format.
"""

import argparse
import os
import numpy as np

import torch

from framegen.backend import get_device
from framegen.model import Model
from framegen.weights import fetch


def to_tensor(arr, dev):
    return torch.from_numpy(arr.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(dev)


def main():
    ap = argparse.ArgumentParser(description="Fine-tune RIFE on a folder of (I0, I1, GT) triplets")
    ap.add_argument("--data", required=True, help="folder of .npy triplets (I0, I1, GT) uint8")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--model", default=None, help="dir with flownet.pkl (default: auto-download)")
    ap.add_argument("--save", default="models/fine-tuned")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--fp16", action="store_true")
    args = ap.parse_args()

    dev = get_device(args.device)
    model = Model()
    if args.model:
        model.load(args.model)
    else:
        _, pkl = fetch(want="torch", download=True, verbose=True)
        model.load(os.path.dirname(pkl))
    model.to(dev)
    if args.fp16 and dev.type != "cpu":
        model.flownet.half()

    files = sorted(f for f in os.listdir(args.data) if f.endswith(".npy"))
    if not files:
        raise SystemExit(f"No .npy triplets in {args.data}. Put (I0, I1, GT) uint8 .npy files there.")
    triples = []
    for f in files:
        arr = np.load(os.path.join(args.data, f), allow_pickle=False)
        I0, I1, GT = arr[0], arr[1], arr[2]
        triples.append((I0, I1, GT))
    print(f"Loaded {len(triples)} triplets on {dev}")

    for ep in range(1, args.epochs + 1):
        model.train()
        tot = 0.0
        last = None
        for i in range(0, len(triples), args.batch):
            bs = triples[i:i + args.batch]
            imgs = torch.stack([torch.cat([to_tensor(x[0], dev), to_tensor(x[1], dev)], dim=1)
                                for x in bs], dim=0)
            gt = torch.stack([to_tensor(x[2], dev) for x in bs], dim=0)
            out, metrics = model.update(imgs, gt, learning_rate=args.lr, training=True)
            tot += metrics["loss_l1"].item() * args.batch
            last = metrics["loss_tea"].item()
        print(f"epoch {ep}: mean loss_l1={tot/len(triples):.4f}  loss_tea={last:.4f}")

    os.makedirs(args.save, exist_ok=True)
    model.save(args.save)
    print(f"Saved fine-tuned checkpoint to {args.save}/flownet.pkl")


if __name__ == "__main__":
    main()
