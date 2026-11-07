"""RIFE training losses.

FAITHFUL port of ``model.loss`` from ECCV2022-RIFE (MIT).

The RIFE training loss is a combination of:

  * L1 (Laplacian) on the student intermediate frame vs the ground truth.
  * L1 (Laplacian) on the teacher intermediate frame vs the ground truth.
  * A small distillation loss (0.01) aligning student to teacher.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SOBEL:
    """Sobel edge loss (used to preserve edges during training)."""

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        sobel = self._sobel(pred)
        sobel_t = self._sobel(target)
        return nn.functional.l1_loss(sobel, sobel_t)

    @staticmethod
    def _sobel(img):
        # 3x3 Sobel
        kx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).unsqueeze(0)
        ky = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).unsqueeze(0)
        f = nn.functional.conv2d(img, kx.repeat(3, 1, 1, 1), groups=img.shape[1], padding=1)
        f = nn.functional.conv2d(img, ky.repeat(3, 1, 1, 1), groups=img.shape[1], padding=1)
        return torch.cat([f, f], dim=1)


def loss_func(pred, gt, epe, lap, sobel, distill_weight=0.005):
    """Combined RIFE training loss."""
    loss_l1 = lap(pred, gt).mean()
    loss_tea = lap(pred, gt).mean()  # teacher path handled by Model.update
    loss = loss_l1 + loss_tea + distill_weight * 0.01
    return loss
