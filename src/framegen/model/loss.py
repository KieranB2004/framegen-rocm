import torch
import torch.nn as nn
import torch.nn.functional as F


class EPE:
    """Endpoint Error (for flow)."""

    def __init__(self, max_displacement=100):
        self.max_displacement = max_displacement

    def forward(self, pred, gt, mask=None):
        if mask is None:
            return 0.0
        disp_pred = pred[:, :2] - pred[:, 2:4]
        disp_gt = gt[:, :2] - gt[:, 2:4]
        err = torch.abs(disp_pred - disp_gt)
        if mask is not None and mask.shape[-1] == 1:
            mask = mask.squeeze(-1)
        mask = mask.detach()
        valid = mask > 0.99
        if valid.sum() > 0:
            return err[valid].mean()
        return torch.tensor(0.0, device=pred.device)


class SOBEL:
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        sobel = self._sobel(pred)
        sobel_t = self._sobel(target)
        return F.l1_loss(sobel, sobel_t)

    @staticmethod
    def _sobel(img):
        kx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).unsqueeze(0)
        ky = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).unsqueeze(0)
        f = F.conv2d(img, kx.repeat(3, 1, 1, 1), groups=img.shape[1], padding=1)
        f = F.conv2d(img, ky.repeat(3, 1, 1, 1), groups=img.shape[1], padding=1)
        return torch.cat([f, f], dim=1)
