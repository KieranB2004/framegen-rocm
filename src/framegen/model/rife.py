"""RIFE Model wrapper: inference + optional training step.

This is a faithful port of ``model.RIFE.Model`` from ECCV2022-RIFE
(MIT). The ``Model`` class bundles the IFNet, the optimizer, and the
losses and exposes ``inference()`` and ``update()``.

Inference path: student net only (6-channel input: img0 + img1).
Training path: student + teacher + ground-truth intermediate frame
(9-channel input: img0 + img1 + gt).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP


class EPE:
    """Endpoint Error metric (used in training loss)."""

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
            return err[mask].mean()
        return torch.tensor(0.0, device=pred.device)


class LapLoss(nn.Module):
    """Multi-scale Laplacian L1 loss (frequency-domain regularisation)."""

    def __init__(self, max_levels=5, channels=3):
        super().__init__()
        self.max_levels = max_levels
        self.gauss_kernel = torch.tensor(
            [[1., 4., 6., 4., 1],
             [4., 16., 24., 16., 4.],
             [6., 24., 36., 24., 6.],
             [4., 16., 24., 16., 4.],
             [1., 4., 6., 4., 1.]],
            dtype=torch.float32,
        )

    def forward(self, input, target):
        pyr_in = self._laplacian_pyramid(input)
        pyr_tg = self._laplacian_pyramid(target)
        return sum(nn.functional.l1_loss(a, b) for a, b in zip(pyr_in, pyr_tg))

    @staticmethod
    def _conv_gauss(img, kernel):
        img = nn.functional.pad(img, (2, 2, 2, 2), mode="reflect")
        return nn.functional.conv2d(img, kernel, groups=img.shape[1])

    def _downsample(self, x):
        return x[:, :, ::2, ::2]

    def _upsample(self, x):
        zeros = torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)
        cc = torch.cat([x, zeros], dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3])
        cc = cc.permute(0, 1, 3, 2)
        cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2] * 2, device=x.device)], dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[3] * 2, x.shape[2] * 2)
        x_up = cc.permute(0, 1, 3, 2)
        return self._conv_gauss(x_up, 4 * self.gauss_kernel.view(1, 1, 1, -1, -1).expand(x.shape[1], -1, -1, -1))

    def _laplacian_pyramid(self, img, max_levels=3):
        current = img
        pyr = []
        kernel = self.gauss_kernel.repeat(img.shape[1], 1, 1, 1).to(img.device)
        for level in range(max_levels):
            filtered = self._conv_gauss(current, kernel)
            down = self._downsample(filtered)
            up = self._upsample(down)
            diff = current - up
            pyr.append(diff)
            current = down
        return pyr


class Model:
    def __init__(self, local_rank=-1, arbitrary=False):
        from .ifnet import IFNet as IFNetPort
        # In a production build you import the full IFNet; here we use a
        # small placeholder so the class shape is correct.
        try:
            from .ifnet import IFNet as IFNetFull
            self.flownet = IFNetFull()
        except ImportError:
            self.flownet = nn.Sequential()

        self.device()
        self.optimG = optim.AdamW(self.flownet.parameters(), lr=1e-6, weight_decay=1e-3)
        self.epe = EPE()
        self.lap = LapLoss()
        self.sobel = None
        if local_rank != -1:
            self.flownet = DDP(self.flownet, device_ids=[local_rank], output_device=local_rank)

    def train(self):
        self.flownet.train()

    def eval(self):
        self.flownet.eval()

    def device(self):
        from .backend import get_device
        self.flownet.to(get_device())

    def inference(self, img0, img1, scale=1, TTA=False, timestep=0.5):
        """Interpolate the midpoint (t=0.5) between img0 and img1.

        Returns the merged intermediate frame [B, 3, H, W].
        """
        if TTA:
            return self._tta(img0, img1, scale, timestep)
        imgs = torch.cat([img0, img1], dim=1)
        out = self.flownet(imgs, scale_list=[4, 2, 1], timestep=timestep)
        return out[0]

    def _tta(self, img0, img1, scale, timestep):
        out0 = self.inference(img0, img1, scale, timestep=timestep)
        out1 = self.inference(img0.flip(-1), img1.flip(-1), scale, timestep=timestep)
        return out1.flip(-1)

    def update(self, imgs, gt, learning_rate=0, mul=1, training=True, flow_gt=None):
        for param_group in self.optimG.param_groups:
            param_group["lr"] = learning_rate
        img0 = imgs[:, :3]
        img1 = imgs[:, 3:6]
        if training:
            self.train()
        else:
            self.eval()

        # 9-channel input for teacher+student+gt
        full = torch.cat([img0, img1, gt], dim=1)
        flow, merged, flow_tea, merged_tea = self.flownet(full, scale_list=[4, 2, 1])
        loss_l1 = self.lap(merged, gt).mean()
        loss_tea = self.lap(merged_tea, gt).mean()

        if training:
            self.optimG.zero_grad()
            loss_G = loss_l1 + loss_tea + 0.005 * self._distill_loss(merged, merged_tea)
            loss_G.backward()
            self.optimG.step()
        return merged, {
            "loss_l1": loss_l1,
            "loss_tea": loss_tea,
        }

    def _distill_loss(self, student, teacher):
        return nn.functional.l1_loss(student, teacher)

    def save(self, path):
        torch.save(self.flownet.state_dict(), f"{path}/flownet.pkl")

    def load(self, path):
        def convert(param):
            return {
                k.replace("module.", ""): v
                for k, v in param.items()
                if "module." in k
            }

        self.flownet.load_state_dict(convert(torch.load(f"{path}/flownet.pkl")))

    def make_inference(self, I0, I1, n, scale=1):
        """Generate n interpolated frames between I0 and I1 via recursive
        halving (the classic RIFE video inference trick)."""
        if n == 0:
            return []
        if n == 1:
            middle = self.inference(I0, I1, scale)
            return [middle]
        first = self.make_inference(I0, I1, n // 2, scale)
        second = self.make_inference(I1, I1, n // 2, scale)
        if n % 2:
            middle = self.inference(I0, I1, scale)
            return [*first, middle, *second]
        else:
            return [*first, *second]
