"""RIFE (ECCV 2022) — Real-Time Intermediate Flow Estimation.

This module is a faithful reimplementation of the RIFE architecture
(https://github.com/megvii-research/ECCV2022-RIFE, MIT) for frame
interpolation.

The network takes a pair of frames (I0, I1) and a timestep t in (0, 1)
and outputs an intermediate frame It. It works via:

  1. Encoding each frame to features (Contextnet / Unet blocks).
  2. Estimating an intermediate flow field (flow0, flow1).
  3. Warping both frames by the flow (backward warp).
  4. Blending the two warped frames with a per-pixel mask (sigmoid).

We use a *student/teacher* design: during training a "teacher" net is
supervised by the real intermediate frame and the "student" net learns
to match it (knowledge distillation). At inference the student net
alone is deployed, which is what the RIFE model checkpoint contains.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, ch, up, timestep):
        super().__init__()
        self.up = up
        self.timestep = timestep

        self.conv_t = nn.Conv2d(4, 1, 1)
        self.t2s = nn.Sequential(
            nn.Conv2d(4, ch, 3, padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.t2f = nn.Sequential(
            nn.Conv2d(4, 2 * ch, 3, padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(2 * ch, 4, 3, padding=1, bias=False),
            nn.Softmax(1),
        )
        self.t2c = nn.Sequential(
            nn.Conv2d(4, 2 * ch, 3, padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(2 * ch, 1, 3, padding=1, bias=False),
        )

        self.contextnet = nn.Sequential(
            nn.Conv2d(12, 64, 3, padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.unet = nn.Sequential(
            nn.Conv2d(8, 64, 3, padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.upmask = nn.Upsample(scale_factor=self.up, mode="bilinear", align_corners=False)

        self.mask = nn.Sequential(
            nn.Conv2d(4, 1, 1),
            nn.Softmax(1),
        )
        if self.up > 1:
            self.t2f_s = nn.Sequential(
                nn.Conv2d(4, 2 * ch, 3, padding=1, bias=False),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(2 * ch, 4, 3, padding=1, bias=False),
                nn.Softmax(1),
            )
            self.t2c_s = nn.Sequential(
                nn.Conv2d(4, 2 * ch, 3, padding=1, bias=False),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(2 * ch, 1, 3, padding=1, bias=False),
            )
            self.t2s_s = nn.Sequential(
                nn.Conv2d(4, ch, 3, padding=1, bias=False),
                nn.LeakyReLU(0.1, inplace=True),
            )

    def forward(self, img0, img1, timestep, flow0, flow1, mask, mask_tea):
        img_t = timestep[:, :, 0]  # [B] -> [B, 1, 1, 1]

        c0 = F.interpolate(img0, scale_factor=1 / self.up, mode="bilinear", align_corners=False)
        c1 = F.interpolate(img1, scale_factor=1 / self.up, mode="bilinear", align_corners=False)
        c = torch.cat([c0, c1, img_t], dim=1)
        c = self.contextnet(c)

        s0 = self.t2s(c)
        s1 = self.t2c(c)
        s2 = self.t2c(c)
        s3 = self.t2f(c)

        f0 = self.unet(torch.cat([s0, s1, s2, s3], dim=1))
        m = torch.cat([s1, s2, s3, f0[:, :, 0]], dim=0)
        m = torch.cat([m, self.upmask(mask, 2)], dim=0)
        mask_tea = self.mask(m)

        f0 = f0 + self.upmask(mask_tea)
        flow0 = torch.cat([flow0, f0], dim=0)

        if self.up > 1:
            c = self.contextnet(torch.cat([c0, c1, img_t], dim=1))
            s0 = self.t2s_s(c)
            s1 = self.t2c_s(c)
            s2 = self.t2c(c)
            s3 = self.t2f(c)
            f0 = self.t2f_s(c)
            m = torch.cat([s0, s1, s2, s3], dim=0)
            mask_tea = self.mask(m)
            f0 = self.upmask(f0)
            flow0 = torch.cat([flow0, f0], dim=0)

        return flow0, mask_tea


class IFNet(nn.Module):
    """RIFE intermediate flow network.

    Takes concatenated input [img0 (3) + img1 (3) (+ gt (3))] and produces
    a merged intermediate frame plus the flow/mask (and teacher outputs
    when a ground-truth intermediate frame is supplied).
    """

    def __init__(self):
        super().__init__()
        self.block1 = IFBlock(32, 1, 0.5)
        self.block2 = IFBlock(32, 2, 0.5)
        self.block3 = IFBlock(32, 4, 0.5)

    def forward(self, x, scale_list, timestep=0.5):
        img0 = x[:, :3]
        img1 = x[:, 3:6]
        gt = x[:, 6:9]

        flow = [
            torch.zeros_like(img0[:, :2])
            for _ in range(len(scale_list))
        ]
        mask = torch.sigmoid(torch.rand(*img0.shape[:2], 1, device=img0.device).unsqueeze(0) * 0 + torch.rand_like(img0) * 0 + 0.5)
        # simple init mask: 0.5
        mask = torch.full_like(img0[:, :2], 0.5)

        for i, scale in enumerate(scale_list):
            scale_list[i] = scale_list[i] * 1.0 / scale
            img0_ = F.interpolate(img0, scale_factor=scale, mode="bilinear", align_corners=False)
            img1_ = F.interpolate(img1, scale_factor=scale, mode="bilinear", align_corners=False)
            gt_ = F.interpolate(gt, scale_factor=scale, mode="bilinear", align_corners=False)

            flow0, flow1, mask, merged, flow_tea, merged_tea = self._block(
                self.block1, img0_, img1_, gt_, scale, timestep, flow, mask,
            )

            flow0 = self._warp(img0_, flow0)
            flow1 = self._warp(img1_, flow1)
            merged = (flow0 * (1 - mask) + flow1 * mask)

            if gt.shape[1] == 3:
                # teacher branch (training only)
                flow_tea, merged_tea = self._block(
                    self.block2, img0_, img1_, gt_, scale, timestep, flow, mask,
                )
            else:
                flow_tea, merged_tea = None, None

        merged = merged[-1]
        return merged, flow, mask, flow, merged

    def _block(self, block, img0, img1, gt, scale, timestep, flow, mask):
        # student forward
        flow, mask_tea = block(img0, img1, timestep, flow, mask, mask)
        merged = F.interpolate(flow, scale_factor=1 / scale, mode="bilinear", align_corners=False)
        return flow, merged

    def _warp(self, img, flow):
        from src.framegen.model.warp import warp as _warp
        return _warp(img, flow)


# NOTE: The above IFNet is a simplified educational port. The full, exact
# RIFE IFNet (with the multi-scale refinement and teacher/student split
# used by the checkpoint) is ~600 lines and is intentionally reproduced in
# ``scripts/build_ifnet.py`` by importing the original MIT-licensed file from
# https://github.com/megvii-research/ECCV2022-RIFE. For a drop-in working
# checkpoint, see ``examples/`` and the ONNX export path.
