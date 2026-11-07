import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import itertools

from .warp import warp
from .ifnet import *  # IFNet, IFBlock


class Model:
    """RIFE Model — bundles the IFNet, the optimizer, and the metrics.

    This mirrors ``model.RIFE.Model`` from ECCV2022-RIFE (MIT).

    Attributes
    ----------
    flownet : IFNet
    optimG  : torch.optim.AdamW
    """

    def __init__(self, local_rank=-1, arbitrary=False):
        self.flownet = IFNet()
        self.device()
        self.optimG = optim.AdamW(self.flownet.parameters(), lr=1e-6, weight_decay=1e-3)
        self.epe = EPE()
        self.lap = LapLoss()
        self.sobel = SOBEL()
        if local_rank != -1:
            self.flownet = torch.nn.parallel.DistributedDataParallel(
                self.flownet, device_ids=[local_rank], output_device=local_rank
            )

    def train(self):
        self.flownet.train()

    def eval(self):
        self.flownet.eval()

    def device(self):
        from .backend import get_device
        self.flownet.to(get_device())

    def inference(self, img0, img1, scale=1, TTA=False, timestep=0.5):
        """Interpolate the frame at `timestep` between img0 and img1.

        Parameters
        ----------
        img0, img1 : torch.Tensor [B, 3, H, W] in [0, 1]
        scale      : float (1.0 = full res; 0.5 = half res for 4K)

        Returns
        -------
        merged : torch.Tensor [B, 3, H, W] the intermediate frame
        """
        imgs = torch.cat((img0, img1), 1)
        flow, mask, merged, flow_teacher, merged_teacher, loss_distill = self.flownet(imgs, scale_list=[4, 2, 1], timestep=timestep)
        if TTA:
            flow2, mask2, merged2, flow_tea2, merged_tea2, loss_distill2 = self.flownet(imgs.flip(2).flip(3), scale_list=[4, 2, 1], timestep=timestep)
            return (merged[2] + merged2[2].flip(2).flip(3)) / 2
        else:
            return merged[2]

    def make_inference(self, I0, I1, n, scale=1):
        """Generate `n` frames between I0 and I1 via the classic recursive halving.

        This is the classic RIFE video inference trick: to get `n` frames we
        compute the midpoint, then recursively interpolate each half, which
        keeps flow estimation local and stable. Returns a list of `n` frames.

        n=1 -> [m]
        n=2 -> [m01, m12]   (two frames, splitting the interval in half)
        n=3 -> [m01, mid, m12]
        """
        if n <= 0:
            return []
        middle = self.inference(I0, I1, scale)
        if n == 1:
            return [middle]
        first_half = self.make_inference(I0, middle, n // 2, scale)
        second_half = self.make_inference(middle, I1, n // 2, scale)
        if n % 2:
            return [*first_half, middle, *second_half]
        else:
            return [*first_half, *second_half]

    def update(self, imgs, gt, learning_rate=0, mul=1, training=True, flow_gt=None):
        """One training step: student + teacher + distillation.

        ``imgs`` : [B, 6, H, W] = img0 + img1
        ``gt``   : [B, 3, H, W] = true intermediate frame
        """
        for param_group in self.optimG.param_groups:
            param_group['lr'] = learning_rate
        if training:
            self.flownet.train()
        else:
            self.flownet.eval()

        full = torch.cat([imgs, gt], 1)
        flow, mask, merged, flow_teacher, merged_teacher, loss_distill = self.flownet(full, scale_list=[4, 2, 1])
        loss_l1 = self.lap(merged[2], gt).mean()
        loss_tea = self.lap(merged_teacher, gt).mean()

        if training:
            self.optimG.zero_grad()
            loss_G = loss_l1 + loss_tea + loss_distill * 0.01
            loss_G.backward()
            self.optimG.step()
        return merged[2], {
            'merged_tea': merged_teacher,
            'mask': mask,
            'mask_tea': mask,
            'flow': flow[2][:, :2],
            'flow_tea': flow_teacher,
            'loss_l1': loss_l1,
            'loss_tea': loss_tea,
            'loss_distill': loss_distill,
        }

    def save(self, path):
        torch.save(self.flownet.state_dict(), f"{path}/flownet.pkl")

    def load(self, path):
        def convert(param):
            return {k.replace("module.", ""): v for k, v in param.items()}
        self.flownet.load_state_dict(convert(torch.load(f"{path}/flownet.pkl", map_location="cpu")))