import torch
import torch.nn as nn
import torch.nn.functional as F


def warp(img, flow):
    """Backward warp `img` by `flow` (grid_sample).

    `flow` has the same shape as `img` (N, C, H, W). Negative flow
    samples from the *next* frame and positive flow from the *previous*
    frame, which is the convention the RIFE paper uses.
    """
    if img.shape[1] == 3:
        # replicate channel for backward warp
        img = torch.cat([img, img], dim=1)
    return F.grid_sample(
        img,
        torch.stack([torch.linspace(0, 1, img.shape[2], device=img.device).unsqueeze(1).repeat(img.shape[2], 1)[:, :, None, :],
                     torch.linspace(0, 1, img.shape[3], device=img.device).unsqueeze(1).repeat(img.shape[3], 1)[:, :, None, :]],
        dim=1,
    ).flatten(1, 2),
    1,
)


def contextnet(x, up, timestep):
    """Context net: small U-Net that produces an intermediate flow
    for each input frame. This is the "teacher"-style network in RIFE.
    """
    # We use a simple 1-block context net as a reference implementation.
    # The real RIFE uses a richer context net; this is a faithful
    # single-block variant for educational purposes.
    up = up
    f0 = x[:, :3]
    f1 = x[:, 3:6]
    c0 = F.interpolate(f0, scale_factor=1 / up, mode='bilinear', align_corners=False)
    c1 = F.interpolate(f1, scale_factor=1 / up, mode='bilinear', align_corners=False)
    c = torch.cat([c0, c1, c0 * f0 * f1], dim=1)
    c = nn.Conv2d(12, 64, 3, padding=1)(c)
    c = nn.LeakyReLU(0.1, inplace=True)(c)
    c = nn.Conv2d(64, 32, 3, padding=1)(c)
    c = nn.LeakyReLU(0.1, inplace=True)(c)
    c = nn.Upsample(scale_factor=up, mode='bilinear', align_corners=False)(c)
    flow = nn.Conv2d(32, 2, 3, padding=1, bias=False)(c)
    # normalize flow to [-1, 1]
    flow = torch.tanh(flow)
    return flow


def unet(x, up, timestep):
    """Unet that refines the flow and mask. Reference implementation."""
    f0 = x[:, :3]
    f1 = x[:, 3:6]
    # downsample
    d0 = F.interpolate(f0, scale_factor=1 / up, mode='bilinear', align_corners=False)
    d1 = F.interpolate(f1, scale_factor=1 / up, mode='bilinear', align_corners=False)
    d = torch.cat([d0, d1], dim=1)
    d = nn.Conv2d(6, 128, 3, padding=1)(d)
    d = nn.LeakyReLU(0.1, inplace=True)(d)
    d = nn.Conv2d(128, 64, 3, padding=1)(d)
    d = nn.LeakyReLU(0.1, inplace=True)(d)
    # upsample
    d = nn.Upsample(scale_factor=up, mode='bilinear', align_corners=False)(d)
    flow = nn.Conv2d(64, 2, 3, padding=1, bias=False)(d)
    mask = nn.Conv2d(64, 1, 3, padding=1)(d)
    flow = torch.tanh(flow)
    mask = torch.sigmoid(mask)
    return flow, mask
