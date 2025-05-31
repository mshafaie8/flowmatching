"""
Method for sampling from a flow matching model.
"""

import torch
import torch.nn as nn


@torch.inference_mode()
def fm_sample(
    unet: nn.Module,
    img_wh: tuple[int, int],
    num_ts: int,
    num_samples: int,
    device: torch.device,
    seed: int = 0,
) -> torch.Tensor:
    """Algorithm 2

    Args:
        unet: TimeConditionalUNet
        img_wh: (H, W) output image width and height.
        num_ts: int, number of timesteps.
        seed: int, random seed.

    Returns:
        (N, C, H, W) final sample.
    """
    unet.eval()
    x_t = torch.randn((num_samples, 1, img_wh[0], img_wh[1]))
    x_t = x_t.to(device)
    step_size = 1 / num_ts
    t = torch.zeros((num_samples,))
    t = t.to(device)
    while t[0].item() < 1:
      x_t = x_t + step_size * unet(x_t, t)
      t += step_size
    return x_t