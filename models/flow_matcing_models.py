"""
Flow matching model definitions.
"""

import torch
import torch.nn as nn
from models.denoisers import *


class FlowMatchingClassUncond(nn.Module):
    def __init__(
        self,
        denoiser: TimeConditionalUNet,
        device: torch.device,
        num_ts: int = 50,
        img_hw: tuple[int, int] = (28, 28),
    ):
        
        super().__init__()
        self.denoiser = denoiser
        self.num_ts = num_ts
        self.img_hw = img_hw
        self.device = device


    def forward(self, x_1: torch.Tensor) -> torch.Tensor:
        """Outputs predicted deviation from true data for random initial noises and timesteps.

        Randomly samples initial noises and timesteps, and produces intermediate
        noises by taking convex combinations of the initial noises and correspoinding
        sample images in x. Outputs predicted deviations from the sample image for
        each intermediate noise. 
        
        Args:
            x_1: (N, C, H, W) input tensor of sample images.

        Returns:
            Predicted deviations from sample images.
        """

        self.denoiser.train()
        x_0 = torch.randn_like(x_1)
        x_0 = x_0.to(self.device)
        target = x_1 - x_0
        t = torch.rand((x_0.shape[0],))
        t = t.to(self.device)
        t_sample = t.view(*t.shape, 1, 1, 1)
        x_t = x_0 + t_sample*(x_1 - x_0)
        pred_deviation = self.denoiser(x_t, t)
        return ((pred_deviation - target)**2).mean()


    @torch.inference_mode()
    def sample(
        self,
        img_wh: tuple[int, int],
        num_samples,
        seed: int = 0,
    ):
        """Generates novel images using learned denoiser.
        
        Randomly samples num_samples initial noises of size given by img_wh (based on seed, if given)
        and iteratively denoises them according to flow matching denoising procedure. Returns loss
        between predicted deviation from true image and actual deviation for each intermediate noise.

        Args:
            img_wh: (H, W) output image width and height
            num_samples: integer number of images to generate
            seed: int, random seed
        """

        self.denoiser.eval()
        if seed:
            torch.manual_seed(seed)
        x_t = torch.randn((num_samples, 1, img_wh[0], img_wh[1]))
        x_t = x_t.to(self.device)
        step_size = 1 / self.num_ts
        t = torch.zeros((num_samples,))
        t = t.to(self.device)
        while t[0].item() < 1:
            x_t = x_t + step_size * self.denoiser(x_t, t)
            t += step_size
        return x_t


class FlowMatchingClassCond(nn.Module):
    def __init__(
        self,
        denoiser: ClassConditionalUNet,
        device: torch.device,
        p_uncond: float,
        num_ts: int = 50,
        img_hw: tuple[int, int] = (28, 28),
    ):

        super().__init__()
        self.denoiser = denoiser
        self.num_ts = num_ts
        self.img_hw = img_hw
        self.device = device
        self.p_uncond = p_uncond

    def forward(self,
                x_1: torch.Tensor,
                c: torch.Tensor,
                ) -> torch.Tensor:
        
        """Outputs error between true and predicted deviations from data for random initial noises and timesteps.

        Puts denoiser in training mode. Then randomly samples initial noises and timesteps,
        and produces intermediate noises by taking convex combinations of the initial noises
        and correspoinding sample images in x. Outputs loss between predicted deviations from
        the sample image and the true deviation for each intermediate noise. 
        
        Args:
            x_1: (N, C, H, W) input tensor of sample images.

        Returns:
            Predicted deviations from sample images.
        """
        
        # Put denoiser in train mode
        self.denoiser.train()
        
        # Randomly sample intermediate noise
        x_0 = torch.randn_like(x_1).to(self.device)
        target = x_1 - x_0
        t = torch.rand((x_0.shape[0],)).to(self.device)
        x_t = x_0 + t.view(*t.shape, 1, 1, 1)*(x_1 - x_0)

        # Apply mask to true classes with probability p_uncond
        mask = torch.bernoulli((1-self.p_uncond) * torch.ones((c.shape[0],)))
        mask = mask.to(self.device)
        
        # Compute squared error between true and predicted deviation from sample image 
        output = self.denoiser(x_t, c, t, mask)
        return ((target - output)**2).mean()


    @torch.inference_mode()
    def sample(
        self,
        c: torch.Tensor,
        img_wh: tuple[int, int],
        num_samples: int,
        guidance_scale: float = 5.0,
        seed: int = 0,
    ):
        
        """Uses learned flow matching model to produce images of a specified class.
        
        Randomly samples num_samples initial noises of size given by img_wh (based on seed, if given)
        and iteratively denoises them according to flow matching denoising procedure, guided by the desired
        classes and the specified guidance scale. Returns denoised images.

        Args:
            c: tensor of desired classes of produced images
            img_wh: (H, W) output image width and height
            num_samples: integer number of images to generate
            guidance_scale: guidance scale for classifier-free guidance
            seed: int, random seed
        """

        self.denoiser.eval()
        if seed:
            torch.manual_seed(seed)
        x_t = torch.randn((num_samples, 1, img_wh[0], img_wh[1]))
        x_t = x_t.to(self.device)
        step_size = 1 / self.num_ts
        t = torch.zeros((num_samples,))
        t = t.to(self.device)
        c = c.to(self.device)
        while t[0].item() < 1:
            mask_uncond = torch.zeros((c.shape[0],)).to(self.device)
            mask_cond = torch.ones((c.shape[0],)).to(self.device)
            u_uncond = self.denoiser(x_t, c, t, mask_uncond)
            u_cond = self.denoiser(x_t, c, t, mask_cond)
            u = u_uncond + guidance_scale * (u_cond - u_uncond)
            x_t = x_t + step_size * u
            t += step_size
        return x_t



