"""
Contains simple operations for constructing UNet architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=3,
                              stride=1,
                              padding=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return F.gelu(x)


class DownConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.downconv = nn.Conv2d(in_channels,
                                  out_channels,
                                  kernel_size=3,
                                  stride=2,
                                  padding=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downconv(x)
        x = self.bn(x)
        return F.gelu(x)


class UpConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv_trans = nn.ConvTranspose2d(in_channels,
                                             out_channels,
                                             kernel_size=4,
                                             stride=2,
                                             padding=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_trans(x)
        x = self.bn(x)
        return F.gelu(x)


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()
        self.avg_pool = nn.AvgPool2d(7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.avg_pool(x))


class Unflatten(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv_trans = nn.ConvTranspose2d(in_channels,
                                             in_channels,
                                             kernel_size=7,
                                             stride=7,
                                             padding=0)
        self.bn = nn.BatchNorm2d(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_trans(x)
        x = self.bn(x)
        return F.gelu(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = Conv(in_channels, out_channels)
        self.conv2 = Conv(out_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        return self.conv2(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.down_conv = DownConv(in_channels, out_channels)
        self.conv_block = ConvBlock(out_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down_conv(x)
        return self.conv_block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up_conv = UpConv(in_channels, out_channels)
        self.conv_block = ConvBlock(out_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up_conv(x)
        return self.conv_block(x)

class FCBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.Linear1 = nn.Linear(in_channels, out_channels)
        self.Linear2 = nn.Linear(out_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 1:
          x = x.view(*x.shape, 1)
        x = F.gelu(self.Linear1(x))
        x = self.Linear2(x)
        x = x.view(*x.shape, 1, 1)
        return x
