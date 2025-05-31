"""
Encodes various UNet architectures for performing denoising direction prediction in the flow matching denoising process.
"""


from models.blocks import *

class UnconditionalUNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_hiddens: int,
    ):
        super().__init__()
        self.conv_block_init = ConvBlock(in_channels, num_hiddens)
        self.down_block_1 = DownBlock(num_hiddens, num_hiddens)
        self.down_block_2 = DownBlock(num_hiddens, 2*num_hiddens)
        self.flatten = Flatten()
        self.unflatten = Unflatten(2*num_hiddens)
        self.upblock1 = UpBlock(4*num_hiddens, num_hiddens)
        self.upblock2 = UpBlock(2*num_hiddens, num_hiddens)
        self.conv_block_final = ConvBlock(2*num_hiddens, num_hiddens)
        self.conv_output = nn.Conv2d(num_hiddens,
                                       in_channels,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-2:] == (28, 28), "Expect input shape to be (28, 28)."
        down1 = self.conv_block_init(x)
        down2 = self.down_block_1(down1)
        down3 = self.down_block_2(down2)
        bottom = self.flatten(down3)
        up1 = torch.cat((down3, self.unflatten(bottom)), 1)
        up2 = torch.cat((down2, self.upblock1(up1)), 1)
        up3 = torch.cat((down1, self.upblock2(up2)), 1)
        return self.conv_output(self.conv_block_final(up3))
    

class TimeConditionalUNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_hiddens: int,
    ):
        super().__init__()
        self.conv_block_init = ConvBlock(in_channels, num_hiddens)
        self.down_block_1 = DownBlock(num_hiddens, num_hiddens)
        self.down_block_2 = DownBlock(num_hiddens, 2*num_hiddens)
        self.flatten = Flatten()
        self.unflatten = Unflatten(2*num_hiddens)
        self.upblock1 = UpBlock(4*num_hiddens, num_hiddens)
        self.upblock2 = UpBlock(2*num_hiddens, num_hiddens)
        self.conv_block_final = ConvBlock(2*num_hiddens, num_hiddens)
        self.conv_output = nn.Conv2d(num_hiddens,
                                       in_channels,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)
        self.fcblock1 = FCBlock(in_channels=1, out_channels=2*num_hiddens)
        self.fcblock2 = FCBlock(in_channels=1, out_channels=num_hiddens)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: (N, C, H, W) input tensor.
            t: (N,) normalized time tensor.

        Returns:
            (N, C, H, W) output tensor.
        """
        assert x.shape[-2:] == (28, 28), "Expect input shape to be (28, 28)."
        down1 = self.conv_block_init(x)
        down2 = self.down_block_1(down1)
        down3 = self.down_block_2(down2)
        bottom = self.flatten(down3)
        up1 = torch.cat((down3, self.unflatten(bottom) * self.fcblock1(t)), 1)
        up2 = torch.cat((down2, self.upblock1(up1) * self.fcblock2(t)), 1)
        up3 = torch.cat((down1, self.upblock2(up2)), 1)
        return self.conv_output(self.conv_block_final(up3))


class ClassConditionalUNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        num_hiddens: int
    ):
        super().__init__()
        self.num_classes = num_classes
        self.conv_block_init = ConvBlock(in_channels, num_hiddens)
        self.down_block_1 = DownBlock(num_hiddens, num_hiddens)
        self.down_block_2 = DownBlock(num_hiddens, 2*num_hiddens)
        self.flatten = Flatten()
        self.unflatten = Unflatten(2*num_hiddens)
        self.upblock1 = UpBlock(4*num_hiddens, num_hiddens)
        self.upblock2 = UpBlock(2*num_hiddens, num_hiddens)
        self.conv_block_final = ConvBlock(2*num_hiddens, num_hiddens)
        self.conv_output = nn.Conv2d(num_hiddens,
                                       in_channels,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)
        self.fcblockt1 = FCBlock(in_channels=1, out_channels=2*num_hiddens)
        self.fcblockc1 = FCBlock(in_channels=num_classes, out_channels=2*num_hiddens)
        self.fcblockt2 = FCBlock(in_channels=1, out_channels=num_hiddens)
        self.fcblockc2 = FCBlock(in_channels=num_classes, out_channels=num_hiddens)


    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        t: torch.Tensor,
        mask:torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (N, C, H, W) input tensor.
            c: (N,) int64 condition tensor.
            t: (N,) normalized time tensor.
            mask: (N,) mask tensor. If not None, mask out condition when mask == 0.

        Returns:
            (N, C, H, W) output tensor.
        """
        assert x.shape[-2:] == (28, 28), "Expect input shape to be (28, 28)."
        down1 = self.conv_block_init(x)
        down2 = self.down_block_1(down1)
        down3 = self.down_block_2(down2)
        bottom = self.flatten(down3)
        c_onehot = F.one_hot(c, num_classes=self.num_classes) * mask.view(*mask.shape, 1)
        up1 = torch.cat((down3, self.unflatten(bottom) * self.fcblockc1(c_onehot) + self.fcblockt1(t)), 1)
        up2 = torch.cat((down2, self.upblock1(up1) * self.fcblockc2(c_onehot) + self.fcblockt2(t)), 1)
        up3 = torch.cat((down1, self.upblock2(up2)), 1)
        return self.conv_output(self.conv_block_final(up3))
    


