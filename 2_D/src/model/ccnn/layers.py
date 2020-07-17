""" Layers of the CCNN model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpDouble(nn.Module):
    """Upscaling two images merge them, then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2, skip1, skip2):
        """
        x1 and x2 should have the same size
        skip1 and skip1 should have the same size

        :param x1: Input from left
        :param x2: Input from right
        :param skip1: Skip connection from the left
        :param skip2: Skip connection from the right
        :return:
        """
        x1 = self.up(x1)
        x2 = self.up(x2)
        # input is CHW
        diff_y = skip1.size()[2] - x1.size()[2]
        diff_x = skip1.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])
        x2 = F.pad(x2, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([skip1, x1, x2, skip2], dim=1)
        return self.conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, skip_connections=True):
        super().__init__()
        self.skip_connections = skip_connections

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, skip1=None, skip2=None):
        """
        x2 and x3 should have the same size

        :param x1: Input from previous layer
        :param skip1: Skip connection from the left
        :param skip2: Skip connection from the right
        :return:
        """
        x1 = self.up(x1)
        if self.skip_connections:
            # input is CHW
            diff_y = skip1.size()[2] - x1.size()[2]
            diff_x = skip1.size()[3] - x1.size()[3]

            x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

            x1 = torch.cat([skip1, x1, skip2], dim=1)
        return self.conv(x1)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.out = nn.Tanh()

    def forward(self, x):
        x = self.conv(x)
        return self.out(x)
