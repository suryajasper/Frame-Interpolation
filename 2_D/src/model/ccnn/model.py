""" CCNN Network Architecture """

from torch import nn
from torch.nn import functional as F
import torch

from src.model.ccnn.layers import DoubleConv, Up, Down, OutConv, UpDouble
from src.model.flownet.model import get_flow_net


class CCNN(nn.Module):
    def __init__(self, n_channels=3, bilinear=True):
        super(CCNN, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(self.n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = UpDouble(2048, 256, bilinear)
        self.up2 = Up(768, 128, bilinear)
        self.up3 = Up(384, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.out = OutConv(64, self.n_channels)

    def forward(self, frame1, frame2):
        frame1_1 = self.inc(frame1)
        frame1_2 = self.down1(frame1_1)
        frame1_3 = self.down2(frame1_2)
        frame1_4 = self.down3(frame1_3)
        frame1_5 = self.down4(frame1_4)
        # Right down
        frame2_1 = self.inc(frame2)
        frame2_2 = self.down1(frame2_1)
        frame2_3 = self.down2(frame2_2)
        frame2_4 = self.down3(frame2_3)
        frame2_5 = self.down4(frame2_4)
        # Up
        x = self.up1(frame1_5, frame2_5, frame1_4, frame2_4)
        x = self.up2(x, frame1_3, frame2_3)
        x = self.up3(x, frame1_2, frame2_2)
        x = self.up4(x, frame1_1, frame2_1)
        return self.out(x)


class DoubleStreamedCCNN(nn.Module):
    def __init__(self, n_channels=3, bilinear=True):
        super(DoubleStreamedCCNN, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        self.inc_left = DoubleConv(self.n_channels, 64)
        self.down1_left = Down(64, 128)
        self.down2_left = Down(128, 256)
        self.down3_left = Down(256, 512)
        self.down4_left = Down(512, 512)
        self.inc_right = DoubleConv(self.n_channels, 64)
        self.down1_right = Down(64, 128)
        self.down2_right = Down(128, 256)
        self.down3_right = Down(256, 512)
        self.down4_right = Down(512, 512)
        self.up1 = UpDouble(2048, 256, bilinear)
        self.up2 = Up(768, 128, bilinear)
        self.up3 = Up(384, 64, bilinear)
        self.up4 = Up(192, 64, bilinear)
        self.out = OutConv(64, self.n_channels)

    def forward(self, frame1, frame2):
        frame1_1 = self.inc_left(frame1)
        frame1_2 = self.down1_left(frame1_1)
        frame1_3 = self.down2_left(frame1_2)
        frame1_4 = self.down3_left(frame1_3)
        frame1_5 = self.down4_left(frame1_4)
        # Right down
        frame2_1 = self.inc_right(frame2)
        frame2_2 = self.down1_right(frame2_1)
        frame2_3 = self.down2_right(frame2_2)
        frame2_4 = self.down3_right(frame2_3)
        frame2_5 = self.down4_right(frame2_4)
        # Up
        x = self.up1(frame1_5, frame2_5, frame1_4, frame2_4)
        x = self.up2(x, frame1_3, frame2_3)
        x = self.up3(x, frame1_2, frame2_2)
        x = self.up4(x, frame1_1, frame2_1)
        return self.out(x)


class DoubleStreamedFlowCCNN(nn.Module):
    def __init__(self, n_channels=3, bilinear=True):
        super(DoubleStreamedFlowCCNN, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        self.inc_left = DoubleConv(self.n_channels+2, 64)
        self.down1_left = Down(64, 128)
        self.down2_left = Down(128, 256)
        self.down3_left = Down(256, 512)
        self.down4_left = Down(512, 512)
        self.inc_right = DoubleConv(self.n_channels+2, 64)
        self.down1_right = Down(64, 128)
        self.down2_right = Down(128, 256)
        self.down3_right = Down(256, 512)
        self.down4_right = Down(512, 512)
        self.up1 = UpDouble(2048, 256, bilinear)
        self.up2 = Up(768, 128, bilinear)
        self.up3 = Up(384, 64, bilinear)
        self.up4 = Up(192, 64, bilinear)
        self.out = OutConv(64, self.n_channels)

    def forward(self, frame1, frame2):
        with torch.no_grad():
            b, c, w, h = frame1.shape

            frame1_up = F.upsample(frame1, size=(256, 256), mode='bilinear')
            frame2_up = F.upsample(frame2, size=(256, 256), mode='bilinear')

            flow_net = get_flow_net()
            frames_1_2_up = torch.cat([frame1_up[None], frame2_up[None]])
            frames_2_1_up = torch.cat([frame2_up[None], frame1_up[None]])

            frames_1_2_up = frames_1_2_up.permute(1, 2, 0, 3, 4)
            frames_2_1_up = frames_2_1_up.permute(1, 2, 0, 3, 4)

            flow_1_2 = flow_net(frames_1_2_up)
            flow_2_1 = flow_net(frames_2_1_up)

            flow_1_2 = F.upsample(flow_1_2, size=(w, h), mode='bilinear')
            flow_2_1 = F.upsample(flow_2_1, size=(w, h), mode='bilinear')
        input1 = torch.cat([frame1, flow_1_2], 1)
        frame1_1 = self.inc_left(input1)
        frame1_2 = self.down1_left(frame1_1)
        frame1_3 = self.down2_left(frame1_2)
        frame1_4 = self.down3_left(frame1_3)
        frame1_5 = self.down4_left(frame1_4)
        # Right down
        input2 = torch.cat([frame2, flow_2_1], 1)
        frame2_1 = self.inc_right(input2)
        frame2_2 = self.down1_right(frame2_1)
        frame2_3 = self.down2_right(frame2_2)
        frame2_4 = self.down3_right(frame2_3)
        frame2_5 = self.down4_right(frame2_4)
        # Up
        x = self.up1(frame1_5, frame2_5, frame1_4, frame2_4)
        x = self.up2(x, frame1_3, frame2_3)
        x = self.up3(x, frame1_2, frame2_2)
        x = self.up4(x, frame1_1, frame2_1)
        return self.out(x)


class FlowCCNN(nn.Module):
    def __init__(self, n_channels=3, bilinear=True):
        super(FlowCCNN, self).__init__()
        self.n_channels = n_channels + 2
        self.bilinear = bilinear

        self.inc = DoubleConv(self.n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = UpDouble(2048, 256, bilinear)
        self.up2 = Up(768, 128, bilinear)
        self.up3 = Up(384, 64, bilinear)
        self.up4 = Up(192, 64, bilinear)
        self.out = OutConv(64, n_channels)

    def forward(self, frame1, frame2):
        # Compute flows in both directions
        with torch.no_grad():
            b, c, w, h = frame1.shape

            frame1_up = F.upsample(frame1, size=(256, 256), mode='bilinear')
            frame2_up = F.upsample(frame2, size=(256, 256), mode='bilinear')

            flow_net = get_flow_net()
            frames_1_2_up = torch.cat([frame1_up[None], frame2_up[None]])
            frames_2_1_up = torch.cat([frame2_up[None], frame1_up[None]])

            frames_1_2_up = frames_1_2_up.permute(1, 2, 0, 3, 4)
            frames_2_1_up = frames_2_1_up.permute(1, 2, 0, 3, 4)

            flow_1_2 = flow_net(frames_1_2_up)
            flow_2_1 = flow_net(frames_2_1_up)

            flow_1_2 = F.upsample(flow_1_2, size=(w, h), mode='bilinear')
            flow_2_1 = F.upsample(flow_2_1, size=(w, h), mode='bilinear')

        # Left down
        input1 = torch.cat([frame1, flow_1_2], 1)
        frame1_1 = self.inc(input1)
        frame1_2 = self.down1(frame1_1)
        frame1_3 = self.down2(frame1_2)
        frame1_4 = self.down3(frame1_3)
        frame1_5 = self.down4(frame1_4)
        # Right down
        input2 = torch.cat([frame2, flow_2_1], 1)
        frame2_1 = self.inc(input2)
        frame2_2 = self.down1(frame2_1)
        frame2_3 = self.down2(frame2_2)
        frame2_4 = self.down3(frame2_3)
        frame2_5 = self.down4(frame2_4)
        # Up
        x = self.up1(frame1_5, frame2_5, frame1_4, frame2_4)
        x = self.up2(x, frame1_3, frame2_3)
        x = self.up3(x, frame1_2, frame2_2)
        x = self.up4(x, frame1_1, frame2_1)
        return self.out(x)
