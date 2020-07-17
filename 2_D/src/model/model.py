import torch

from src.model.sepconv_network import SepConvNet
from src.model.ccnn.model import CCNN, DoubleStreamedCCNN, DoubleStreamedFlowCCNN, FlowCCNN

__all__ = ['CycleCCNN', 'CycleSepConv', 'DoubleStreamedFlowCycleCCNN', 'DoubleStreamedCycleCCNN', 'FlowCycleCCNN']


class CycleCCNN(torch.nn.Module):

    def __init__(self):
        super(CycleCCNN, self).__init__()
        self.backbone = CCNN()

    def forward_single(self, frame0, frame2):
        frame1_pred = self.backbone(frame0, frame2)
        return frame1_pred

    def forward(self, frame0, frame2, frame4):
        frame1_pred = self.backbone(frame0, frame2)
        frame3_pred = self.backbone(frame2, frame4)
        frame2_pred = self.backbone(frame1_pred, frame3_pred)
        return frame1_pred, frame2_pred, frame3_pred

    def predict(self, frame0, frame2):
        return self.backbone(frame0, frame2)


class DoubleStreamedFlowCycleCCNN(torch.nn.Module):

    def __init__(self):
        super(DoubleStreamedFlowCycleCCNN, self).__init__()
        self.backbone = DoubleStreamedFlowCCNN()

    def forward_single(self, frame0, frame2):
        frame1_pred = self.backbone(frame0, frame2)
        return frame1_pred

    def forward(self, frame0, frame2, frame4):
        frame1_pred = self.backbone(frame0, frame2)
        frame3_pred = self.backbone(frame2, frame4)
        frame2_pred = self.backbone(frame1_pred, frame3_pred)
        return frame1_pred, frame2_pred, frame3_pred

    def predict(self, frame0, frame2):
        return self.backbone(frame0, frame2)


class CycleSepConv(torch.nn.Module):

    def __init__(self, kernel_size):
        super(CycleSepConv, self).__init__()
        self.backbone = SepConvNet(kernel_size)

    def forward_single(self, frame0, frame2):
        frame1_pred = self.backbone(frame0, frame2)
        return frame1_pred

    def forward(self, frame0, frame2, frame4):
        frame1_pred = self.backbone(frame0, frame2)
        frame3_pred = self.backbone(frame2, frame4)
        frame2_pred = self.backbone(frame1_pred, frame3_pred)
        return frame1_pred, frame2_pred, frame3_pred

    def predict(self, frame0, frame2):
        return self.backbone(frame0, frame2)


class DoubleStreamedCycleCCNN(torch.nn.Module):

    def __init__(self):
        super(DoubleStreamedCycleCCNN, self).__init__()
        self.backbone = DoubleStreamedCCNN()

    def forward_single(self, frame0, frame2):
        frame1_pred = self.backbone(frame0, frame2)
        return frame1_pred

    def forward(self, frame0, frame2, frame4):
        frame1_pred = self.backbone(frame0, frame2)
        frame3_pred = self.backbone(frame2, frame4)
        frame2_pred = self.backbone(frame1_pred, frame3_pred)
        return frame1_pred, frame2_pred, frame3_pred

    def predict(self, frame0, frame2):
        return self.backbone(frame0, frame2)


class FlowCycleCCNN(torch.nn.Module):

    def __init__(self):
        super(FlowCycleCCNN, self).__init__()
        self.backbone = FlowCCNN()

    def forward_single(self, frame0, frame2):
        frame1_pred = self.backbone(frame0, frame2)
        return frame1_pred

    def forward(self, frame0, frame2, frame4):
        frame1_pred = self.backbone(frame0, frame2)
        frame3_pred = self.backbone(frame2, frame4)
        frame2_pred = self.backbone(frame1_pred, frame3_pred)
        return frame1_pred, frame2_pred, frame3_pred

    def predict(self, frame0, frame2):
        return self.backbone(frame0, frame2)
