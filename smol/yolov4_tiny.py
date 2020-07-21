from typing import Optional, Union, Tuple
import torch
from torch import Tensor, nn

from smol.modules import ConvLayer, Route, YoloLayer


class YoloV4Tiny(nn.Module):
    def __init__(self, num_classes: int = 80):
        super().__init__()
        self.num_classes = num_classes
        self.out_channels = 3 * (num_classes + 5)

        self.conv0 = ConvLayer(3, 32, 3, 2, activ="leaky")
        self.conv1 = ConvLayer(32, 64, 3, 2, activ="leaky")
        self.conv2 = ConvLayer(64, 64, 3, 1, activ="leaky")
        self.route3 = Route(groups=2, group_id=1)
        self.conv4 = ConvLayer(32, 32, 3, 1, activ="leaky")
        self.conv5 = ConvLayer(32, 32, 3, 1, activ="leaky")
        self.route6 = Route()
        self.conv7 = ConvLayer(64, 64, 1, 1, activ="leaky")
        self.route8 = Route()
        self.pool9 = nn.MaxPool2d(2, 2)
        self.conv10 = ConvLayer(128, 128, 3, 1, activ="leaky")
        self.route11 = Route(groups=2, group_id=1)
        self.conv12 = ConvLayer(64, 64, 3, 1, activ="leaky")
        self.conv13 = ConvLayer(64, 64, 3, 1, activ="leaky")
        self.route14 = Route()
        self.conv15 = ConvLayer(128, 128, 1, 1, activ="leaky")
        self.route16 = Route()
        self.pool17 = nn.MaxPool2d(2, 2)
        self.conv18 = ConvLayer(256, 256, 3, 1, activ="leaky")
        self.route19 = Route(groups=2, group_id=1)
        self.conv20 = ConvLayer(128, 128, 3, 1, activ="leaky")
        self.conv21 = ConvLayer(128, 128, 3, 1, activ="leaky")
        self.route22 = Route()
        self.conv23 = ConvLayer(256, 256, 1, 1, activ="leaky")
        self.route24 = Route()
        self.pool25 = nn.MaxPool2d(2, 2)
        self.conv26 = ConvLayer(512, 512, 3, 1, activ="leaky")

        ##################################

        self.conv27 = ConvLayer(512, 256, 1, 1, activ="leaky")
        self.conv28 = ConvLayer(256, 512, 3, 1, activ="leaky")
        self.conv29 = ConvLayer(512, self.out_channels, 1, 1, batch_norm=False)
        self.yolo1 = YoloLayer(32, [(81, 82), (135, 169), (344, 319)], num_classes)

        self.route31 = Route()
        self.conv32 = ConvLayer(256, 128, 1, 1, activ="leaky")
        self.upsample33 = nn.Upsample(scale_factor=2)
        self.route34 = Route()
        self.conv35 = ConvLayer(384, 256, 3, 1, activ="leaky")
        self.conv36 = ConvLayer(256, self.out_channels, 1, 1, batch_norm=False)
        self.yolo2 = YoloLayer(16, [(23, 27), (37, 58), (81, 82)], num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x0 = self.conv0(x)
        x1 = self.conv1(x0)
        x2 = self.conv2(x1)
        x3 = self.route3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.route6(x5, x4)
        x7 = self.conv7(x6)
        x8 = self.route8(x2, x7)
        x9 = self.pool9(x8)
        x10 = self.conv10(x9)
        x11 = self.route11(x10)
        x12 = self.conv12(x11)
        x13 = self.conv13(x12)
        x14 = self.route14(x13, x12)
        x15 = self.conv15(x14)
        x16 = self.route16(x10, x15)
        x17 = self.pool17(x16)
        x18 = self.conv18(x17)
        x19 = self.route19(x18)
        x20 = self.conv20(x19)
        x21 = self.conv21(x20)
        x22 = self.route22(x21, x20)
        x23 = self.conv23(x22)
        x24 = self.route24(x18, x23)
        x25 = self.pool25(x24)
        x26 = self.conv26(x25)

        x27 = self.conv27(x26)
        x28 = self.conv28(x27)
        x29 = self.conv29(x28)

        x31 = self.route31(x27)
        x32 = self.conv32(x31)
        x33 = self.upsample33(x32)
        x34 = self.route34(x33, x23)
        x35 = self.conv35(x34)
        x36 = self.conv36(x35)

        yolo1 = self.yolo1(x29)
        yolo2 = self.yolo2(x36)
        return torch.cat([yolo1, yolo2], dim=1)
