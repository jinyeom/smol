from typing import Tuple, Optional, List

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class Mish(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x * torch.tanh(F.softplus(x))


class ConvLayer(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        batch_norm: bool = True,
        activ: Optional[str] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (kernel_size - 1) // 2
        self.batch_norm = batch_norm
        self.activ = activ

        self.add_module(
            "conv",
            nn.Conv2d(
                self.in_channels,
                self.out_channels,
                self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                bias=not self.batch_norm,
            ),
        )

        if self.batch_norm:
            self.add_module("norm", nn.BatchNorm2d(self.out_channels))

        if self.activ is None:  # linear
            pass
        elif self.activ == "leaky":
            self.add_module(self.activ, nn.LeakyReLU(0.1, inplace=True))
        elif self.activ == "mish":
            self.add_module(self.activ, Mish())
        else:
            raise ValueError(f"Invalid activation function: {self.activ}")


class Route(nn.Module):
    def __init__(self, groups: Optional[int] = None, group_id: Optional[int] = None):
        super().__init__()
        self.groups = groups
        self.group_id = group_id

    def forward(self, *tensors: Tensor) -> Tensor:
        if self.groups is None and self.group_id is None:
            return torch.cat(tensors, dim=1)
        else:
            if len(tensors) > 1:
                raise ValueError(
                    "Route expects only one Tensor when "
                    "`groups` and `group_id` are specified"
                )
            tensor = tensors[0]
            chunks = torch.chunk(tensor, self.groups, dim=1)
            return chunks[self.group_id]


class YoloLayer(nn.Module):
    def __init__(self, stride: int, anchors: List[Tuple[int, int]], num_classes: int):
        super().__init__()
        self.stride = stride
        self.anchors = anchors
        self.num_classes = num_classes

    def forward(self, pred: Tensor) -> Tuple[Tensor, Tensor]:
        device = pred.device
        B, C, H, W = pred.shape

        pred = pred.view(B, -1, self.num_classes + 5, H, W)
        pred = pred.permute(0, 1, 3, 4, 2).contiguous()

        x = torch.sigmoid(pred[..., 0])
        y = torch.sigmoid(pred[..., 1])
        w = pred[..., 2]
        h = pred[..., 3]

        grid_x = torch.arange(W).repeat(W, 1).view([1, 1, W, W]).to(torch.float32)
        grid_y = torch.arange(H).repeat(H, 1).view([1, 1, H, H]).to(torch.float32)

        anchors = [(aw / self.stride, ah / self.stride) for aw, ah in self.anchors]
        anchor_w, anchor_h = list(zip(*anchors))
        anchor_w = torch.FloatTensor(anchor_w).view(1, -1, 1, 1)
        anchor_h = torch.FloatTensor(anchor_h).view(1, -1, 1, 1)

        x = x + grid_x.to(device)
        y = y + grid_y.to(device)
        w = torch.exp(w) * anchor_w.to(device)
        h = torch.exp(h) * anchor_h.to(device)

        pred_boxes = torch.cat([x, y, w, h], dim=-1)
        pred_boxes = self.stride * pred_boxes.view(B, -1, 4)
        box_conf = torch.sigmoid(pred[..., 4]).view(B, -1, 1)
        cls_conf = torch.sigmoid(pred[..., 5:]).view(B, -1, self.num_classes)

        return torch.cat([pred_boxes, box_conf, cls_conf], dim=-1)
