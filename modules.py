from typing import Tuple, Optional

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
    def __init__(self):
        super().__init__()
