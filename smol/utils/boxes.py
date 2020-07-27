from typing import List
import torch
from torch import Tensor


def xywh2xyxy(xywh: Tensor) -> Tensor:
    xy, wh = torch.split(xywh, (2, 2), dim=-1)
    x1y1 = xy - wh / 2
    x2y2 = xy + wh / 2
    xyxy = torch.cat([x1y1, x2y2], dim=-1)
    return xyxy


def box_iou(b1: Tensor, b2: Tensor) -> Tensor:
    b1_x1, b1_y1, b1_x2, b1_y2 = b1[:, 0], b1[:, 1], b1[:, 2], b1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = b2[:, 0], b2[:, 1], b2[:, 2], b2[:, 3]

    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.max(b1_x2, b2_x2)
    inter_y2 = torch.max(b1_y2, b2_y2)

    inter_w = torch.clamp(inter_x2 - inter_x1 + 1, min=0)
    inter_h = torch.clamp(inter_y2 - inter_y1 + 1, min=0)
    intersect = inter_w * inter_h

    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
    union = b1_area + b2_area - intersect

    return intersect / union


def weighted_nms(dets: Tensor, nms_thresh: float) -> Tensor:
    sort_by_score = torch.argsort(dets[:, 4], descending=True)
    dets = dets[sort_by_score]

    keep = []
    while dets.size(0):
        overlap = box_iou(dets[:1, :4], dets[:, :4]) > nms_thresh
        match = dets[:1, -1] == dets[:, -1]
        invalid = overlap & match

        weights = dets[invalid, 4:5]
        weighted_box = torch.sum(weights * dets[invalid, :4], dim=0)
        weighted_box = weighted_box / torch.sum(weights)
        dets[0, :4] = weighted_box

        keep.append(dets[0])
        dets = dets[~invalid]

    keep = torch.stack(keep) if keep else None
    return keep
