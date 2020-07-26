from typing import List
import torch
from torch import Tensor


def xywh2xyxy(xywh: Tensor) -> Tensor:
    xy, wh = torch.split(xywh, (2, 2), dim=-1)
    x1y1 = xy - wh / 2
    x2y2 = xy + wh / 2
    xyxy = torch.cat([x1y1, x2y2], dim=-1)
    return xyxy


def bbox_iou(b1: Tensor, b2: Tensor) -> Tensor:
    b1_x1, b1_y1, b1_x2, b1_y2 = b1[:, 0], b1[:, 1], b1[:, 2], b1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = b2[:, 0], b2[:, 1], b2[:, 2], b2[:, 3]

    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.max(b1_x2, b2_x2)
    inter_y2 = torch.max(b1_y2, b2_y2)

    inter_w = torch.clamp(inter_x2 - inter_x1 + 1, min=0)
    inter_h = torch.clamp(inter_y2 - inter_y1 + 1, min=0)
    inter_area = inter_w * inter_h

    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)
    return iou


def greedy_nms(preds: Tensor, beta_nms: float) -> List[Tensor]:
    outputs = []
    preds[..., :4] = xywh2xyxy(preds[..., :4])
    for i, pred in enumerate(preds):
        boxes, box_score, cls_score = pred[:, :4], pred[:, 4:5], pred[:, 5:]
        cls_score, cls_id = torch.max(cls_score, dim=1, keepdim=True)
        score = box_score * max_cls_score
        pred = pred[torch.argsort(score, dim=1, descending=True)]
        dets = torch.cat([boxes, box_score, cls_score, cls_id.float()], dim=1)
        keep = []
        while dets.size(0):
            large_overlap = bbox_iou(dets[:1, :4], dets[:, :4]) > beta_nms
            label_match = dets[:1, -1] == dets[:, -1]
            invalid = large_overlap & label_match
            weights = dets[invalid, 4:5]
            dets[0, :4] = (weights * dets[invalid, :4]).sum(0) / weights.sum()
            keep.append(dets[0])
            dets = dets[~invalid]
        keep = torch.stack(keep) if keep else None
        outputs.append(keep)
    return outputs
