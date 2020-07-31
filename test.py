import torch
from torch.nn import functional as F
from time import time
import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches

from smol import YoloV4Tiny
from smol.utils.darknet import load_darknet_weights
from smol.utils.export import export_onnx


coco_labels = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

model = YoloV4Tiny().eval()
success = load_darknet_weights(model, "darknet/weights/yolov4-tiny.weights")
assert success

export_onnx(
    model, "yolov4-tiny.onnx", input_shape=(1, 3, 416, 416), simplify=True,
)

img = cv2.imread("images/dog.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
height, width, _ = img.shape

fig, ax = plt.subplots(1)
ax.imshow(img)

img = torch.from_numpy(img.transpose(2, 0, 1))
with torch.no_grad():
    boxes = model.detect([img])
boxes = boxes[0].cpu().numpy()

for box in boxes:
    x1 = box[0]
    y1 = box[1]
    x2 = box[2]
    y2 = box[3]
    score = box[4]
    cls_id = box[5]

    x = int(x1)
    y = int(y1)
    w = int(x2 - x1)
    h = int(y2 - y1)
    cls_id = int(cls_id)

    rect = patches.Rectangle(
        (x, y), w, h, facecolor="none", linewidth=1, edgecolor="red"
    )
    ax.add_patch(rect)
    print(coco_labels[cls_id])

plt.show()
