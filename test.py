import torch

from smol import YoloV4Tiny
from smol.utils.darknet import load_darknet_weights 

model = YoloV4Tiny().cuda()
success = load_darknet_weights(model, "darknet/yolov4-tiny.weights")
assert success

x = torch.FloatTensor(1, 3, 416, 416).uniform_().cuda()
with torch.no_grad():
    y = model(x)
print(y.shape)