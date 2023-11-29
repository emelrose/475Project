
import torchvision.models as models
import torch
import torch.nn as nn
class Vet_Net(nn.Module):
    def __init__(self):
        super(Vet_Net, self).__init__()
        self.yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=False)

    def forward(self, x):
        x = self.yolo(x)
        return x