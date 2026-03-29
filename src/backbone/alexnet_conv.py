import torch
import torchvision.models as models
from torch import nn

class AlexNetConv(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        alexnet = models.alexnet(pretrained=pretrained)
        self.features = alexnet.features  
        self.target_layer = 'features.12'  

    def forward(self, x):
        return self.features(x)
