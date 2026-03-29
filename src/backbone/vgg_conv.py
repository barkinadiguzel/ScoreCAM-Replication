import torch
import torchvision.models as models
from torch import nn

class VGGConv(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        vgg = models.vgg16(pretrained=pretrained)
        self.features = vgg.features  
        self.target_layer = 'features.29'  

    def forward(self, x):
        return self.features(x)
