import torch
import torchvision.models as models
from torch import nn

class ResNetConv(nn.Module):
    def __init__(self, pretrained=True, target_layer='layer4'):
        super().__init__()
        resnet = models.resnet18(pretrained=pretrained)  
        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4  
        )
        self.target_layer = target_layer

    def forward(self, x):
        return self.features(x)  
