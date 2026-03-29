import torch


class Config:
    INPUT_SIZE = (224, 224)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    MODEL_NAME = "resnet18"   # resnet18 / resnet50 / vgg16

    TARGET_LAYER = "layer4"
    TARGET_CLASS = None 

    EPSILON = 1e-8
