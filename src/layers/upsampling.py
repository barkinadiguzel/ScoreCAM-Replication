import torch.nn.functional as F

def upsample_activation(activation, target_size):
    return F.interpolate(activation, size=target_size, mode='bilinear', align_corners=False)
