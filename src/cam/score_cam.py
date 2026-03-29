import torch
import torch.nn.functional as F

def score_cam(feature_maps, alpha_k):
    weighted_sum = torch.sum(alpha_k[:, None, None, None] * feature_maps, dim=0)
    return F.relu(weighted_sum)
