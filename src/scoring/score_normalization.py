import torch.nn.functional as F

def normalize_scores(scores):
    return F.softmax(scores, dim=0)
