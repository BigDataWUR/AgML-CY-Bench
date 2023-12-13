
import torch


def batch_tensors(*ts):
    return torch.cat([t.unsqueeze(0) for t in ts], dim=0)

