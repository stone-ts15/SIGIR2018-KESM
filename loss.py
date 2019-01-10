
import torch
import torch.nn.functional as F


def pairwise_loss(*outputs):
    out_pos, out_neg = outputs
    # out_pos: [batch, 1]
    # out_neg: [batch, 1]

    return torch.mean(F.relu(1 - out_pos.squeeze() + out_neg.squeeze()))

