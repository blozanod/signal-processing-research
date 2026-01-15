import torch
import torch.nn as nn
import torch.nn.functional as F

class L1_Charbonnier_loss(nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self, eps=1e-3, reduction="mean"):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, X, Y):
        diff = X - Y
        loss = torch.sqrt(diff * diff + self.eps * self.eps)
        if self.reduction == "sum":
            return loss.sum()
        return loss.mean()