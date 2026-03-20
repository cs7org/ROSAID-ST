import torch
from torch import nn


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, targets):
        ce = torch.nn.functional.cross_entropy(
            logits, targets, weight=self.weight, reduction="none"
        )
        p_t = torch.exp(-ce)
        loss = ((1 - p_t) ** self.gamma * ce).mean()
        return loss
