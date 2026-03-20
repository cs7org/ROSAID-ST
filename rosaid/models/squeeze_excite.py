import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeExciteLinear(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, max(2, channels // reduction), bias=False)
        self.fc2 = nn.Linear(max(2, channels // reduction), channels, bias=False)

    def forward(self, x):
        # x: [B, C, L]
        b, c, _ = x.size()
        y = x.mean(-1)  # [B, C]
        y = F.relu(self.fc1(y))  # [B, C//r]
        y = torch.sigmoid(self.fc2(y))  # [B, C]
        return x * y.view(b, c, 1)


class SqueezeExciteConv1D(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Conv1d(
            channels, max(2, channels // reduction), kernel_size=1, bias=False
        )
        self.fc2 = nn.Conv1d(
            max(2, channels // reduction), channels, kernel_size=1, bias=False
        )

    def forward(self, x):
        # x: [B, C, L]
        b, c, _ = x.size()
        y = x.mean(-1, keepdim=True)  # [B, C, 1]
        y = F.relu(self.fc1(y))  # [B, C//r, 1]
        y = torch.sigmoid(self.fc2(y))  # [B, C, 1]
        return x * y


class SqueezeExciteConv2D(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Conv2d(
            channels, max(2, channels // reduction), kernel_size=1, bias=False
        )
        self.fc2 = nn.Conv2d(
            max(2, channels // reduction), channels, kernel_size=1, bias=False
        )

    def forward(self, x):
        # x: [B, C, H, W]
        b, c, _, _ = x.size()
        y = x.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
        y = F.relu(self.fc1(y))  # [B, C//r, 1, 1]
        y = torch.sigmoid(self.fc2(y))  # [B, C, 1, 1]
        return x * y
