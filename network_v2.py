"""Connect4 için PyTorch CNN (Aşama 2: Hafif 130K mimari). Policy + Value head."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from board import ROWS, COLS, PLAYER, AI


class Connect4NetV2(nn.Module):
    """
    Hafif 4-Conv + 1x1 Conv Head mimarisi (~115K-130K parametre).
    Girdi: (batch, 3, 6, 7)
    Çıktı:
      policy: (batch, 7) logits
      value:  (batch, 1) tanh -> -1..1
    """

    def __init__(self, channels=64):
        super().__init__()
        # Backbone: 4 Conv (3x3, padding=1) + BatchNorm
        self.conv1 = nn.Conv2d(3, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

        self.conv3 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(channels)

        self.conv4 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(channels)

        # Policy Head: 1x1 Conv (64 -> 2) + BatchNorm -> Flatten (2 * 6 * 7 = 84) -> Linear(84, 7)
        self.policy_conv = nn.Conv2d(channels, 2, 1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * ROWS * COLS, COLS)

        # Value Head: 1x1 Conv (64 -> 1) + BatchNorm -> Flatten (1 * 6 * 7 = 42) -> Linear(42, 64) -> Linear(64, 1)
        self.value_conv = nn.Conv2d(channels, 1, 1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * ROWS * COLS, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))

        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.flatten(1)
        policy_logits = self.policy_fc(p)

        # Value head
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.flatten(1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        return policy_logits, value


def count_params(model):
    """Toplam eğitilebilir parametre sayısı."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    net = Connect4NetV2()
    n = count_params(net)
    print(f"Connect4NetV2 Parametre Sayısı: {n:,}")
    dummy = torch.randn(2, 3, 6, 7)
    p, v = net(dummy)
    print(f"Çıktı şekilleri: policy={p.shape}, value={v.shape}")
