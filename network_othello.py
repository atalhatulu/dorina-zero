"""network_othello.py — 8x8 Othello (Reversi) için AlphaZero Residual CNN Mimarisi (OthelloNet).

Girdi: (batch, 3, 8, 8)
  - Kanal 0: Sıradaki oyuncunun taşları (1.0 / 0.0)
  - Kanal 1: Rakip oyuncunun taşları (1.0 / 0.0)
  - Kanal 2: 1'lerden oluşan sabit kanal (renk/sıra göstergesi)

Çıktı:
  - Policy: (batch, 65) logits (64 kare hamlesi + 1 pass hamlesi)
  - Value: (batch, 1) tanh [-1, +1] (bu tahtayı sıradaki oyuncu mu kazanır)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

BOARD_SIZE = 8
ACTION_SIZE = BOARD_SIZE * BOARD_SIZE + 1  # 65


class ResidualBlock(nn.Module):
    """Standart 2x Conv 3x3 Residual Blok."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class OthelloNet(nn.Module):
    """
    Othello için 4 Residual Bloklu Residual CNN (~250K parametre).
    Hızlı CPU self-play ve derin taktiksel genelleme için dengeli mimari.
    """
    def __init__(self, in_channels=3, num_res_blocks=4, channels=64):
        super().__init__()
        # Giriş bloğu
        self.conv_in = nn.Conv2d(in_channels, channels, 3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(channels)

        # Residual omurga
        self.res_blocks = nn.ModuleList([
            ResidualBlock(channels) for _ in range(num_res_blocks)
        ])

        # Policy Head: 1x1 Conv (channels -> 4) + BN -> Linear(4*8*8, 65)
        self.policy_conv = nn.Conv2d(channels, 4, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(4)
        self.policy_fc = nn.Linear(4 * BOARD_SIZE * BOARD_SIZE, ACTION_SIZE)

        # Value Head: 1x1 Conv (channels -> 2) + BN -> Linear(2*8*8, 64) -> Linear(64, 1)
        self.value_conv = nn.Conv2d(channels, 2, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(2)
        self.value_fc1 = nn.Linear(2 * BOARD_SIZE * BOARD_SIZE, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # Backbone
        out = F.relu(self.bn_in(self.conv_in(x)))
        for block in self.res_blocks:
            out = block(out)

        # Policy Head
        p = F.relu(self.policy_bn(self.policy_conv(out)))
        p = p.flatten(1)
        policy_logits = self.policy_fc(p)

        # Value Head
        v = F.relu(self.value_bn(self.value_conv(out)))
        v = v.flatten(1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        return policy_logits, value


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    net = OthelloNet()
    n = count_params(net)
    print(f"OthelloNet Parametre Sayısı: {n:,}")
    dummy = torch.randn(2, 3, 8, 8)
    pol, val = net(dummy)
    print(f"Çıktı şekilleri: policy={pol.shape} (beklenen: (2, 65)), value={val.shape} (beklenen: (2, 1))")
    assert pol.shape == (2, 65)
    assert val.shape == (2, 1)
    print("Test başarılı!")
