"""Connect4 için PyTorch CNN. Policy + Value head. CPU için optimize."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from board import ROWS, COLS, EMPTY, PLAYER, AI


class Connect4Net(nn.Module):
    """
    Girdi: (batch, 3, 6, 7)
      kanal 0: mevcut oyuncunun taşları
      kanal 1: rakibin taşları
      kanal 2: sabit 1 kanalı (sıra kimde bilgisi yerine basit sabit)
    Çıktı:
      policy: (batch, 7) logits (her sütun için)
      value:  (batch, 1) tanh → -1..1 (mevcut oyuncunun kazanma tahmini)
    """

    def __init__(self, channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(3, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv3 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv4 = nn.Conv2d(channels, channels, 3, padding=1)
        self.fc = nn.Linear(channels * ROWS * COLS, 128)

        # Policy head
        self.policy_fc = nn.Linear(128, COLS)

        # Value head
        self.value_fc1 = nn.Linear(128, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.flatten(1)
        x = F.relu(self.fc(x))

        policy_logits = self.policy_fc(x)              # (B, 7)
        v = F.relu(self.value_fc1(x))
        value = torch.tanh(self.value_fc2(v))          # (B, 1)

        return policy_logits, value


def board_to_tensor(board, current_player):
    """
    Tahtayı NN girdisine çevirir.
    current_player: PLAYER (1) veya AI (2)
    Dönüş: (3, 6, 7) numpy array
    """
    me = current_player
    opp = AI if current_player == PLAYER else PLAYER
    me_ch = np.array([[1.0 if board[r][c] == me else 0.0
                       for c in range(COLS)] for r in range(ROWS)], dtype=np.float32)
    opp_ch = np.array([[1.0 if board[r][c] == opp else 0.0
                        for c in range(COLS)] for r in range(ROWS)], dtype=np.float32)
    const_ch = np.ones((ROWS, COLS), dtype=np.float32)
    return np.stack([me_ch, opp_ch, const_ch], axis=0)


def batch_to_tensor(boards_and_players):
    """
    [(board, current_player), ...] → (B, 3, 6, 7) tensor
    """
    arrs = [board_to_tensor(b, p) for (b, p) in boards_and_players]
    return torch.from_numpy(np.stack(arrs, axis=0))


def predict(model, board, current_player):
    """
    Tek pozisyon için policy (7 olasılık) ve value (-1..1) döndürür.
    CPU'da, no_grad ile.
    """
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(board_to_tensor(board, current_player)).unsqueeze(0)
        policy_logits, value = model(x)
        policy = F.softmax(policy_logits, dim=1).squeeze(0).numpy()
        v = value.item()
    return policy, v


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
