"""Eğitim verisi deposu. FIFO (eski veri atılır)."""
from collections import deque
import random


class ReplayBuffer:
    def __init__(self, capacity=50_000):
        self.buffer = deque(maxlen=capacity)

    def add(self, data):
        """data: (board, player, policy, value) listesi."""
        self.buffer.extend(data)

    def sample(self, batch_size):
        """Rastgele batch döndür."""
        if len(self.buffer) < batch_size:
            batch = list(self.buffer)
        else:
            batch = random.sample(self.buffer, batch_size)
        return batch

    def __len__(self):
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()