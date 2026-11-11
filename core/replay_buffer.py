"""core/replay_buffer.py — Oyun-Agnostik Replay Buffer (Draw Undersampling Destekli)."""
from collections import deque
import random


class GenericReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity)

    def add(self, data):
        """data: [(encoded_state, policy, value), ...] listesi."""
        self.buffer.extend(data)

    def sample(self, batch_size, max_draw_ratio=None):
        """
        Rastgele batch döndürür.
        max_draw_ratio: Beraberlik (abs(value) <= 1e-4) verilerinin maksimum yüzdesi (örn: 0.15).
        """
        if len(self.buffer) == 0:
            return []

        if max_draw_ratio is not None:
            indices = list(range(len(self.buffer)))
            random.shuffle(indices)

            decisive_indices = [i for i in indices if abs(self.buffer[i][2]) > 1e-4]
            draw_indices = [i for i in indices if abs(self.buffer[i][2]) <= 1e-4]

            max_draws = int(batch_size * max_draw_ratio)
            allowed_draws = draw_indices[:max_draws]
            needed_decisive = batch_size - len(allowed_draws)

            selected_indices = decisive_indices[:needed_decisive] + allowed_draws
            if len(selected_indices) < batch_size:
                selected_set = set(selected_indices)
                remaining = [i for i in indices if i not in selected_set]
                selected_indices.extend(remaining[:batch_size - len(selected_indices)])

            return [self.buffer[i] for i in selected_indices[:batch_size]]

        if len(self.buffer) < batch_size:
            return list(self.buffer)
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()
