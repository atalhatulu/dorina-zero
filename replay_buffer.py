"""Eğitim verisi deposu. FIFO (eski veri atılır)."""
from collections import deque
import random


class ReplayBuffer:
    def __init__(self, capacity=50_000):
        self.buffer = deque(maxlen=capacity)

    def add(self, data):
        """data: (board, player, policy, value) listesi."""
        self.buffer.extend(data)

    def sample(self, batch_size, max_draw_ratio=None):
        """
        Rastgele batch döndür.
        max_draw_ratio: Beraberlik (value == 0.0) örneklerinin maksimum oranı (örn: 0.20 -> en fazla %20 draw).
        """
        if len(self.buffer) <= batch_size:
            items = list(self.buffer)
        else:
            items = random.sample(self.buffer, min(len(self.buffer), batch_size * 2))

        if max_draw_ratio is not None and len(self.buffer) > 0:
            indices = list(range(len(self.buffer)))
            random.shuffle(indices)

            decisive_indices = [i for i in indices if abs(self.buffer[i][3]) > 1e-4]
            draw_indices = [i for i in indices if abs(self.buffer[i][3]) <= 1e-4]

            max_draws = int(batch_size * max_draw_ratio)
            allowed_draws = draw_indices[:max_draws]
            needed_decisive = batch_size - len(allowed_draws)

            selected_indices = decisive_indices[:needed_decisive] + allowed_draws
            if len(selected_indices) < batch_size:
                # Kalanı diğer indekslerden tamamla
                selected_set = set(selected_indices)
                remaining = [i for i in indices if i not in selected_set]
                selected_indices.extend(remaining[:batch_size - len(selected_indices)])

            return [self.buffer[i] for i in selected_indices[:batch_size]]

        if len(self.buffer) < batch_size:
            batch = list(self.buffer)
        else:
            batch = random.sample(self.buffer, batch_size)
        return batch

    def __len__(self):
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()