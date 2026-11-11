"""NN eğitimi: policy + value loss. CPU için optimize."""
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from network import Connect4Net, board_to_tensor


class SelfPlayDataset(Dataset):
    """(board, player, policy, value) → tensörler."""
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        board, player, policy, value = self.data[idx]
        x = board_to_tensor(board, player)  # (3, 6, 7)
        return (
            torch.from_numpy(x),
            torch.from_numpy(policy).float(),
            torch.tensor([value], dtype=torch.float32),
        )


def train_one_epoch(model, data, optimizer, batch_size=64, device="cpu"):
    """Bir epoch eğit. (avg_policy_loss, avg_value_loss) döndür."""
    model.train()
    dataset = SelfPlayDataset(data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    total_policy_loss = 0.0
    total_value_loss = 0.0
    n_batches = 0

    for x, policy_target, value_target in loader:
        x = x.to(device)
        policy_target = policy_target.to(device)
        value_target = value_target.to(device)

        policy_logits, value = model(x)

        # Policy loss: cross-entropy (soft target ile)
        # policy_target zaten olasılık dağılımı → log_softmax + NLL
        log_probs = F.log_softmax(policy_logits, dim=1)
        policy_loss = -(policy_target * log_probs).sum(dim=1).mean()

        # Value loss: MSE
        value_loss = F.mse_loss(value, value_target)

        loss = policy_loss + 2.0 * value_loss  # 1.5 → 2.0

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
        n_batches += 1

    return (
        total_policy_loss / max(n_batches, 1),
        total_value_loss / max(n_batches, 1),
    )


def train(model, data, epochs=1, batch_size=64, lr=1e-3, device="cpu", optimizer=None):
    """N epoch eğit. (policy_loss, value_loss) döndür (son epoch)."""
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    pl, vl = 0.0, 0.0
    for epoch in range(epochs):
        pl, vl = train_one_epoch(model, data, optimizer=optimizer, batch_size=batch_size, device=device)
    return pl, vl


def save_checkpoint(model, path, iteration):
    torch.save({
        "iteration": iteration,
        "model_state": model.state_dict(),
    }, path)


def load_checkpoint(model, path):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    return ckpt.get("iteration", 0)
