"""core/trainer.py — Oyun-Agnostik AlphaZero Eğitim Motoru."""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class AlphaZeroDataset(Dataset):
    """(encoded_state, policy, value) -> PyTorch tensörleri."""
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        encoded_x, policy, value = self.data[idx]
        return (
            torch.from_numpy(encoded_x).float(),
            torch.from_numpy(policy).float(),
            torch.tensor([value], dtype=torch.float32),
        )


class GenericTrainer:
    def __init__(self, model, lr=2e-4, weight_decay=1e-4, value_weight=2.0, device="cpu"):
        self.model = model.to(device)
        self.device = device
        self.value_weight = value_weight
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def train_epoch(self, data, batch_size=64):
        self.model.train()
        dataset = AlphaZeroDataset(data)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        n_batches = 0

        for x, policy_target, value_target in loader:
            x = x.to(self.device)
            policy_target = policy_target.to(self.device)
            value_target = value_target.to(self.device)

            policy_logits, value = self.model(x)

            log_probs = F.log_softmax(policy_logits, dim=1)
            policy_loss = -(policy_target * log_probs).sum(dim=1).mean()
            value_loss = F.mse_loss(value, value_target)

            loss = policy_loss + self.value_weight * value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            n_batches += 1

        return (
            total_policy_loss / max(n_batches, 1),
            total_value_loss / max(n_batches, 1),
        )

    def train(self, data, epochs=3, batch_size=64):
        pl, vl = 0.0, 0.0
        for _ in range(epochs):
            pl, vl = self.train_epoch(data, batch_size=batch_size)
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
