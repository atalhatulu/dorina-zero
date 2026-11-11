import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import pytest

from board import create_board, PLAYER
from network import Connect4Net, count_params
from replay_buffer import ReplayBuffer
from train import train_one_epoch, train, save_checkpoint, load_checkpoint


def make_fake_data(n=64):
    """Rastgele eğitim verisi."""
    data = []
    for i in range(n):
        board = create_board()
        player = 1 if i % 2 == 0 else 2
        policy = torch.softmax(torch.randn(7), dim=0).numpy()
        value = (i % 3) - 1  # -1, 0, 1
        data.append((board, player, policy.astype("float32"), float(value)))
    return data


def test_replay_buffer_add_sample():
    rb = ReplayBuffer(capacity=100)
    data = make_fake_data(20)
    rb.add(data)
    assert len(rb) == 20
    sample = rb.sample(8)
    assert len(sample) == 8


def test_replay_buffer_capacity():
    rb = ReplayBuffer(capacity=10)
    rb.add(make_fake_data(50))
    assert len(rb) == 10


def test_train_one_epoch_runs():
    model = Connect4Net()
    data = make_fake_data(32)
    pl, vl = train_one_epoch(model, data, batch_size=8, lr=1e-3)
    assert pl > 0
    assert vl >= 0


def test_train_reduces_loss():
    """Aynı veriyle birkaç epoch → loss düşmeli."""
    model = Connect4Net()
    data = make_fake_data(64)
    pl1, vl1 = train_one_epoch(model, data, batch_size=8, lr=1e-3)
    for _ in range(3):
        pl2, vl2 = train_one_epoch(model, data, batch_size=8, lr=1e-3)
    assert pl2 < pl1, f"Policy loss düşmedi: {pl1} → {pl2}"
    assert vl2 < vl1, f"Value loss düşmedi: {vl1} → {vl2}"


def test_save_load_checkpoint(tmp_path):
    model = Connect4Net()
    path = tmp_path / "ckpt.pt"
    save_checkpoint(model, path, iteration=5)
    assert path.exists()

    model2 = Connect4Net()
    it = load_checkpoint(model2, path)
    assert it == 5
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2)


def test_train_does_not_break_output():
    """Eğitimden sonra model hâlâ geçerli olasılık üretmeli."""
    model = Connect4Net()
    train(model, make_fake_data(32), epochs=1, batch_size=8)
    model.eval()
    x = torch.zeros(2, 3, 6, 7)
    with torch.no_grad():
        policy_logits, value = model(x)
    probs = torch.softmax(policy_logits, dim=1)
    assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-4)