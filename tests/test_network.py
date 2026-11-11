import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import pytest

from board import create_board, drop_piece, ROWS, COLS, PLAYER, AI
from network import (
    Connect4Net, board_to_tensor, batch_to_tensor, predict, count_params,
)


def test_model_forward_shape():
    model = Connect4Net()
    x = torch.zeros(2, 3, ROWS, COLS)
    policy, value = model(x)
    assert policy.shape == (2, COLS)
    assert value.shape == (2, 1)


def test_value_range():
    model = Connect4Net()
    x = torch.zeros(4, 3, ROWS, COLS)
    _, value = model(x)
    assert torch.all(value >= -1.0)
    assert torch.all(value <= 1.0)


def test_board_to_tensor_shape():
    b = create_board()
    t = board_to_tensor(b, PLAYER)
    assert t.shape == (3, ROWS, COLS)
    assert t.dtype == np.float32


def test_board_to_tensor_pieces():
    b = create_board()
    drop_piece(b, ROWS - 1, 0, PLAYER)
    drop_piece(b, ROWS - 1, 1, AI)
    t = board_to_tensor(b, PLAYER)
    # PLAYER perspektifinden: me=PLAYER, opp=AI
    assert t[0, ROWS - 1, 0] == 1.0   # me
    assert t[1, ROWS - 1, 1] == 1.0   # opp
    assert t[0, ROWS - 1, 1] == 0.0
    assert t[1, ROWS - 1, 0] == 0.0
    # const kanal hep 1
    assert np.all(t[2] == 1.0)


def test_board_to_tensor_switches_perspective():
    b = create_board()
    drop_piece(b, ROWS - 1, 0, PLAYER)
    t_player = board_to_tensor(b, PLAYER)
    t_ai = board_to_tensor(b, AI)
    # PLAYER perspektifinde me kanalı, AI perspektifinde opp kanalı olmalı
    assert t_player[0, ROWS - 1, 0] == 1.0
    assert t_ai[1, ROWS - 1, 0] == 1.0


def test_batch_to_tensor():
    b1 = create_board()
    b2 = create_board()
    drop_piece(b2, ROWS - 1, 3, AI)
    x = batch_to_tensor([(b1, PLAYER), (b2, AI)])
    assert x.shape == (2, 3, ROWS, COLS)
    assert x.dtype == torch.float32


def test_predict_returns_valid():
    model = Connect4Net()
    b = create_board()
    policy, value = predict(model, b, PLAYER)
    assert policy.shape == (COLS,)
    assert abs(policy.sum() - 1.0) < 1e-5   # olasılık dağılımı
    assert -1.0 <= value <= 1.0


def test_count_params_reasonable():
    model = Connect4Net()
    n = count_params(model)
    # CPU için 100K-2M arası olmalı
    assert 100_000 < n < 2_000_000, f"Parametre sayısı {n} — CPU için uygun değil"


def test_model_save_load(tmp_path):
    model = Connect4Net()
    path = tmp_path / "model.pt"
    torch.save(model.state_dict(), path)

    model2 = Connect4Net()
    model2.load_state_dict(torch.load(path))
    model2.eval()

    b = create_board()
    p1, v1 = predict(model, b, PLAYER)
    p2, v2 = predict(model2, b, PLAYER)
    assert np.allclose(p1, p2, atol=1e-6)
    assert abs(v1 - v2) < 1e-6