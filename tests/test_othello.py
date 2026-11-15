"""tests/test_othello.py — Othello motoru, OthelloNet ve D4 simetri doğrulama testleri."""
import numpy as np
import torch
import pytest

from games.othello import OthelloGame, BOARD_SIZE
from network_othello import OthelloNet, count_params
from core.self_play import augment_data
from core.mcts import GenericMCTS


def test_othello_rules_and_moves():
    game = OthelloGame()
    s = game.get_initial_state()
    assert s.shape == (8, 8)
    # Başlangıçta 4 taş olmalı
    assert np.count_nonzero(s == 1) == 2
    assert np.count_nonzero(s == 2) == 2

    # P1 için ilk geçerli hamleler (4 adet: 19, 26, 37, 44)
    v1 = game.get_valid_moves(s, 1)
    assert v1.shape == (65,)
    assert v1[64] == False  # Hamle varken pass olamaz
    valid_indices = np.where(v1)[0]
    assert len(valid_indices) == 4
    assert set(valid_indices) == {19, 26, 37, 44}

    # Bir hamle yapalım
    s_next = game.get_next_state(s, 19, 1)
    # 19 nolu kare dolmalı ve aradaki taş dönmeli
    assert s_next[19 // 8, 19 % 8] == 1
    assert np.count_nonzero(s_next == 1) == 4
    assert np.count_nonzero(s_next == 2) == 1


def test_othello_encode_state():
    game = OthelloGame()
    s = game.get_initial_state()
    enc = game.encode_state(s, 1)
    assert enc.shape == (3, 8, 8)
    assert enc.dtype == np.float32
    # 3. kanal tamamı 1 olmalı
    np.testing.assert_array_equal(enc[2], np.ones((8, 8), dtype=np.float32))
    # 1. kanal P1 taşları
    assert np.sum(enc[0]) == 2.0
    # 2. kanal P2 taşları
    assert np.sum(enc[1]) == 2.0


def test_othello_net_forward():
    net = OthelloNet()
    x = torch.randn(4, 3, 8, 8)
    policy_logits, value = net(x)
    assert policy_logits.shape == (4, 65)
    assert value.shape == (4, 1)
    assert torch.all(value >= -1.0) and torch.all(value <= 1.0)


def test_othello_d4_symmetry():
    game = OthelloGame()
    s = game.get_initial_state()
    enc = game.encode_state(s, 1)
    policy = np.zeros(65, dtype=np.float32)
    policy[19] = 0.7
    policy[64] = 0.3  # pass

    history = [(enc, policy, 1)]
    aug = augment_data(history, win_player=1, game_name="othello")

    # 8 simetri (D4)
    assert len(aug) == 8, f"Beklenen 8 simetri, gelen {len(aug)}"
    for x, p, v in aug:
        assert x.shape == (3, 8, 8)
        assert p.shape == (65,)
        assert v == 1.0
        # Pass hamlesi daima sabit kalmalı
        assert np.isclose(p[64], 0.3)
        # Toplam olasılık 1.0 olmalı
        assert np.isclose(np.sum(p), 1.0)


def test_othello_mcts_single_search():
    game = OthelloGame()
    net = OthelloNet()
    net.eval()
    mcts = GenericMCTS(net, game=game, num_simulations=10)

    s = game.get_initial_state()
    policy = mcts.search(s, current_player=1, temperature=1.0)

    assert policy.shape == (65,)
    assert np.isclose(np.sum(policy), 1.0)
    # Yalnızca geçerli hamlelerde olasılık olmalı
    valid = game.get_valid_moves(s, 1)
    assert np.all(policy[~valid] == 0.0)
