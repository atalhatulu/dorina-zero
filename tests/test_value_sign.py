"""tests/test_value_sign.py — Value Perspective & Sign Bütünlük Testi."""
import numpy as np
import torch
import pytest

from games.connect4 import Connect4Game, PLAYER, AI
from network_v2 import Connect4NetV2
from core.mcts import GenericMCTS


def test_encode_state_symmetry():
    """encode_state oyuncu değişiminde kanalları doğru takas etmeli."""
    game = Connect4Game()
    state = game.get_initial_state()
    state = game.get_next_state(state, 3, PLAYER)  # Player 1 ortaya taş attı

    # Player 1 perspektifinden
    enc_p1 = game.encode_state(state, PLAYER)
    # Player 2 perspektifinden
    enc_p2 = game.encode_state(state, AI)

    # p1 kanal 0 (kendi taşları) == p2 kanal 1 (rakip taşları) olmalı
    assert np.array_equal(enc_p1[0], enc_p2[1]), "Kanal 0 ve Kanal 1 perspektif takası hatalı!"
    assert np.array_equal(enc_p1[1], enc_p2[0]), "Kanal 1 ve Kanal 0 perspektif takası hatalı!"


def test_terminal_value_perspective():
    """Terminal değer her zaman 'current_player' perspektifinden olmalı."""
    game = Connect4Game()
    state = game.get_initial_state()

    # Player 1'e 4'lü kazanç diz
    for r in range(4):
        state[r][0] = PLAYER

    # Player 1 için terminal kontrolü (+1 olmalı)
    assert game.check_terminal_value(state, PLAYER) == 1.0
    # Player 2 (AI) için terminal kontrolü (-1 olmalı)
    assert game.check_terminal_value(state, AI) == -1.0


def test_mcts_root_value_symmetry():
    """MCTS simülasyonunda anlık kazanç ve zorunlu blokaj pozisyonlarında politika doğru sütunu seçmelidir."""
    game = Connect4Game()
    model = Connect4NetV2()
    model.eval()

    # Durum 1: P1'in sütun 0'da 3'lüsü var, sıra P1'de -> sütun 0 ile kazanmalı
    state_win = game.get_initial_state()
    for _ in range(3):
        state_win = game.get_next_state(state_win, 0, PLAYER)
        state_win = game.get_next_state(state_win, 1, AI)

    mcts_p1 = GenericMCTS(model, game, num_simulations=50)
    pi_p1 = mcts_p1.search(state_win, PLAYER, temperature=0.0, dirichlet_epsilon=0.0)
    assert int(np.argmax(pi_p1)) == 0, f"P1 anlık kazancı kaçırdı: {pi_p1}"

    # Durum 2: P1'in sütun 0'da 3'lüsü var, P2'nin 3'lüsü yok, sıra P2'de -> sütun 0'ı bloklamalı
    state_block = game.get_initial_state()
    p2_cols = [1, 2, 1]
    for c_p2 in p2_cols:
        state_block = game.get_next_state(state_block, 0, PLAYER)
        state_block = game.get_next_state(state_block, c_p2, AI)

    mcts_p2 = GenericMCTS(model, game, num_simulations=100)
    pi_p2 = mcts_p2.search(state_block, AI, temperature=0.0, dirichlet_epsilon=0.0)
    assert int(np.argmax(pi_p2)) == 0, f"P2 zorunlu blokajı kaçırdı: {pi_p2}"


def test_minimax_both_players():
    """Minimax AI hem PLAYER (1) hem AI (2) olarak anlık kazancı görmelidir."""
    from minimax_ai import get_ai_move

    game = Connect4Game()
    state = game.get_initial_state()
    # P1 sütun 0'da 3 taş, P2 (AI) sütun 1'de 3 taş
    for _ in range(3):
        state = game.get_next_state(state, 0, PLAYER)
        state = game.get_next_state(state, 1, AI)

    assert get_ai_move(game.clone_state(state), depth=2, ai_player=PLAYER) == 0
    assert get_ai_move(game.clone_state(state), depth=2, ai_player=AI) == 1


def test_mcts_deterministic_when_temp_zero():
    """temperature=0.0 iken Dirichlet gürültüsü uygulanmamalı ve sonuç deterministik olmalıdır."""
    game = Connect4Game()
    state = game.get_initial_state()
    model = Connect4NetV2()
    model.eval()

    mcts = GenericMCTS(model, game, num_simulations=30)
    pi1 = mcts.search(state, PLAYER, temperature=0.0, dirichlet_epsilon=0.25)
    mcts.tt.clear()
    pi2 = mcts.search(state, PLAYER, temperature=0.0, dirichlet_epsilon=0.25)
    np.testing.assert_array_equal(pi1, pi2)
