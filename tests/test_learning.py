"""Modelin gerçekten öğrendiğini doğrulayan testler."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import torch
import numpy as np

from board import (
    create_board, drop_piece, get_next_open_row,
    PLAYER, AI, ROWS, COLS,
)
from network import Connect4Net, predict


# --- Hızlı testler ---

def test_untrained_model_gives_near_zero():
    """Eğitilmemiş model her pozisyona ~0 vermeli."""
    model = Connect4Net()
    b = create_board()
    _, v = predict(model, b, PLAYER)
    assert abs(v) < 0.2, f"Rastgele init value={v:.3f}, ~0 olmalı"


def test_untrained_model_winning_not_learned():
    """Eğitilmemiş model kazanacak pozisyonu bilmemeli (~0)."""
    model = Connect4Net()
    b = create_board()
    for c in range(3):
        row = get_next_open_row(b, c)
        drop_piece(b, row, c, AI)
    _, v = predict(model, b, AI)
    # Eğitilmemiş: -0.2..+0.2 arası (öğrenmemiş)
    assert abs(v) < 0.2, f"Eğitilmemiş model value={v:.3f}, ~0 olmalı"


# --- Yavaş test (eğitim gerektirir) ---
@pytest.mark.slow
def test_model_after_training_learns_winning():
    """
    Kısa eğitim sonrası model, kazanacak/kaybedecek pozisyonları ayırt etmeli.
    Not: Bu test yavaş (~20-30s). CI'da 'slow' marker ile işaretlenebilir.
    """
    from self_play import play_games
    from train import train

    # Hızlı eğitim verisi topla
    model = Connect4Net()
    all_data = []
    for _ in range(3):
        data, _ = play_games(model, num_games=10, num_simulations=20, log=False)
        all_data.extend(data)

    # Eğit
    for _ in range(3):
        train(model, all_data, epochs=3, batch_size=32, lr=2e-3)

    # Test 1: AI 3 taş yan yana → kazanacak, value pozitif olmalı
    b = create_board()
    for c in range(3):
        row = get_next_open_row(b, c)
        drop_piece(b, row, c, AI)
    _, v_win = predict(model, b, AI)
    assert v_win > 0.02, f"Kazanacak pozisyonda value={v_win:.3f}, pozitif olmalı"

    # Test 2: PLAYER 3 taş yan yana → AI kaybedecek, value negatif olmalı
    b2 = create_board()
    for c in range(3):
        row = get_next_open_row(b2, c)
        drop_piece(b2, row, c, PLAYER)
    _, v_lose = predict(model, b2, AI)
    assert v_lose < -0.02, f"Kaybedecek pozisyonda value={v_lose:.3f}, negatif olmalı"


# --- Ek: Symmetry testi (yeni) ---

def test_symmetry_augmentation_correct():
    """Symmetry augmentation doğru flip yapıyor mu?"""
    from self_play import history_to_training_data

    b = create_board()
    row = get_next_open_row(b, 0)
    drop_piece(b, row, 0, PLAYER)

    policy = np.zeros(COLS, dtype=np.float32)
    policy[0] = 1.0

    history = [(b, PLAYER, policy)]
    data = history_to_training_data(history, "player1")

    assert len(data) == 2, "Symmetry 2x örnek üretmeli"

    # Orijinal: sol sütunda taş
    orig_board, _, orig_policy, _ = data[0]
    assert orig_board[row][0] == PLAYER
    assert orig_policy[0] == 1.0

    # Flipped: sağ sütunda taş
    flip_board, _, flip_policy, _ = data[1]
    assert flip_board[row][COLS - 1] == PLAYER
    assert flip_policy[COLS - 1] == 1.0


# --- Ek: Dirichlet noise testi (yeni) ---

def test_dirichlet_noise_only_in_training():
    """Dirichlet noise sadece temperature > 0'da uygulanmalı."""
    from mcts import MCTS

    model = Connect4Net()
    mcts = MCTS(model, num_simulations=10)
    b = create_board()

    # temperature=0 → noise yok (deterministik)
    p1 = mcts.search(b, PLAYER, temperature=0)
    p2 = mcts.search(b, PLAYER, temperature=0)
    assert np.argmax(p1) == np.argmax(p2), "temperature=0 deterministik olmalı"

    # temperature=1 → noise var (stokastik)
    # Aynı tahtada 2 çağrı farklı sonuç verebilir
    p3 = mcts.search(b, PLAYER, temperature=1)
    p4 = mcts.search(b, PLAYER, temperature=1)
    # En azından geçerli dağılım olmalı
    assert abs(p3.sum() - 1.0) < 1e-4
    assert abs(p4.sum() - 1.0) < 1e-4
