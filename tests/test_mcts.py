import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest

from board import (
    create_board, drop_piece, get_next_open_row,
    ROWS, COLS, PLAYER, AI, winning_move,
)
from network import Connect4Net
from mcts import MCTS, sample_move


@pytest.fixture
def model():
    return Connect4Net()


@pytest.fixture
def mcts(model):
    return MCTS(model, num_simulations=30, c_puct=1.5)


def test_search_returns_valid_distribution(mcts):
    b = create_board()
    probs = mcts.search(b, PLAYER)
    assert probs.shape == (COLS,)
    assert abs(probs.sum() - 1.0) < 1e-4
    assert np.all(probs >= 0)


def test_search_only_valid_columns(mcts):
    b = create_board()
    # Sütun 0'ı tamamen doldur
    for r in range(ROWS):
        drop_piece(b, r, 0, PLAYER)
    probs = mcts.search(b, AI)
    assert probs[0] == 0.0
    assert probs[1:].sum() > 0.99


def test_best_move_takes_winning_move():
    """AI 3 taş yan yana — MCTS 4.'yü oynamalı."""
    model = Connect4Net()
    mcts = MCTS(model, num_simulations=100)
    b = create_board()
    for c in range(3):
        row = get_next_open_row(b, c)
        drop_piece(b, row, c, AI)
    move = mcts.best_move(b, AI)
    assert move == 3, f"MCTS kazanmalıydı, sütun {move} seçti"


def test_best_move_blocks_opponent_win():
    """Player 3 taş yan yana — MCTS 4.'yü engellemeli."""
    model = Connect4Net()
    mcts = MCTS(model, num_simulations=100)
    b = create_board()
    for c in range(3):
        row = get_next_open_row(b, c)
        drop_piece(b, row, c, PLAYER)
    move = mcts.best_move(b, AI)
    assert move == 3, f"MCTS engellemeliydi, sütun {move} seçti"


def test_terminal_board_search():
    """Oyun bitmiş tahtada search çökmemeli."""
    model = Connect4Net()
    mcts = MCTS(model, num_simulations=10)
    b = create_board()
    for c in range(4):
        row = get_next_open_row(b, c)
        drop_piece(b, row, c, PLAYER)  # PLAYER kazandı
    # Bu tahtada AI'ın sırası yok, ama search çağrılabilmeli
    probs = mcts.search(b, AI)
    assert probs.shape == (COLS,)


def test_sample_move_respects_probs():
    rng = np.random.default_rng(42)
    probs = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    for _ in range(10):
        assert sample_move(probs, rng) == 2


def test_search_temperature_zero_deterministic(mcts):
    b = create_board()
    p1 = mcts.search(b, PLAYER, temperature=0)
    p2 = mcts.search(b, PLAYER, temperature=0)
    # temperature=0 → one-hot, aynı tahtada aynı sonuç
    # (rastgelelik yoksa)
    assert np.argmax(p1) == np.argmax(p2)