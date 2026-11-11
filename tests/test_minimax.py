import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from board import (
    create_board, drop_piece, get_next_open_row,
    ROWS, COLS, PLAYER, AI,
)
from minimax_ai import get_ai_move, score_position


def test_ai_returns_valid_column():
    b = create_board()
    col = get_ai_move(b)
    assert col is not None
    assert 0 <= col < COLS


def test_ai_blocks_immediate_win():
    """Player 3 taş yan yana — AI 4.'yü engellemeli."""
    b = create_board()
    # Player alt sırada 0,1,2 sütunlarına koysun
    for c in range(3):
        row = get_next_open_row(b, c)
        drop_piece(b, row, c, PLAYER)
    # AI'ın 3. sütunu kapatması lazım
    col = get_ai_move(b, depth=4)
    assert col == 3, f"AI 3. sütunu kapatmalıydı, {col} seçti"


def test_ai_takes_winning_move():
    """AI 3 taş yan yana — 4.'yü oynamalı (kazanmalı)."""
    b = create_board()
    for c in range(3):
        row = get_next_open_row(b, c)
        drop_piece(b, row, c, AI)
    col = get_ai_move(b, depth=4)
    assert col == 3, f"AI 3. sütunda kazanmalıydı, {col} seçti"


def test_score_position_empty():
    b = create_board()
    # Boş tahtada skor 0 olmalı (AI perspektifinden)
    assert score_position(b, AI) == 0