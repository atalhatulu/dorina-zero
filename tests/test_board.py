# tests/test_board.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from board import (
    create_board, drop_piece, is_valid_location, get_next_open_row,
    get_valid_locations, winning_move, is_full, is_terminal, clone_board,
    ROWS, COLS, EMPTY, PLAYER, AI,
)


# --- Temel ---

def test_create_board():
    b = create_board()
    assert len(b) == ROWS
    assert all(len(row) == COLS for row in b)
    assert all(cell == EMPTY for row in b for cell in row)


def test_drop_piece():
    b = create_board()
    drop_piece(b, 5, 3, PLAYER)
    assert b[5][3] == PLAYER


def test_clone_board_independent():
    b = create_board()
    drop_piece(b, 5, 0, PLAYER)
    b2 = clone_board(b)
    drop_piece(b2, 5, 0, AI)
    assert b[5][0] == PLAYER   # orijinal değişmedi
    assert b2[5][0] == AI


# --- Geçerlilik ---

def test_is_valid_location_empty():
    b = create_board()
    assert all(is_valid_location(b, c) for c in range(COLS))


def test_is_valid_location_full_column():
    b = create_board()
    for r in range(ROWS):
        drop_piece(b, r, 0, PLAYER)
    assert not is_valid_location(b, 0)
    assert is_valid_location(b, 1)


def test_get_next_open_row_empty():
    b = create_board()
    assert get_next_open_row(b, 0) == ROWS - 1  # en alt


def test_get_next_open_row_partial():
    b = create_board()
    drop_piece(b, ROWS - 1, 0, PLAYER)  # en alta koy
    assert get_next_open_row(b, 0) == ROWS - 2


def test_get_valid_locations_full():
    b = create_board()
    for r in range(ROWS):
        drop_piece(b, r, 3, PLAYER)
    valid = get_valid_locations(b)
    assert 3 not in valid
    assert len(valid) == COLS - 1


# --- Kazanma ---

def test_winning_horizontal():
    b = create_board()
    for c in range(4):
        drop_piece(b, ROWS - 1, c, PLAYER)
    assert winning_move(b, PLAYER)
    assert not winning_move(b, AI)


def test_winning_vertical():
    b = create_board()
    for r in range(ROWS - 4, ROWS):
        drop_piece(b, r, 2, AI)
    assert winning_move(b, AI)
    assert not winning_move(b, PLAYER)


def test_winning_diagonal_down_right():
    b = create_board()
    # (5,0), (4,1), (3,2), (2,3)
    for i in range(4):
        drop_piece(b, ROWS - 1 - i, i, PLAYER)
    assert winning_move(b, PLAYER)


def test_winning_diagonal_up_right():
    b = create_board()
    # (2,0), (3,1), (4,2), (5,3)
    for i in range(4):
        drop_piece(b, 2 + i, i, AI)
    assert winning_move(b, AI)


def test_no_win_three_in_row():
    b = create_board()
    for c in range(3):
        drop_piece(b, ROWS - 1, c, PLAYER)
    assert not winning_move(b, PLAYER)


# --- Terminal ---

def test_not_full():
    b = create_board()
    assert not is_full(b)


def test_full_board():
    b = create_board()
    for r in range(ROWS):
        for c in range(COLS):
            drop_piece(b, r, c, PLAYER if (r + c) % 2 == 0 else AI)
    assert is_full(b)


def test_is_terminal_win():
    b = create_board()
    for c in range(4):
        drop_piece(b, ROWS - 1, c, PLAYER)
    assert is_terminal(b)


def test_is_terminal_not():
    b = create_board()
    drop_piece(b, ROWS - 1, 0, PLAYER)
    assert not is_terminal(b)