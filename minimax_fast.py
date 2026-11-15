"""Optimize Edilmis Transpozisyon Tablolu ve Hizlandirilmis Minimax AI (Derinlik 8 Destegi)."""
import math
import random
from board import (
    ROWS, COLS, EMPTY, PLAYER, AI,
    drop_piece, get_next_open_row,
    get_valid_locations, winning_move,
)

# Transposition Table (Pozisyon Hafizasi)
# Anahtar: (tahta_tuple, oyuncu), Deger: (deger, en_iyi_kolon, derinlik, bayrak)
_TRANSPOSITION_TABLE = {}
_MAX_TT_SIZE = 1_000_000


def evaluate_window(window, piece):
    score = 0
    opp = AI if piece == PLAYER else PLAYER
    if window.count(piece) == 4:
        score += 100
    elif window.count(piece) == 3 and window.count(EMPTY) == 1:
        score += 8
    elif window.count(piece) == 2 and window.count(EMPTY) == 2:
        score += 3
    if window.count(opp) == 3 and window.count(EMPTY) == 1:
        score -= 6
    return score


def score_position(board, piece):
    score = 0
    center = [board[r][COLS // 2] for r in range(ROWS)]
    score += center.count(piece) * 4
    for r in range(ROWS):
        for c in range(COLS - 3):
            score += evaluate_window(board[r][c:c + 4], piece)
    for c in range(COLS):
        col_arr = [board[r][c] for r in range(ROWS)]
        for r in range(ROWS - 3):
            score += evaluate_window(col_arr[r:r + 4], piece)
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            score += evaluate_window([board[r + i][c + i] for i in range(4)], piece)
            score += evaluate_window([board[r + 3 - i][c + i] for i in range(4)], piece)
    return score


def _order_moves(valid_locations):
    # Connect 4'te merkez sutunlar her zaman daha guclu oldugu icin once merkeze bakar
    # Merkez (3), sonra 2 ve 4, sonra 1 ve 5, en son 0 ve 6
    center_order = [3, 2, 4, 1, 5, 0, 6]
    return [c for c in center_order if c in valid_locations]


def minimax_fast(board, depth, alpha, beta, maximizing, ai_player=AI):
    opp_player = PLAYER if ai_player == AI else AI
    valid = get_valid_locations(board)
    terminal = winning_move(board, ai_player) or winning_move(board, opp_player) or not valid

    if depth == 0 or terminal:
        if terminal:
            if winning_move(board, ai_player):
                return None, 1_000_000 + depth  # Daha erken galibiyet tercih edilir
            if winning_move(board, opp_player):
                return None, -1_000_000 - depth # Daha gec maglubiyet tercih edilir
            return None, 0
        return None, score_position(board, ai_player)

    ordered_moves = _order_moves(valid)

    # Transposition Table kontrolu
    board_key = (tuple(tuple(r) for r in board), maximizing, ai_player)
    if board_key in _TRANSPOSITION_TABLE:
        stored_depth, stored_val, stored_col = _TRANSPOSITION_TABLE[board_key]
        if stored_depth >= depth:
            return stored_col, stored_val

    if maximizing:
        value, column = -math.inf, ordered_moves[0]
        for col in ordered_moves:
            row = get_next_open_row(board, col)
            b = [r[:] for r in board]
            drop_piece(b, row, col, ai_player)
            s = minimax_fast(b, depth - 1, alpha, beta, False, ai_player)[1]
            if s > value:
                value, column = s, col
            alpha = max(alpha, value)
            if alpha >= beta:
                break
    else:
        value, column = math.inf, ordered_moves[0]
        for col in ordered_moves:
            row = get_next_open_row(board, col)
            b = [r[:] for r in board]
            drop_piece(b, row, col, opp_player)
            s = minimax_fast(b, depth - 1, alpha, beta, True, ai_player)[1]
            if s < value:
                value, column = s, col
            beta = min(beta, value)
            if alpha >= beta:
                break

    if len(_TRANSPOSITION_TABLE) < _MAX_TT_SIZE:
        _TRANSPOSITION_TABLE[board_key] = (depth, value, column)

    return column, value


def get_ai_move_v2(board, depth=8, ai_player=AI):
    """Transpozisyon tablosu ve merkez siralamali hizli Minimax."""
    col, _ = minimax_fast(board, depth, -math.inf, math.inf, True, ai_player)
    return col
