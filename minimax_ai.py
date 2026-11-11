"""Klasik minimax + alpha-beta AI. Baseline/benchmark için."""
import math
import random
from board import (
    ROWS, COLS, EMPTY, PLAYER, AI,
    create_board, drop_piece, get_next_open_row,
    get_valid_locations, winning_move,
)


def evaluate_window(window, piece):
    score = 0
    opp = AI if piece == PLAYER else PLAYER
    if window.count(piece) == 4:
        score += 100
    elif window.count(piece) == 3 and window.count(EMPTY) == 1:
        score += 5
    elif window.count(piece) == 2 and window.count(EMPTY) == 2:
        score += 2
    if window.count(opp) == 3 and window.count(EMPTY) == 1:
        score -= 4
    return score


def score_position(board, piece):
    score = 0
    center = [board[r][COLS // 2] for r in range(ROWS)]
    score += center.count(piece) * 3
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


def minimax(board, depth, alpha, beta, maximizing):
    valid = get_valid_locations(board)
    terminal = winning_move(board, AI) or winning_move(board, PLAYER) or not valid
    if depth == 0 or terminal:
        if terminal:
            if winning_move(board, AI):
                return None, 1_000_000
            if winning_move(board, PLAYER):
                return None, -1_000_000
            return None, 0
        return None, score_position(board, AI)

    if maximizing:
        value, column = -math.inf, random.choice(valid)
        for col in valid:
            row = get_next_open_row(board, col)
            b = [r[:] for r in board]
            drop_piece(b, row, col, AI)
            s = minimax(b, depth - 1, alpha, beta, False)[1]
            if s > value:
                value, column = s, col
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return column, value
    else:
        value, column = math.inf, random.choice(valid)
        for col in valid:
            row = get_next_open_row(board, col)
            b = [r[:] for r in board]
            drop_piece(b, row, col, PLAYER)
            s = minimax(b, depth - 1, alpha, beta, True)[1]
            if s < value:
                value, column = s, col
            beta = min(beta, value)
            if alpha >= beta:
                break
        return column, value


def get_ai_move(board, depth=4):
    """AI'ın (oyuncu AI) oynayacağı sütunu döndürür."""
    col, _ = minimax(board, depth, -math.inf, math.inf, True)
    return col