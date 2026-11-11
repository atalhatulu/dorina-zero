# board.py
"""Connect4 kural motoru. Saf mantık — pygame/UI bağımlılığı yok."""

ROWS, COLS = 6, 7
EMPTY, PLAYER, AI = 0, 1, 2


def create_board():
    """6x7 boş tahta döndürür (0 = boş)."""
    return [[EMPTY] * COLS for _ in range(ROWS)]


def drop_piece(board, row, col, piece):
    """Taşı (row, col) hücresine koyar. Yerinde değiştirir."""
    board[row][col] = piece


def is_valid_location(board, col):
    """Sütunun en üstü boşsa True."""
    return board[0][col] == EMPTY


def get_next_open_row(board, col):
    """Sütunda taşın düşeceği satırı döndürür (en alt boş)."""
    for r in range(ROWS - 1, -1, -1):
        if board[r][col] == EMPTY:
            return r
    return None  # sütun dolu


def get_valid_locations(board):
    """Geçerli sütunların listesi."""
    return [c for c in range(COLS) if is_valid_location(board, c)]


def winning_move(board, piece):
    """`piece` oyuncusu 4'lü yaptı mı?"""
    # Yatay
    for c in range(COLS - 3):
        for r in range(ROWS):
            if all(board[r][c + i] == piece for i in range(4)):
                return True
    # Dikey
    for c in range(COLS):
        for r in range(ROWS - 3):
            if all(board[r + i][c] == piece for i in range(4)):
                return True
    # Çapraz (aşağı-sağ)
    for c in range(COLS - 3):
        for r in range(ROWS - 3):
            if all(board[r + i][c + i] == piece for i in range(4)):
                return True
    # Çapraz (yukarı-sağ)
    for c in range(COLS - 3):
        for r in range(3, ROWS):
            if all(board[r - i][c + i] == piece for i in range(4)):
                return True
    return False


def is_full(board):
    """Tahta tamamen dolu mu?"""
    return all(board[0][c] != EMPTY for c in range(COLS))


def is_terminal(board):
    """Oyun bitti mi? (kazanan var veya berabere)"""
    return (
        winning_move(board, PLAYER)
        or winning_move(board, AI)
        or is_full(board)
    )


def clone_board(board):
    """Tahtanın derin kopyası (MCTS için şart)."""
    return [row[:] for row in board]
