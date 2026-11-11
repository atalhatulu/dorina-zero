"""games/connect4.py — BaseGame arayüzünü uygulayan Connect 4 modülü.
Hem 2B liste uyumluluğunu hem de ultra hızlı bitboard desteğini içinde barındırır.
"""
import numpy as np
from games.base_game import BaseGame
import board_fast as bf

ROWS = 6
COLS = 7
EMPTY = 0
PLAYER = 1
AI = 2


class Connect4Game(BaseGame):
    def __init__(self):
        self._name = "Connect4"
        self._action_space_size = COLS
        self._board_shape = (3, ROWS, COLS)

    @property
    def name(self) -> str:
        return self._name

    @property
    def action_space_size(self) -> int:
        return self._action_space_size

    @property
    def board_shape(self) -> tuple:
        return self._board_shape

    def get_initial_state(self):
        """6x7 boş tahta listesi."""
        return [[EMPTY] * COLS for _ in range(ROWS)]

    def get_next_state(self, state, action: int, player: int):
        """Yeni bir tahta kopyası üzerinde hamleyi uygular."""
        new_board = [row[:] for row in state]
        row = bf.get_next_open_row(new_board, action)
        if row is not None:
            bf.drop_piece(new_board, row, action, player)
        return new_board

    def get_valid_moves(self, state, player: int = PLAYER) -> np.ndarray:
        """Geçerli sütunların boolean maskesi (7 elemanlı)."""
        valid = np.zeros(self._action_space_size, dtype=bool)
        for c in range(COLS):
            if bf.is_valid_location(state, c):
                valid[c] = True
        return valid

    def check_terminal_value(self, state, player: int):
        """Oyun bitti mi kontrol eder (+1, -1, 0 veya None)."""
        opp = AI if player == PLAYER else PLAYER
        if bf.winning_move(state, player):
            return 1.0
        if bf.winning_move(state, opp):
            return -1.0
        if bf.is_full(state):
            return 0.0
        return None

    def encode_state(self, state, player: int) -> np.ndarray:
        """(3, 6, 7) float32 tensör formatı."""
        opp = AI if player == PLAYER else PLAYER
        me_ch = np.array([[1.0 if state[r][c] == player else 0.0 for c in range(COLS)] for r in range(ROWS)], dtype=np.float32)
        opp_ch = np.array([[1.0 if state[r][c] == opp else 0.0 for c in range(COLS)] for r in range(ROWS)], dtype=np.float32)
        const_ch = np.ones((ROWS, COLS), dtype=np.float32)
        return np.stack([me_ch, opp_ch, const_ch], axis=0)

    def clone_state(self, state):
        return [row[:] for row in state]

    def get_canonical_state(self, state, player: int):
        """Oyuncu perspektifi."""
        if player == PLAYER:
            return state
        canonical = []
        for row in state:
            new_row = []
            for cell in row:
                if cell == PLAYER:
                    new_row.append(AI)
                elif cell == AI:
                    new_row.append(PLAYER)
                else:
                    new_row.append(EMPTY)
            canonical.append(new_row)
        return canonical

    def state_to_key(self, state, player: int):
        """Transposition table için hash anahtarı."""
        return (tuple(tuple(r) for r in state), player)
