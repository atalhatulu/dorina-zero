"""games/othello.py — BaseGame arayüzünü uygulayan 8x8 Othello (Reversi) motoru."""
import numpy as np
from games.base_game import BaseGame

BOARD_SIZE = 8
DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


class OthelloGame(BaseGame):
    def __init__(self):
        self._name = "Othello"
        self._action_space_size = BOARD_SIZE * BOARD_SIZE + 1  # 64 kare + 1 pass hamlesi
        self._board_shape = (2, BOARD_SIZE, BOARD_SIZE)

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
        # 0: Boş, 1: Siyah (Player 1), 2: Beyaz (Player 2)
        b = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        mid = BOARD_SIZE // 2
        b[mid - 1, mid - 1] = 2
        b[mid, mid] = 2
        b[mid - 1, mid] = 1
        b[mid, mid - 1] = 1
        return b

    def _is_on_board(self, r, c):
        return 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE

    def _get_flips(self, board, r, c, player):
        if board[r, c] != 0:
            return []
        opp = 3 - player
        flips = []
        for dr, dc in DIRECTIONS:
            curr_r, curr_c = r + dr, c + dc
            line = []
            while self._is_on_board(curr_r, curr_c) and board[curr_r, curr_c] == opp:
                line.append((curr_r, curr_c))
                curr_r += dr
                curr_c += dc
            if line and self._is_on_board(curr_r, curr_c) and board[curr_r, curr_c] == player:
                flips.extend(line)
        return flips

    def get_valid_moves(self, state, player: int) -> np.ndarray:
        valid = np.zeros(self._action_space_size, dtype=bool)
        has_any = False
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if len(self._get_flips(state, r, c, player)) > 0:
                    valid[r * BOARD_SIZE + c] = True
                    has_any = True
        if not has_any:
            # Geçerli hamle yoksa 'Pass' hamlesi zorunludur (indeks: 64)
            valid[BOARD_SIZE * BOARD_SIZE] = True
        return valid

    def get_next_state(self, state, action: int, player: int):
        new_b = state.copy()
        if action == BOARD_SIZE * BOARD_SIZE:
            # Pass hamlesi
            return new_b

        r = action // BOARD_SIZE
        c = action % BOARD_SIZE
        flips = self._get_flips(new_b, r, c, player)
        new_b[r, c] = player
        for fr, fc in flips:
            new_b[fr, fc] = player
        return new_b

    def check_terminal_value(self, state, player: int):
        v1 = self.get_valid_moves(state, 1)
        v2 = self.get_valid_moves(state, 2)
        # İki taraf da pass geçmek zorundaysa oyun biter
        if v1[BOARD_SIZE * BOARD_SIZE] and v2[BOARD_SIZE * BOARD_SIZE]:
            p1_count = np.count_nonzero(state == 1)
            p2_count = np.count_nonzero(state == 2)
            if p1_count == p2_count:
                return 0.0
            if player == 1:
                return 1.0 if p1_count > p2_count else -1.0
            else:
                return 1.0 if p2_count > p1_count else -1.0
        return None

    def encode_state(self, state, player: int) -> np.ndarray:
        opp = 3 - player
        me_ch = (state == player).astype(np.float32)
        opp_ch = (state == opp).astype(np.float32)
        return np.stack([me_ch, opp_ch], axis=0)

    def clone_state(self, state):
        return state.copy()

    def get_canonical_state(self, state, player: int):
        if player == 1:
            return state.copy()
        new_b = np.zeros_like(state)
        new_b[state == 1] = 2
        new_b[state == 2] = 1
        return new_b

    def state_to_key(self, state, player: int):
        return (state.tobytes(), player)
