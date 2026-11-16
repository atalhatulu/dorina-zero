"""games/othello_minimax.py — Othello (Reversi) için Alfa-Beta Budamalı Heuristic Minimax Motoru.

Klasik Othello Usta Stratejileri:
  1. Köşe Hakimiyeti: (0,0), (0,7), (7,0), (7,7) kareleri geri alınamaz, değeri +100'dür.
  2. Tehlikeli Kareler (X-Squares ve C-Squares): Köşe boşken köşenin yanına oynamak intihardır (-40/-20).
  3. Mobilite (Hareket Alanı): Rakibin geçerli hamle sayısını kısıtlamak zaferin anahtarıdır.
  4. Parite ve Kararlılık: Kenarlar ve iç taşlar.
"""
import numpy as np
from games.othello import OthelloGame, BOARD_SIZE

# 8x8 Klasik Othello Pozisyonel Ağırlık Matrisi
POSITION_WEIGHTS = np.array([
    [ 100, -20,  10,   5,   5,  10, -20,  100],
    [ -20, -50,  -2,  -2,  -2,  -2, -50,  -20],
    [  10,  -2,  -1,  -1,  -1,  -1,  -2,   10],
    [   5,  -2,  -1,   0,   0,  -1,  -2,    5],
    [   5,  -2,  -1,   0,   0,  -1,  -2,    5],
    [  10,  -2,  -1,  -1,  -1,  -1,  -2,   10],
    [ -20, -50,  -2,  -2,  -2,  -2, -50,  -20],
    [ 100, -20,  10,   5,   5,  10, -20,  100],
], dtype=np.float32)

CORNERS = [(0, 0), (0, 7), (7, 0), (7, 7)]


def evaluate_board(game: OthelloGame, state: np.ndarray, player: int) -> float:
    """
    Pozisyonel ağırlık, köşe kontrolü ve mobiliteye dayalı değerlendirme fonksiyonu.
    Dönüş: `player` perspektifinden skor (+ büyükse avantaj, - ise dezavantaj).
    """
    opp = 3 - player

    # 1. Pozisyonel Ağırlık Skoru
    my_tiles = (state == player)
    opp_tiles = (state == opp)
    pos_score = np.sum(POSITION_WEIGHTS * my_tiles) - np.sum(POSITION_WEIGHTS * opp_tiles)

    # 2. Köşe Kontrolü (Ekstra Güçlü Çarpan)
    my_corners = sum(1 for r, c in CORNERS if state[r, c] == player)
    opp_corners = sum(1 for r, c in CORNERS if state[r, c] == opp)
    corner_score = 30.0 * (my_corners - opp_corners)

    # 3. Mobilite (Geçerli Hamle Sayısı)
    my_valid = game.get_valid_moves(state, player)
    opp_valid = game.get_valid_moves(state, opp)
    my_moves = np.count_nonzero(my_valid[:64])
    opp_moves = np.count_nonzero(opp_valid[:64])

    mobility_score = 0.0
    if my_moves + opp_moves > 0:
        mobility_score = 10.0 * (my_moves - opp_moves) / (my_moves + opp_moves)

    # 4. Oyun Sonu Taş Sayısı Ağırlığı (Tahta doldukça taş sayısı önem kazanır)
    total_disks = np.count_nonzero(state != 0)
    disk_parity = 0.0
    if total_disks > 50:
        disk_parity = 2.0 * (np.sum(my_tiles) - np.sum(opp_tiles))

    return float(pos_score + corner_score + mobility_score + disk_parity)


def minimax(game: OthelloGame, state: np.ndarray, depth: int, alpha: float, beta: float,
            maximizing_player: bool, root_player: int) -> tuple:
    """
    Alfa-Beta Budamalı Minimax Arama Algoritması.
    Dönüş: (en_iyi_skor, en_iyi_aksiyon)
    """
    current_player = root_player if maximizing_player else (3 - root_player)

    # Terminal Kontrolü
    terminal_val = game.check_terminal_value(state, current_player)
    if terminal_val is not None:
        if terminal_val == 0.0:
            return 0.0, None
        # Oyun bittiyse çok büyük skor dön
        score = 10000.0 if terminal_val > 0 else -10000.0
        return (score if maximizing_player else -score), None

    if depth == 0:
        return evaluate_board(game, state, root_player), None

    valid_mask = game.get_valid_moves(state, current_player)
    valid_actions = [int(i) for i in np.where(valid_mask)[0]]

    # Geçerli hamle yoksa (sadece pass varsa)
    if len(valid_actions) == 1 and valid_actions[0] == 64:
        next_state = game.get_next_state(state, 64, current_player)
        score, _ = minimax(game, next_state, depth - 1, alpha, beta, not maximizing_player, root_player)
        return score, 64

    # Hamle Sıralama (Köşeleri ve yüksek ağırlıklı kareleri önce dene -> Alfa-Beta hızlandırıcı)
    def action_priority(a):
        if a == 64:
            return -999
        r, c = a // BOARD_SIZE, a % BOARD_SIZE
        return POSITION_WEIGHTS[r, c]

    valid_actions.sort(key=action_priority, reverse=maximizing_player)

    best_action = valid_actions[0]

    if maximizing_player:
        max_eval = -float("inf")
        for action in valid_actions:
            next_state = game.get_next_state(state, action, current_player)
            eval_score, _ = minimax(game, next_state, depth - 1, alpha, beta, False, root_player)
            if eval_score > max_eval:
                max_eval = eval_score
                best_action = action
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break
        return max_eval, best_action
    else:
        min_eval = float("inf")
        for action in valid_actions:
            next_state = game.get_next_state(state, action, current_player)
            eval_score, _ = minimax(game, next_state, depth - 1, alpha, beta, True, root_player)
            if eval_score < min_eval:
                min_eval = eval_score
                best_action = action
            beta = min(beta, eval_score)
            if beta <= alpha:
                break
        return min_eval, best_action


def get_othello_ai_move(state: np.ndarray, depth: int = 4, ai_player: int = 2) -> int:
    """
    Dışarıdan çağrılan ana Minimax arayüzü.
    `state`: 8x8 numpy array
    `depth`: Arama derinliği (varsayılan 4)
    `ai_player`: Hangi oyuncu için hamle seçileceği (1 veya 2)
    """
    game = OthelloGame()
    valid_mask = game.get_valid_moves(state, ai_player)
    valid_actions = [int(i) for i in np.where(valid_mask)[0]]

    if not valid_actions:
        return 64  # Pass

    if len(valid_actions) == 1:
        return valid_actions[0]

    _, best_action = minimax(
        game=game,
        state=state,
        depth=depth,
        alpha=-float("inf"),
        beta=float("inf"),
        maximizing_player=True,
        root_player=ai_player
    )
    return best_action if best_action is not None else valid_actions[0]
