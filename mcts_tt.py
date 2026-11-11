"""mcts_tt.py — Transposition Table (LRU Cache) destekli MCTS prototipi.
Mevcut mcts.py'yi bozmadan test etmek için izole hazırlandı.
"""
import math
import collections
import numpy as np
import torch

from board import (
    ROWS, COLS, EMPTY, PLAYER, AI,
    create_board, drop_piece, get_next_open_row, get_valid_locations,
    winning_move, is_full, clone_board,
)
from network import board_to_tensor, predict
from mcts import MCTSNode, sample_move


def board_key(board, current_player):
    """Hızlı ve benzersiz hash anahtarı (tuple of tuples + player)."""
    return (tuple(tuple(r) for r in board), current_player)


class TranspositionTable:
    """Sabit boyutlu LRU Cache (OrderedDict tabanlı)."""
    def __init__(self, maxsize=20_000):
        self.maxsize = maxsize
        self.cache = collections.OrderedDict()

    def get(self, key):
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.maxsize:
            self.cache.popitem(last=False)

    def clear(self):
        self.cache.clear()

    def __len__(self):
        return len(self.cache)


class MCTSWithTT:
    def __init__(self, model, num_simulations=200, c_puct=1.5, tt_size=20_000):
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.tt = TranspositionTable(maxsize=tt_size)

    def _terminal_value(self, board, current_player):
        opp = AI if current_player == PLAYER else PLAYER
        if winning_move(board, current_player):
            return 1.0
        if winning_move(board, opp):
            return -1.0
        if is_full(board):
            return 0.0
        return None

    def _expand(self, node, board, current_player):
        terminal = self._terminal_value(board, current_player)
        if terminal is not None:
            node.is_expanded = True
            return terminal

        # Transposition Table kontrolü
        key = board_key(board, current_player)
        cached = self.tt.get(key)
        if cached is not None:
            policy, v = cached
        else:
            with torch.no_grad():
                x = torch.from_numpy(board_to_tensor(board, current_player)).unsqueeze(0)
                policy_logits, value = self.model(x)
                policy = torch.softmax(policy_logits, dim=1).squeeze(0).numpy()
                v = value.item()
            self.tt.put(key, (policy, v))

        valid = get_valid_locations(board)
        masked = np.zeros(COLS, dtype=np.float32)
        for c in valid:
            masked[c] = policy[c]
        s = masked.sum()
        if s > 0:
            masked /= s
        else:
            for c in valid:
                masked[c] = 1.0 / len(valid)

        for c in valid:
            node.children[c] = MCTSNode(prior=masked[c], player_just_moved=current_player)
        node.is_expanded = True
        return v

    def _select(self, node):
        best_score = -float("inf")
        best_col = None
        for col, child in node.children.items():
            if child.visits == 0:
                q = 0.0
            else:
                q = -child.q_value()
            u = self.c_puct * child.prior * (math.sqrt(node.visits) / (1 + child.visits))
            score = q + u
            if score > best_score:
                best_score = score
                best_col = col
        return best_col

    def _simulate(self, root, root_board, root_player):
        path = [root]
        current_board = clone_board(root_board)
        current_player = root_player

        node = root
        while not node.is_leaf():
            col = self._select(node)
            if col is None:
                break
            row = get_next_open_row(current_board, col)
            drop_piece(current_board, row, col, current_player)
            current_player = AI if current_player == PLAYER else PLAYER
            node = node.children[col]
            path.append(node)

        value = self._expand(node, current_board, current_player)

        for n in reversed(path):
            node_player = n.player_just_moved
            v_for_node = value if node_player == current_player else -value
            n.visits += 1
            n.value_sum += v_for_node

    def search(self, board, current_player, temperature=1.0,
               dirichlet_alpha=1.0, dirichlet_epsilon=0.10):
        if winning_move(board, PLAYER) or winning_move(board, AI) or is_full(board):
            probs = np.zeros(COLS, dtype=np.float32)
            valid = get_valid_locations(board)
            if valid:
                for c in valid:
                    probs[c] = 1.0 / len(valid)
            return probs

        root = MCTSNode(prior=0.0, player_just_moved=AI if current_player == PLAYER else PLAYER)
        self._expand(root, board, current_player)
        root.visits = 1

        if dirichlet_epsilon > 0 and len(root.children) > 0:
            noise = np.random.dirichlet([dirichlet_alpha] * len(root.children))
            for i, child in enumerate(root.children.values()):
                child.prior = ((1 - dirichlet_epsilon) * child.prior + dirichlet_epsilon * noise[i])

        for _ in range(self.num_simulations):
            self._simulate(root, board, current_player)

        visits = np.zeros(COLS, dtype=np.float32)
        for col, child in root.children.items():
            visits[col] = child.visits

        if visits.sum() == 0:
            valid = get_valid_locations(board)
            for c in valid:
                visits[c] = 1.0

        if temperature == 0:
            probs = np.zeros(COLS, dtype=np.float32)
            probs[np.argmax(visits)] = 1.0
            return probs

        visits_t = visits ** (1.0 / temperature)
        probs = visits_t / visits_t.sum()
        return probs

    def best_move(self, board, current_player):
        probs = self.search(board, current_player, temperature=0)
        return int(np.argmax(probs))


if __name__ == "__main__":
    import time
    from network import Connect4Net
    from board import create_board

    model = Connect4Net()
    model.eval()

    mcts_tt = MCTSWithTT(model, num_simulations=300)
    b = create_board()

    # 1. Çağrı (Cache boş - Soğuk başlangıç)
    t0 = time.time()
    p1 = mcts_tt.search(b, PLAYER, temperature=0)
    dt1 = time.time() - t0

    # 2. Çağrı (Cache dolu - Sıcak başlangıç)
    t0 = time.time()
    p2 = mcts_tt.search(b, PLAYER, temperature=0)
    dt2 = time.time() - t0

    print(f"1. Arama (Soğuk): {dt1*1000:.2f} ms")
    print(f"2. Arama (Cache Aktif): {dt2*1000:.2f} ms")
    print(f"Cache boyutu: {len(mcts_tt.tt)} pozisyon")
    print(f"Hızlanma Oranı: {dt1/max(dt2, 1e-6):.2f}x kat")
