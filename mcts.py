"""Monte Carlo Tree Search — AlphaZero tarzı (PUCT + NN değerlendirme)."""
import math
import numpy as np
import torch

from board import (
    ROWS, COLS, EMPTY, PLAYER, AI,
    get_next_open_row, get_valid_locations,
    winning_move, is_full, clone_board,
)
from network import board_to_tensor, predict


class MCTSNode:
    __slots__ = ("prior", "visits", "value_sum", "children", "is_expanded", "player_just_moved")

    def __init__(self, prior, player_just_moved):
        self.prior = prior
        self.visits = 0
        self.value_sum = 0.0
        self.children = {}
        self.is_expanded = False
        self.player_just_moved = player_just_moved

    def q_value(self):
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits

    def is_leaf(self):
        return not self.is_expanded


class MCTS:
    def __init__(self, model, num_simulations=200, c_puct=1.5):
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct

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

        with torch.no_grad():
            x = torch.from_numpy(board_to_tensor(board, current_player)).unsqueeze(0)
            policy_logits, value = self.model(x)
            policy = torch.softmax(policy_logits, dim=1).squeeze(0).numpy()
            v = value.item()

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
        best_child = None
        total_visits = sum(child.visits for child in node.children.values())
        sqrt_total = math.sqrt(total_visits + 1)

        for col, child in node.children.items():
            q = child.q_value()
            u = self.c_puct * child.prior * sqrt_total / (1 + child.visits)
            score = q + u
            if score > best_score:
                best_score = score
                best_col = col
                best_child = child

        if best_child is None:
            raise RuntimeError("_select: node has no children (should not happen)")
        return best_col, best_child

    def _simulate(self, root, root_board, root_player):
        node = root
        board = clone_board(root_board)
        current_player = root_player
        path = [node]

        while not node.is_leaf():
            if not node.children:
                break
            col, node = self._select(node)
            row = get_next_open_row(board, col)
            board[row][col] = current_player
            current_player = AI if current_player == PLAYER else PLAYER
            path.append(node)

        if node.is_expanded and not node.children:
            value = self._terminal_value(board, current_player)
            if value is None:
                value = 0.0
        else:
            value = self._expand(node, board, current_player)

        for n in reversed(path):
            node_player = n.player_just_moved
            v_for_node = value if node_player == current_player else -value
            n.visits += 1
            n.value_sum += v_for_node

    def search(self, board, current_player, temperature=1.0,
               dirichlet_alpha=1.0, dirichlet_epsilon=0.10):
        # Oyun bittiyse: hamle yok, geçerli sütunlara eşit dağıt
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
        root.value_sum = 0.0

        # Dirichlet noise (sadece eğitimde, temperature > 0 ise)
        if temperature > 0 and len(root.children) > 0:
            valid_cols = list(root.children.keys())
            noise = np.random.dirichlet([dirichlet_alpha] * len(valid_cols))
            for i, col in enumerate(valid_cols):
                child = root.children[col]
                child.prior = ((1 - dirichlet_epsilon) * child.prior
                               + dirichlet_epsilon * noise[i])

        for _ in range(self.num_simulations - 1):
            self._simulate(root, board, current_player)

        visits = np.zeros(COLS, dtype=np.float32)
        for col, child in root.children.items():
            visits[col] = child.visits

        if visits.sum() == 0:
            valid = get_valid_locations(board)
            for c in valid:
                visits[c] = 1.0
            visits /= visits.sum()
            return visits

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

    def evaluate_board(self, board, current_player):
        """
        MCTS yapmadan tek bir forward pass ile tahtayı hızlıca değerlendirir.
        Dönüş: (value, win_probability)
          value: -1.0 .. 1.0 (current_player perspektifinden)
          win_probability: 0.0 .. 100.0 (yüzde)
        """
        _, value = predict(self.model, board, current_player)
        win_prob = (value + 1.0) / 2.0 * 100.0
        return float(value), float(win_prob)

    def analyze(self, board, current_player):
        """
        MCTS araması yaparak detaylı analiz döndürür.
        Dönüş: {
            'best_move': int,
            'policy': np.ndarray (7 olasılık),
            'value': float (-1..1),
            'win_probability': float (0..100)
        }
        """
        policy = self.search(board, current_player, temperature=0)
        best_col = int(np.argmax(policy))
        value, win_prob = self.evaluate_board(board, current_player)
        return {
            "best_move": best_col,
            "policy": policy,
            "value": value,
            "win_probability": win_prob,
        }


def sample_move(probs, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    return int(rng.choice(len(probs), p=probs))
