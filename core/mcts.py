"""core/mcts.py — Oyun-Agnostik Monte Carlo Tree Search (Transposition Table & BaseGame Destekli)."""
import math
import collections
import numpy as np
import torch

from games.base_game import BaseGame


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


class TranspositionTable:
    """Sabit boyutlu LRU Cache (OrderedDict)."""
    def __init__(self, maxsize=30_000):
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


class GenericMCTS:
    """BaseGame arayüzü ile çalışan genel MCTS motoru."""
    def __init__(self, model, game: BaseGame, num_simulations=150, c_puct=1.5, tt_size=30_000):
        self.model = model
        self.game = game
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.tt = TranspositionTable(maxsize=tt_size)

    def _expand(self, node, state, current_player):
        terminal = self.game.check_terminal_value(state, current_player)
        if terminal is not None:
            node.is_expanded = True
            return terminal

        key = self.game.state_to_key(state, current_player)
        cached = self.tt.get(key)
        if cached is not None:
            policy, v = cached
        else:
            with torch.no_grad():
                encoded = self.game.encode_state(state, current_player)
                x = torch.from_numpy(encoded).unsqueeze(0)
                policy_logits, value = self.model(x)
                policy = torch.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()
                v = value.item()
            self.tt.put(key, (policy, v))

        valid_mask = self.game.get_valid_moves(state, current_player)
        masked_policy = np.zeros(self.game.action_space_size, dtype=np.float32)
        valid_indices = np.where(valid_mask)[0]

        for idx in valid_indices:
            masked_policy[idx] = policy[idx]

        s = masked_policy.sum()
        if s > 0:
            masked_policy /= s
        else:
            for idx in valid_indices:
                masked_policy[idx] = 1.0 / len(valid_indices)

        for idx in valid_indices:
            node.children[idx] = MCTSNode(prior=masked_policy[idx], player_just_moved=current_player)
        node.is_expanded = True
        return v

    def _select(self, node):
        best_score = -float("inf")
        best_action = None
        for action, child in node.children.items():
            q = -child.q_value() if child.visits > 0 else 0.0
            u = self.c_puct * child.prior * (math.sqrt(node.visits) / (1 + child.visits))
            score = q + u
            if score > best_score:
                best_score = score
                best_action = action
        return best_action

    def _simulate(self, root, root_state, root_player):
        path = [root]
        current_state = self.game.clone_state(root_state)
        current_player = root_player

        node = root
        while not node.is_leaf():
            action = self._select(node)
            if action is None:
                break
            current_state = self.game.get_next_state(current_state, action, current_player)
            current_player = 2 if current_player == 1 else 1
            node = node.children[action]
            path.append(node)

        value = self._expand(node, current_state, current_player)

        for n in reversed(path):
            node_player = n.player_just_moved
            v_for_node = value if node_player == current_player else -value
            n.visits += 1
            n.value_sum += v_for_node

    def search(self, state, current_player, temperature=1.0,
               dirichlet_alpha=1.0, dirichlet_epsilon=0.15):
        terminal = self.game.check_terminal_value(state, current_player)
        if terminal is not None:
            probs = np.zeros(self.game.action_space_size, dtype=np.float32)
            valid_mask = self.game.get_valid_moves(state, current_player)
            valid_indices = np.where(valid_mask)[0]
            if len(valid_indices) > 0:
                probs[valid_indices] = 1.0 / len(valid_indices)
            return probs

        opp = 2 if current_player == 1 else 1
        root = MCTSNode(prior=0.0, player_just_moved=opp)
        self._expand(root, state, current_player)
        root.visits = 1

        if dirichlet_epsilon > 0 and len(root.children) > 0:
            noise = np.random.dirichlet([dirichlet_alpha] * len(root.children))
            for i, child in enumerate(root.children.values()):
                child.prior = ((1 - dirichlet_epsilon) * child.prior + dirichlet_epsilon * noise[i])

        for _ in range(self.num_simulations):
            self._simulate(root, state, current_player)

        visits = np.zeros(self.game.action_space_size, dtype=np.float32)
        for action, child in root.children.items():
            visits[action] = child.visits

        if visits.sum() == 0:
            valid_mask = self.game.get_valid_moves(state, current_player)
            valid_indices = np.where(valid_mask)[0]
            for idx in valid_indices:
                visits[idx] = 1.0

        if temperature == 0:
            probs = np.zeros(self.game.action_space_size, dtype=np.float32)
            probs[np.argmax(visits)] = 1.0
            return probs

        visits_t = visits ** (1.0 / temperature)
        probs = visits_t / visits_t.sum()
        return probs

    def best_move(self, state, current_player):
        probs = self.search(state, current_player, temperature=0)
        return int(np.argmax(probs))

    def evaluate_state(self, state, current_player):
        """Hızlı NN değerlendirmesi (MCTS simülasyonu yapmadan)."""
        self.model.eval()
        with torch.no_grad():
            x = torch.from_numpy(self.game.encode_state(state, current_player)).unsqueeze(0)
            _, value = self.model(x)
            val = float(value.item())
            win_prob = float((val + 1.0) / 2.0 * 100.0)
            return val, win_prob

    def analyze(self, state, current_player):
        policy = self.search(state, current_player, temperature=0)
        best_col = int(np.argmax(policy))
        val, win_prob = self.evaluate_state(state, current_player)
        return {
            "best_move": best_col,
            "policy": policy,
            "value": val,
            "win_probability": win_prob,
        }


def sample_move(probs, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    return int(rng.choice(len(probs), p=probs))
