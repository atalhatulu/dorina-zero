"""core/self_play.py — Oyun-Agnostik Self-Play ve Paralel Simülasyon Motoru."""
import os
import time
import numpy as np
import torch
import multiprocessing as mp

from core.mcts import GenericMCTS, sample_move
from games.base_game import BaseGame


def play_one_game(model, game: BaseGame, num_simulations=150, temperature=1.0,
                  temp_threshold=4, dirichlet_epsilon=0.15):
    """Tek bir oyunu baştan sona oynar, eğitim verisi üretir."""
    mcts = GenericMCTS(model, game=game, num_simulations=num_simulations)
    state = game.get_initial_state()
    current_player = 1
    history = []
    move_count = 0

    while True:
        terminal_val = game.check_terminal_value(state, current_player)
        if terminal_val is not None:
            break

        temp = temperature if move_count < temp_threshold else 0.0
        policy = mcts.search(state, current_player, temperature=temp, dirichlet_epsilon=dirichlet_epsilon)

        encoded = game.encode_state(state, current_player)
        history.append((encoded, policy.copy(), current_player))

        action = sample_move(policy)
        state = game.get_next_state(state, action, current_player)
        move_count += 1
        current_player = 2 if current_player == 1 else 1

    # Oyun bittiğinde terminal değeri belirle
    # Son hamleyi yapan oyuncunun perspektifinden bak
    last_player = 2 if current_player == 1 else 1
    term_from_last = game.check_terminal_value(state, last_player)

    if term_from_last is None or term_from_last == 0.0:
        winner = "draw"
        win_player = 0
    elif term_from_last > 0:
        win_player = last_player
        winner = f"player{win_player}"
    else:
        win_player = 1 if last_player == 2 else 2
        winner = f"player{win_player}"

    # Veriyi (state, policy, value) formatına çevir
    training_data = []
    for encoded_x, pol, player in history:
        if win_player == 0:
            v = 0.0
        elif player == win_player:
            v = 1.0
        else:
            v = -1.0
        training_data.append((encoded_x, pol, v))

    return training_data, winner, move_count


# Paralel worker altyapısı
_worker_model = None
_worker_game = None
_worker_args = None


def _init_worker(model_factory, model_state, game_factory, num_simulations, temp_threshold, dirichlet_epsilon):
    global _worker_model, _worker_game, _worker_args
    torch.set_num_threads(1)
    _worker_model = model_factory()
    _worker_model.load_state_dict(model_state)
    _worker_model.eval()
    _worker_game = game_factory()
    _worker_args = {
        "num_simulations": num_simulations,
        "temp_threshold": temp_threshold,
        "dirichlet_epsilon": dirichlet_epsilon,
    }


def _worker_task(game_id):
    data, winner, moves = play_one_game(
        _worker_model,
        _worker_game,
        num_simulations=_worker_args["num_simulations"],
        temp_threshold=_worker_args["temp_threshold"],
        dirichlet_epsilon=_worker_args["dirichlet_epsilon"],
    )
    return {"game_id": game_id, "data": data, "winner": winner, "moves": moves}


def parallel_self_play(model_factory, model, game_factory, num_games=300,
                       num_simulations=150, temp_threshold=4, dirichlet_epsilon=0.15,
                       num_workers=None, verbose=True):
    if num_workers is None:
        num_workers = max(1, (os.cpu_count() or 4) - 1)

    model_state = model.state_dict()
    game_ids = list(range(num_games))

    if verbose:
        print(f"  {num_workers} worker ile {num_games} oyun oynanıyor...")

    with mp.Pool(
        processes=num_workers,
        initializer=_init_worker,
        initargs=(model_factory, model_state, game_factory, num_simulations, temp_threshold, dirichlet_epsilon),
    ) as pool:
        results = pool.map(_worker_task, game_ids, chunksize=4)

    all_data = []
    stats = {"player1": 0, "player2": 0, "draw": 0, "total_moves": 0}
    for r in results:
        all_data.extend(r["data"])
        stats[r["winner"]] += 1
        stats["total_moves"] += r["moves"]

    stats["avg_moves"] = stats["total_moves"] / max(num_games, 1)
    return all_data, stats
