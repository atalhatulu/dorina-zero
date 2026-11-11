"""Paralel self-play. multiprocessing.Pool ile N worker."""
import os
import time
import pickle
import argparse
import numpy as np
import torch

from board import PLAYER, AI
from network import Connect4Net
from self_play import play_one_game, history_to_training_data

# Worker başına global model (her worker bir kez yükler)
_worker_model = None
_worker_args = None


def _init_worker(model_state, num_simulations, iteration):
    """Her worker başlarken çağrılır."""
    global _worker_model, _worker_args
    torch.set_num_threads(1)  # worker başına 1 thread — over-subscription önle
    _worker_model = Connect4Net()
    _worker_model.load_state_dict(model_state)
    _worker_model.eval()
    _worker_args = {"num_simulations": num_simulations, "iteration": iteration}


def _play_one(args):
    """Bir oyun oyna, eğitim verisi döndür."""
    game_id = args
    history, winner, moves = play_one_game(
        _worker_model,
        num_simulations=_worker_args["num_simulations"],
        game_id=game_id,
        iteration=_worker_args["iteration"],
        log=False,  # log'u ana process yapacak
    )
    data = history_to_training_data(history, winner)
    return {"game_id": game_id, "winner": winner, "moves": moves, "data": data}


def parallel_self_play(model, num_games, num_simulations=200,
                       iteration=0, num_workers=None, verbose=True):
    """
    N oyunu paralel oynat. (all_data, stats) döndür.
    """
    import multiprocessing as mp

    if num_workers is None:
        num_workers = max(1, (os.cpu_count() or 4) - 1)

    model_state = model.state_dict()
    game_ids = list(range(num_games))

    if verbose:
        print(f"  {num_workers} worker ile {num_games} oyun...")

    with mp.Pool(
        processes=num_workers,
        initializer=_init_worker,
        initargs=(model_state, num_simulations, iteration),
    ) as pool:
        results = pool.map(_play_one, game_ids, chunksize=4)

    # Topla
    all_data = []
    stats = {"player1": 0, "player2": 0, "draw": 0,
             "total_moves": 0, "total_duration": 0.0}
    for r in results:
        all_data.extend(r["data"])
        stats[r["winner"]] += 1
        stats["total_moves"] += r["moves"]

    stats["avg_moves"] = stats["total_moves"] / max(num_games, 1)
    return all_data, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--sims", type=int, default=100)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    model = Connect4Net()
    t0 = time.time()
    data, stats = parallel_self_play(
        model, num_games=args.games, num_simulations=args.sims,
        iteration=0, num_workers=args.workers,
    )
    dt = time.time() - t0
    print(f"\n{args.games} oyun: {dt:.1f}s ({dt/args.games:.2f}s/oyun)")
    print(f"Stats: {stats}")
    print(f"Eğitim örneği: {len(data)}")


if __name__ == "__main__":
    main()