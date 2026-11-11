"""Tam eğitim döngüsü (Aşama 2: Connect4NetV2 116K, draw undersampling, temp_threshold=4)."""
import os
import time
import argparse
import random
from collections import deque
import numpy as np
import torch

from network_v2 import Connect4NetV2
from replay_buffer import ReplayBuffer
from train import train, save_checkpoint, load_checkpoint
import logger
from coverage_tracker import get_tracker, track_position
from self_play import play_one_game, history_to_training_data


LOGS_DIR_V2 = "logs_v2"
ITER_PATH_V2 = os.path.join(LOGS_DIR_V2, "iterations_v2.csv")
ITER_FIELDS_V2 = [
    "iteration", "games", "p1_wins", "p2_wins", "draws",
    "avg_moves", "avg_duration_sec",
    "policy_loss", "value_loss",
    "elo_vs_prev", "timestamp",
]


def log_iteration_v2(iteration, games, p1_wins, p2_wins, draws,
                     avg_moves, avg_duration_sec,
                     policy_loss=None, value_loss=None):
    os.makedirs(LOGS_DIR_V2, exist_ok=True)
    import csv
    from datetime import datetime
    file_exists = os.path.exists(ITER_PATH_V2)
    with open(ITER_PATH_V2, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ITER_FIELDS_V2)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "iteration": iteration,
            "games": games,
            "p1_wins": p1_wins,
            "p2_wins": p2_wins,
            "draws": draws,
            "avg_moves": round(avg_moves, 2),
            "avg_duration_sec": round(avg_duration_sec, 3),
            "policy_loss": round(policy_loss, 4) if policy_loss is not None else "",
            "value_loss": round(value_loss, 4) if value_loss is not None else "",
            "elo_vs_prev": "",
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        })
_worker_model = None
_worker_args = None


def _init_worker_v2(model_state, num_simulations, iteration, temp_threshold, dirichlet_epsilon):
    global _worker_model, _worker_args
    torch.set_num_threads(1)
    _worker_model = Connect4NetV2()
    _worker_model.load_state_dict(model_state)
    _worker_model.eval()
    _worker_args = {
        "num_simulations": num_simulations,
        "iteration": iteration,
        "temp_threshold": temp_threshold,
        "dirichlet_epsilon": dirichlet_epsilon,
    }


def _play_one_v2(args):
    game_id = args
    history, winner, moves = play_one_game(
        _worker_model,
        num_simulations=_worker_args["num_simulations"],
        temp_threshold=_worker_args["temp_threshold"],
        game_id=game_id,
        iteration=_worker_args["iteration"],
        log=False,
    )
    data = history_to_training_data(history, winner)
    return {"game_id": game_id, "winner": winner, "moves": moves, "data": data}


def parallel_self_play_v2(model, num_games, num_simulations=150,
                          iteration=0, temp_threshold=4, dirichlet_epsilon=0.15,
                          num_workers=None, verbose=True):
    import multiprocessing as mp

    if num_workers is None:
        num_workers = max(1, (os.cpu_count() or 4) - 1)

    model_state = model.state_dict()
    game_ids = list(range(num_games))

    if verbose:
        print(f"  {num_workers} worker ile {num_games} oyun (sims={num_simulations}, temp_thresh={temp_threshold})...")

    with mp.Pool(
        processes=num_workers,
        initializer=_init_worker_v2,
        initargs=(model_state, num_simulations, iteration, temp_threshold, dirichlet_epsilon),
    ) as pool:
        results = pool.map(_play_one_v2, game_ids, chunksize=4)

    all_data = []
    stats = {"player1": 0, "player2": 0, "draw": 0,
             "total_moves": 0, "total_duration": 0.0}
    for r in results:
        all_data.extend(r["data"])
        stats[r["winner"]] += 1
        stats["total_moves"] += r["moves"]

    stats["avg_moves"] = stats["total_moves"] / max(num_games, 1)
    return all_data, stats


def evaluate_vs_random_v2(model, num_games=20, num_simulations=50):
    from board import create_board, drop_piece, get_next_open_row, get_valid_locations, winning_move, is_full
    from mcts import MCTS

    model.eval()
    mcts = MCTS(model, num_simulations=num_simulations)
    wins, losses, draws = 0, 0, 0

    for i in range(num_games):
        board = create_board()
        current_player = 1
        model_player = 1 if i % 2 == 0 else 2

        while True:
            valid = get_valid_locations(board)
            if not valid:
                draws += 1
                break

            if current_player == model_player:
                policy = mcts.search(board, current_player, temperature=0)
                move = int(np.argmax(policy))
            else:
                move = random.choice(valid)

            row = get_next_open_row(board, move)
            drop_piece(board, row, move, current_player)

            if winning_move(board, current_player):
                if current_player == model_player:
                    wins += 1
                else:
                    losses += 1
                break
            if is_full(board):
                draws += 1
                break
            current_player = 2 if current_player == 1 else 1

    return wins, losses, draws


def main():
    parser = argparse.ArgumentParser(description="dorina4 - Aşama 2 (116K Model Eğitimi)")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--games-per-iter", type=int, default=300)
    parser.add_argument("--sims", type=int, default=150)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--workers", type=int, default=11)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--temp-threshold", type=int, default=4)
    parser.add_argument("--dirichlet-epsilon", type=float, default=0.15)
    args = parser.parse_args()

    # 116K Model
    model = Connect4NetV2()
    start_iter = 0
    if args.resume and os.path.exists(args.resume):
        start_iter = load_checkpoint(model, args.resume)
        print(f"Checkpoint yüklendi: v2_iter_{start_iter:03d}")

    os.makedirs("checkpoints_v2", exist_ok=True)
    buffer = ReplayBuffer(capacity=args.buffer_size)

    print(f"--- Aşama 2 Başlatılıyor (Connect4NetV2 116K) ---")
    print(f"Toplam iterasyon hedefi: {args.iterations}")
    print(f"Her iterasyon: {args.games_per_iter} oyun, {args.sims} sim, {args.epochs} epoch")
    print(f"Buffer boyutu: {args.buffer_size} (Max draw oranı: %20)")
    print(f"Keşif ayarları: temp_threshold={args.temp_threshold}, epsilon={args.dirichlet_epsilon}")
    print(f"Öğrenme hızı: {args.lr}\n")

    # Optimizer'ı kalıcı tut (Momentumu koru)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    for iteration in range(start_iter + 1, start_iter + args.iterations + 1):
        iter_start = time.time()

        # 1. Self-play
        print(f"[v2_Iter {iteration:03d}] Self-play...")
        sp_start = time.time()
        data, stats = parallel_self_play_v2(
            model, num_games=args.games_per_iter,
            num_simulations=args.sims,
            iteration=iteration,
            temp_threshold=args.temp_threshold,
            dirichlet_epsilon=args.dirichlet_epsilon,
            num_workers=args.workers,
        )
        sp_dt = time.time() - sp_start
        draw_pct = (stats['draw'] / max(args.games_per_iter, 1)) * 100.0
        print(f"  {args.games_per_iter} oyun: {sp_dt:.1f}s ({sp_dt/args.games_per_iter:.2f}s/oyun)")
        print(f"  p1={stats['player1']} p2={stats['player2']} draw={stats['draw']} (Draw: %{draw_pct:.1f})")
        print(f"  Örnek: {len(data)}")

        # Coverage takibi
        new_positions = 0
        for board, player, policy, v in data:
            if track_position(board, player):
                new_positions += 1
        print(f"  Yeni pozisyon: {new_positions}")

        # Buffer'a ekle
        buffer.add(data)

        # 2. Eğitim (Draw undersampling ile: en fazla %20 draw)
        print(f"[v2_Iter {iteration:03d}] Eğitim...")
        tr_start = time.time()
        train_data = buffer.sample(min(len(buffer), 20_000), max_draw_ratio=0.20)
        pl, vl = train(model, train_data, epochs=args.epochs,
                       batch_size=args.batch_size, lr=args.lr, optimizer=optimizer)
        tr_dt = time.time() - tr_start
        print(f"  Loss: policy={pl:.4f} value={vl:.4f}, {tr_dt:.1f}s")

        # 3. Değerlendirme
        print(f"[v2_Iter {iteration:03d}] Değerlendirme...")
        ev_start = time.time()
        wins, losses, draws = evaluate_vs_random_v2(model, num_games=10, num_simulations=30)
        ev_dt = time.time() - ev_start
        win_rate = wins / max(wins + losses + draws, 1)
        print(f"  Rastgele rakibe karşı: {wins}W/{losses}L/{draws}D ({win_rate:.1%})")

        # 4. Log
        log_iteration_v2(
            iteration=iteration,
            games=args.games_per_iter,
            p1_wins=stats["player1"],
            p2_wins=stats["player2"],
            draws=stats["draw"],
            avg_moves=stats["avg_moves"],
            avg_duration_sec=sp_dt / max(args.games_per_iter, 1),
            policy_loss=pl,
            value_loss=vl,
        )

        tracker = get_tracker()
        tracker.save_stats()
        stats_cov = tracker.stats()
        print(f"  Coverage: {stats_cov['coverage_str']} ({stats_cov['coverage_pct']:.6f}%)")

        # 5. Checkpoint
        ckpt_path = f"checkpoints_v2/v2_iter_{iteration:03d}.pt"
        save_checkpoint(model, ckpt_path, iteration)
        print(f"  Kaydedildi: {ckpt_path}")

        total_dt = time.time() - iter_start
        print(f"[v2_Iter {iteration:03d}] Toplam: {total_dt:.1f}s\n")

    print("Aşama 2 eğitimi tamamlandı!")


if __name__ == "__main__":
    main()
