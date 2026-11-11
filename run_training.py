"""Tam eğitim döngüsü: self-play → train → eval → checkpoint."""
import os
import time
import argparse
import numpy as np
import torch

from collections import deque
from network import Connect4Net
from replay_buffer import ReplayBuffer
from train import train, save_checkpoint, load_checkpoint
import logger
from run_self_play import parallel_self_play
from coverage_tracker import get_tracker, track_position


class TrainingProgress:
    """Eğitim ilerleme yüzdesi ve ETA takipçisi."""
    def __init__(self, target_iter, window_size=10):
        self.target_iter = target_iter
        self.durations = deque(maxlen=window_size)

    def update(self, current_iter, duration_sec):
        self.durations.append(duration_sec)
        avg_sec = sum(self.durations) / len(self.durations)
        remaining_iters = max(0, self.target_iter - current_iter)
        eta_sec = remaining_iters * avg_sec
        eta_min = eta_sec / 60.0
        pct = (current_iter / max(self.target_iter, 1)) * 100.0

        print(f"[Iter {current_iter}/{self.target_iter}] Tamamlandı. Süre: {duration_sec:.0f}s (ort: {avg_sec:.0f}s)")
        print(f"  İlerleme: %{pct:.1f} | Kalan: ~{remaining_iters} iterasyon (~{eta_min:.1f} dk)\n")


def evaluate_vs_random(model, num_games=20, num_simulations=50):
    """Model vs rastgele oyuncu. Kazanma oranını döndür."""
    from board import (
        create_board, drop_piece, get_next_open_row,
        get_valid_locations, winning_move, is_full,
    )
    from mcts import MCTS
    import random

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--games-per-iter", type=int, default=100)
    parser.add_argument("--sims", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--total-iterations", type=int, default=None, help="Toplam hedef iterasyon numarası")
    args = parser.parse_args()

    # Model
    model = Connect4Net()
    start_iter = 0
    if args.resume and os.path.exists(args.resume):
        start_iter = load_checkpoint(model, args.resume)
        print(f"Checkpoint yüklendi: iter {start_iter}")

    os.makedirs("checkpoints", exist_ok=True)
    buffer = ReplayBuffer(capacity=args.buffer_size)

    target_iter = args.total_iterations if args.total_iterations else (start_iter + args.iterations)
    progress_tracker = TrainingProgress(target_iter=target_iter, window_size=10)

    print(f"Toplam iterasyon hedefi: {target_iter} (Başlangıç: {start_iter})")
    print(f"Her iterasyon: {args.games_per_iter} oyun, {args.sims} sim, {args.epochs} epoch")
    print(f"Buffer boyutu: {args.buffer_size}")
    print()

    for iteration in range(start_iter + 1, start_iter + args.iterations + 1):
        iter_start = time.time()

        # 1. Self-play (paralel)
        print(f"[Iter {iteration}] Self-play...")
        sp_start = time.time()
        data, stats = parallel_self_play(
            model, num_games=args.games_per_iter,
            num_simulations=args.sims,
            iteration=iteration, num_workers=args.workers,
        )
        sp_dt = time.time() - sp_start
        print(f"  {args.games_per_iter} oyun: {sp_dt:.1f}s ({sp_dt/args.games_per_iter:.2f}s/oyun)")
        print(f"  p1={stats['player1']} p2={stats['player2']} draw={stats['draw']}")
        print(f"  Örnek: {len(data)}")

        # --- Coverage tracking (ana process'te, race condition yok) ---
        new_positions = 0
        for board, player, policy, v in data:
            if track_position(board, player):
                new_positions += 1
        print(f"  Yeni pozisyon: {new_positions}")

        # Buffer'a ekle
        buffer.add(data)

        # 2. Eğitim
        print(f"[Iter {iteration}] Eğitim...")
        tr_start = time.time()
        train_data = buffer.sample(min(len(buffer), 20_000))
        pl, vl = train(model, train_data, epochs=args.epochs,
                       batch_size=args.batch_size, lr=args.lr)
        tr_dt = time.time() - tr_start
        print(f"  Loss: policy={pl:.4f} value={vl:.4f}, {tr_dt:.1f}s")

        # 3. Değerlendirme
        print(f"[Iter {iteration}] Değerlendirme...")
        ev_start = time.time()
        wins, losses, draws = evaluate_vs_random(model, num_games=10, num_simulations=30)
        ev_dt = time.time() - ev_start
        win_rate = wins / max(wins + losses + draws, 1)
        print(f"  Rastgele rakibe karşı: {wins}W/{losses}L/{draws}D ({win_rate:.1%})")

        # 4. Log
        logger.log_iteration(
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
        ckpt_path = f"checkpoints/iter_{iteration:03d}.pt"
        save_checkpoint(model, ckpt_path, iteration)
        print(f"  Kaydedildi: {ckpt_path}")

        total_dt = time.time() - iter_start
        progress_tracker.update(current_iter=iteration, duration_sec=total_dt)

    print("Eğitim tamamlandı!")


if __name__ == "__main__":
    main()
