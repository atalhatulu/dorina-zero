"""train_generic.py — Oyun-Agnostik AlphaZero Eğitim Giriş Noktası (Connect4, Othello, vb.)."""
import os
import time
import argparse
import torch

from core.mcts import GenericMCTS
from core.replay_buffer import GenericReplayBuffer
from core.trainer import GenericTrainer, save_checkpoint, load_checkpoint
from core.self_play import parallel_self_play
from games.connect4 import Connect4Game
from games.othello import OthelloGame
from network_v2 import Connect4NetV2


def get_game(name):
    name = name.lower()
    if name == "connect4":
        return Connect4Game()
    elif name == "othello":
        return OthelloGame()
    else:
        raise ValueError(f"Bilinmeyen oyun: {name}")


def main():
    parser = argparse.ArgumentParser(description="dorina-zero: Generic AlphaZero Trainer")
    parser.add_argument("--game", type=str, default="connect4", choices=["connect4", "othello"])
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--games-per-iter", type=int, default=300)
    parser.add_argument("--sims", type=int, default=150)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--max-draw-ratio", type=float, default=0.15)
    parser.add_argument("--workers", type=int, default=11)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    game = get_game(args.game)
    print(f"=== {game.name.upper()} AlphaZero Eğitimi Başlatılıyor ===")
    print(f"Aksiyon boyutu: {game.action_space_size}, Girdi boyutu: {game.board_shape}")

    # Şimdilik Connect4 için Connect4NetV2; Othello için genel ağ desteği
    model = Connect4NetV2()
    start_iter = 0

    if args.resume and os.path.exists(args.resume):
        start_iter = load_checkpoint(model, args.resume)
        print(f"Checkpoint yüklendi: iter {start_iter:03d}")

    ckpt_dir = os.path.join("checkpoints", args.game)
    os.makedirs(ckpt_dir, exist_ok=True)

    buffer = GenericReplayBuffer(capacity=args.buffer_size)
    trainer = GenericTrainer(model, lr=args.lr, weight_decay=1e-4)

    for iteration in range(start_iter + 1, start_iter + args.iterations + 1):
        t0 = time.time()
        print(f"\n[Iter {iteration:03d}] Self-play ({args.games_per_iter} oyun, {args.sims} sim)...")

        data, stats = parallel_self_play(
            model_factory=Connect4NetV2,
            model=model,
            game_factory=Connect4Game,
            num_games=args.games_per_iter,
            num_simulations=args.sims,
            temp_threshold=4,
            dirichlet_epsilon=0.15,
            num_workers=args.workers,
        )

        sp_dt = time.time() - t0
        draw_pct = (stats['draw'] / max(args.games_per_iter, 1)) * 100.0
        print(f"  Self-play bitti: {sp_dt:.1f}s | p1={stats['player1']} p2={stats['player2']} draw={stats['draw']} (Draw: %{draw_pct:.1f})")

        buffer.add(data)

        # Eğitim
        train_data = buffer.sample(min(len(buffer), 20_000), max_draw_ratio=args.max_draw_ratio)
        pl, vl = trainer.train(train_data, epochs=args.epochs, batch_size=args.batch_size)
        print(f"  Eğitim bitti: policy_loss={pl:.4f}, value_loss={vl:.4f}")

        # Checkpoint
        ckpt_path = os.path.join(ckpt_dir, f"iter_{iteration:03d}.pt")
        save_checkpoint(model, ckpt_path, iteration)
        print(f"  Kaydedildi: {ckpt_path} | Toplam tur: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
