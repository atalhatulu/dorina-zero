"""train_generic.py — Oyun-Agnostik AlphaZero Eğitim Giriş Noktası (Connect4, Othello, vb.).

Historical Gating & Checkpoint Güvenlik Kuralları:
  1. `checkpoints/best_connect4.pt` dosyasına ASLA otomatik yazılmaz (manuel korunur).
  2. Eğitim sırasında bulunan en iyi model `checkpoints/<game>/current_best.pt` olarak saklanır.
  3. Her N iterasyonda bir önceki en iyi modele (`current_best.pt`) karşı değerlendirme maçı (gating) yapılır.
     Yeni model kazanma oranı eşiğini (örn. > %55) geçerse yeni `current_best.pt` ilan edilir.
  4. Başlangıçta `checkpoints/best_connect4.md5` kontrol edilir; hash değişmişse uyarı verilir.
"""
import os
import time
import hashlib
import argparse
import copy
import torch

from core.mcts import GenericMCTS
from core.replay_buffer import GenericReplayBuffer
from core.trainer import GenericTrainer, save_checkpoint, load_checkpoint
from core.self_play import parallel_self_play
from games.connect4 import Connect4Game
from games.othello import OthelloGame
from network_v2 import Connect4NetV2
from evaluate import play_match


def verify_champion_hash():
    """checkpoints/best_connect4.pt hash bütünlüğünü doğrula."""
    pt_path = "checkpoints/best_connect4.pt"
    md5_path = "checkpoints/best_connect4.md5"
    if not os.path.exists(pt_path) or not os.path.exists(md5_path):
        return

    with open(pt_path, "rb") as f:
        current_md5 = hashlib.md5(f.read()).hexdigest()

    with open(md5_path, "r", encoding="utf-8") as f:
        expected_line = f.read().strip()
        expected_md5 = expected_line.split()[0] if expected_line else ""

    if current_md5 == expected_md5:
        print(f"[Güvenlik Doğrulandı] best_connect4.pt MD5 eşleşiyor: {current_md5[:10]}...")
    else:
        print(f"[UYARI] best_connect4.pt HASH DEĞİŞMİŞ! Beklenen: {expected_md5}, Mevcut: {current_md5}")


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
    parser.add_argument("--gating-every", type=int, default=5, help="Kaç iterasyonda bir gating maçı yapılacağı")
    parser.add_argument("--gating-games", type=int, default=20, help="Gating maç sayısı")
    parser.add_argument("--gating-threshold", type=float, default=0.55, help="Gating terfi kazanma oranı eşiği")
    args = parser.parse_args()

    # Şampiyon hash kontrolü
    verify_champion_hash()

    game = get_game(args.game)
    print(f"=== {game.name.upper()} AlphaZero Eğitimi Başlatılıyor ===")
    print(f"Aksiyon boyutu: {game.action_space_size}, Girdi boyutu: {game.board_shape}")

    model = Connect4NetV2()
    start_iter = 0

    if args.resume and os.path.exists(args.resume):
        start_iter = load_checkpoint(model, args.resume)
        print(f"Checkpoint yüklendi: iter {start_iter:03d}")

    ckpt_dir = os.path.join("checkpoints", args.game)
    os.makedirs(ckpt_dir, exist_ok=True)

    current_best_path = os.path.join(ckpt_dir, "current_best.pt")
    # Eğer current_best yoksa mevcut modelle başlat
    if not os.path.exists(current_best_path):
        save_checkpoint(model, current_best_path, start_iter)
        print(f"Başlangıç referans modeli oluşturuldu: {current_best_path}")

    current_best_model = Connect4NetV2()
    load_checkpoint(current_best_model, current_best_path)

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

        # Her iterasyonda arşiv checkpoint'i kaydet (best_connect4.pt'ye dokunulmaz!)
        ckpt_path = os.path.join(ckpt_dir, f"iter_{iteration:03d}.pt")
        save_checkpoint(model, ckpt_path, iteration)
        print(f"  Kaydedildi: {ckpt_path} | Toplam tur: {time.time() - t0:.1f}s")

        # Historical Gating: Belirli aralıklarla current_best ile maç yap
        if args.gating_every > 0 and (iteration % args.gating_every == 0):
            print(f"\n  [Historical Gating] Iter {iteration:03d} vs current_best ({args.gating_games} maç, 100 sim)...")
            w, l, d = play_match(model, current_best_model, num_games=args.gating_games, num_simulations=100)
            total = w + l + d
            win_rate = (w / total) if total else 0.0
            print(f"  Gating Sonucu: W={w} L={l} D={d} (Yeni Model Kazanma: %{win_rate*100:.1f})")

            if win_rate >= args.gating_threshold:
                save_checkpoint(model, current_best_path, iteration)
                load_checkpoint(current_best_model, current_best_path)
                print(f"  TERFİ: Yeni model current_best.pt olarak güncellendi! (Kazanma: %{win_rate*100:.1f} >= %{args.gating_threshold*100:.0f})")
            else:
                print(f"  REDDEDİLDİ: Yeni model eşiği geçemedi. current_best.pt korundu.")


if __name__ == "__main__":
    main()
