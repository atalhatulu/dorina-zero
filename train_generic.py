"""train_generic.py — Oyun-Agnostik AlphaZero Eğitim Giriş Noktası (Connect4, Othello, vb.).

Historical Gating, Simetri Augmentation, Historical Rakip ve Curriculum Destekli:
  1. `checkpoints/best_connect4.pt` dosyasına ASLA otomatik yazılmaz (manuel korunur).
  2. Gating: 100 maç, 150 simülasyon, %55 eşik ve (W + 0.5*D)/N puanlama.
  3. Curriculum: Minimax-6 (%10), tüm win/draw/loss oyunları dahil.
  4. Historical: Geçmiş şampiyonlara karşı self-play (%20).
  5. Simetri: Her oyun yatay flip ile 2x veriye dönüştürülür.
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
from core.self_play import parallel_self_play, evaluate_gating_parallel
from games.connect4 import Connect4Game
from games.othello import OthelloGame
from network_v2 import Connect4NetV2
from network_othello import OthelloNet


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


def get_game_and_model_factories(name):
    name = name.lower()
    if name == "connect4":
        return Connect4Game, Connect4NetV2
    elif name == "othello":
        return OthelloGame, OthelloNet
    else:
        raise ValueError(f"Bilinmeyen oyun: {name}")


def main():
    parser = argparse.ArgumentParser(description="dorina-zero: Generic AlphaZero Trainer")
    parser.add_argument("--game", type=str, default="connect4", choices=["connect4", "othello"])
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--games-per-iter", type=int, default=300)
    parser.add_argument("--sims", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--buffer-size", type=int, default=150_000)
    parser.add_argument("--max-draw-ratio", type=float, default=0.15)
    parser.add_argument("--workers", type=int, default=11)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=5, help="Kaç iterasyonda bir checkpoint kaydedileceği")
    parser.add_argument("--gating-every", type=int, default=1, help="Kaç iterasyonda bir gating maçı yapılacağı (0: kapalı)")
    parser.add_argument("--gating-games", type=int, default=100, help="Gating maç sayısı (varsayılan: 100)")
    parser.add_argument("--gating-sims", type=int, default=150, help="Gating MCTS simülasyon adedi (varsayılan: 150)")
    parser.add_argument("--gating-threshold", type=float, default=0.55, help="Gating terfi puan eşiği (varsayılan: 0.55)")
    parser.add_argument("--curriculum", type=str, default=None, help="Curriculum bot türü (örn: minimax-6)")
    parser.add_argument("--curriculum-ratio", type=float, default=0.0, help="Curriculum oyun oranı (örn: 0.10)")
    parser.add_argument("--curriculum-depth", type=int, default=6, help="Curriculum minimax arama derinliği")
    parser.add_argument("--historical-ratio", type=float, default=0.0, help="Geçmiş checkpoint'lere karşı self-play oranı (örn: 0.20)")
    parser.add_argument("--dirichlet-alpha", type=float, default=0.5, help="MCTS Dirichlet gürültüsü alfa katsayısı (Connect4 için 0.4-0.5)")
    parser.add_argument("--out-dir", type=str, default=None, help="Checkpoint kayıt klasörü (varsayılan: checkpoints/<game>)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Ağ eğitimi için donanım (cuda/cpu)")
    args = parser.parse_args()

    verify_champion_hash()

    game_factory, model_factory = get_game_and_model_factories(args.game)
    game = game_factory()
    print(f"=== {game.name.upper()} AlphaZero Eğitimi Başlatılıyor ===")
    print(f"Aksiyon boyutu: {game.action_space_size}, Girdi boyutu: {game.board_shape}")
    print(f"Gating ayarları: Her {args.gating_every} iterasyonda, {args.gating_games} maç, {args.gating_sims} sim, eşik: %{args.gating_threshold*100:.0f} (Puanlama: (W + 0.5*D)/N)")
    if args.curriculum:
        print(f"Curriculum ayarları: Bot={args.curriculum}, Oran=%{args.curriculum_ratio*100:.0f}, Derinlik={args.curriculum_depth}")
    if args.historical_ratio > 0:
        print(f"Historical rakip ayarı: Oran=%{args.historical_ratio*100:.0f}")

    model = model_factory()
    start_iter = 0

    if args.resume and os.path.exists(args.resume):
        start_iter = load_checkpoint(model, args.resume)
        print(f"Fine-tune başlangıç checkpoint'i yüklendi: {args.resume} (iter: {start_iter:03d})")

    if args.out_dir:
        ckpt_dir = args.out_dir
        # Eğer özel out-dir verilmişse logları onun yanına veya içine yerleştir
        log_dir_name = os.path.join(os.path.dirname(ckpt_dir.rstrip("/")), "logs") if "checkpoints" in ckpt_dir else os.path.join(ckpt_dir, "logs")
    else:
        # Standart Birleşik Yapı: models/<game>/
        base_model_dir = os.path.join("models", args.game)
        ckpt_dir = os.path.join(base_model_dir, "checkpoints")
        log_dir_name = os.path.join(base_model_dir, "logs")

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir_name, exist_ok=True)

    current_best_path = os.path.join(ckpt_dir, "current_best.pt")
    if args.resume and os.path.exists(args.resume):
        save_checkpoint(model, current_best_path, start_iter)
        print(f"Referans current_best.pt resume modeline ayarlandı: {args.resume}")
    elif not os.path.exists(current_best_path):
        save_checkpoint(model, current_best_path, start_iter)
        print(f"Başlangıç referans modeli oluşturuldu: {current_best_path}")

    current_best_model = model_factory()
    load_checkpoint(current_best_model, current_best_path)

    # Historical rakip havuzu
    historical_pool = [
        os.path.join(ckpt_dir, "current_best.pt"),
        os.path.join(ckpt_dir, "iter_005.pt"),
        os.path.join(ckpt_dir, "iter_010.pt"),
    ]

    buffer = GenericReplayBuffer(capacity=args.buffer_size)
    trainer = GenericTrainer(model, lr=args.lr, weight_decay=1e-4, device=args.device)
    print(f"Eğitim Cihazı (Trainer Device): {args.device.upper()}")

    for iteration in range(start_iter + 1, start_iter + args.iterations + 1):
        t0 = time.time()
        print(f"\n[Iter {iteration:03d}] Self-play ({args.games_per_iter} oyun, {args.sims} sim)...")

        status_file = os.path.join(log_dir_name, "live_status.json")
        try:
            os.makedirs(os.path.dirname(status_file), exist_ok=True)
            with open(status_file, "w", encoding="utf-8") as sf:
                import json
                json.dump({
                    "stage": "self_play",
                    "iteration": iteration,
                    "target_iterations": start_iter + args.iterations,
                    "completed_games": 0,
                    "total_games": args.games_per_iter,
                    "progress_pct": 0.0,
                    "updated_at": time.time(),
                }, sf)
        except Exception:
            pass

        data, stats = parallel_self_play(
            model_factory=model_factory,
            model=model,
            game_factory=game_factory,
            num_games=args.games_per_iter,
            num_simulations=args.sims,
            temp_threshold=12,
            dirichlet_epsilon=0.25,
            dirichlet_alpha=args.dirichlet_alpha,
            curriculum_mode=args.curriculum,
            curriculum_ratio=args.curriculum_ratio,
            curriculum_depth=args.curriculum_depth,
            historical_pool=historical_pool,
            historical_ratio=args.historical_ratio,
            num_workers=args.workers,
            iteration=iteration,
            status_file=status_file,
        )

        sp_dt = time.time() - t0
        draw_pct = (stats['draw'] / max(args.games_per_iter, 1)) * 100.0
        details = []
        if stats.get("curriculum_total", 0) > 0:
            details.append(f"Curriculum(M={stats['curriculum_model_wins']} MM={stats['curriculum_minimax_wins']} D={stats['curriculum_draws']})")
        if stats.get("historical_total", 0) > 0:
            details.append(f"Historical(P1={stats['historical_p1_wins']} P2={stats['historical_p2_wins']} D={stats['historical_draws']})")
        extra_str = f" | {', '.join(details)}" if details else ""

        aug_label = "8x D4 Simetri" if "othello" in args.game.lower() else "2x Flip"
        print(f"  Self-play bitti: {sp_dt:.1f}s | p1={stats['player1']} p2={stats['player2']} draw={stats['draw']} (Draw: %{draw_pct:.1f}){extra_str} | Üretilen Örnek ({aug_label}): {len(data)}")

        buffer.add(data)

        # Eğitim aşaması canlı durumu
        try:
            with open(status_file, "w", encoding="utf-8") as sf:
                import json
                json.dump({
                    "stage": "training_nn",
                    "iteration": iteration,
                    "target_iterations": start_iter + args.iterations,
                    "completed_games": args.games_per_iter,
                    "total_games": args.games_per_iter,
                    "progress_pct": 100.0,
                    "buffer_size": len(buffer),
                    "updated_at": time.time(),
                }, sf)
        except Exception:
            pass

        train_data = buffer.sample(min(len(buffer), 20_000), max_draw_ratio=args.max_draw_ratio)
        pl, vl = trainer.train(train_data, epochs=args.epochs, batch_size=args.batch_size)
        print(f"  Eğitim bitti: policy_loss={pl:.4f}, value_loss={vl:.4f}")

        # Canlı Durum ve CSV Kaydı
        try:
            os.makedirs(log_dir_name, exist_ok=True)
            csv_path = os.path.join(log_dir_name, f"iterations_{args.game}.csv")
            csv_exists = os.path.exists(csv_path)
            with open(csv_path, "a", encoding="utf-8") as f_csv:
                if not csv_exists:
                    f_csv.write("iteration,policy_loss,value_loss,p1_wins,p2_wins,draws,total_samples,duration\n")
                f_csv.write(f"{iteration},{pl:.4f},{vl:.4f},{stats['player1']},{stats['player2']},{stats['draw']},{len(data)},{sp_dt:.1f}\n")
        except Exception:
            pass

        # Arşiv Checkpoint
        if args.checkpoint_every > 0 and (iteration % args.checkpoint_every == 0):
            ckpt_path = os.path.join(ckpt_dir, f"iter_{iteration:03d}.pt")
            save_checkpoint(model, ckpt_path, iteration)
            if ckpt_path not in historical_pool:
                historical_pool.append(ckpt_path)
            print(f"  Kaydedildi: {ckpt_path}")

        # Historical Gating: current_best ile cesitlendirilmis acilislarla paralel mac yap
        if args.gating_every > 0 and (iteration % args.gating_every == 0):
            print(f"\n  [Historical Gating] Iter {iteration:03d} vs current_best ({args.gating_games} maç, {args.gating_sims} sim, rastgele açılışlı)...")
            try:
                with open(status_file, "w", encoding="utf-8") as sf:
                    import json
                    json.dump({
                        "stage": "gating",
                        "iteration": iteration,
                        "target_iterations": start_iter + args.iterations,
                        "gating_games": args.gating_games,
                        "updated_at": time.time(),
                    }, sf)
            except Exception:
                pass
            w, l, d = evaluate_gating_parallel(
                model_factory=model_factory,
                model_a=model,
                model_b=current_best_model,
                game_factory=game_factory,
                num_games=args.gating_games,
                num_simulations=args.gating_sims,
                num_workers=args.workers,
                seed=42 + iteration,
            )
            total = w + l + d
            # Puanlama formülü: (W + 0.5 * D) / N
            score_rate = (w + 0.5 * d) / max(total, 1)
            print(f"  Gating Sonucu: YeniModel={w} current_best={l} Beraberlik={d} (Skor Oranı: %{score_rate*100:.1f})")

            if score_rate >= args.gating_threshold:
                save_checkpoint(model, current_best_path, iteration)
                load_checkpoint(current_best_model, current_best_path)
                print(f"  TERFİ: Yeni model current_best.pt olarak güncellendi! (%{score_rate*100:.1f} >= %{args.gating_threshold*100:.0f})")
            else:
                load_checkpoint(model, current_best_path)
                trainer.reset_optimizer()
                print(f"  REDDEDİLDİ: Yeni model eşiği geçemedi. Model current_best.pt ağırlıklarına ve optimizer sıfırlandı.")

        print(f"  Toplam tur süresi: {time.time() - t0:.1f}s")
        try:
            with open(status_file, "w", encoding="utf-8") as sf:
                import json
                json.dump({
                    "stage": "completed_iteration",
                    "iteration": iteration,
                    "target_iterations": start_iter + args.iterations,
                    "last_policy_loss": pl,
                    "last_value_loss": vl,
                    "updated_at": time.time(),
                }, sf)
        except Exception:
            pass


if __name__ == "__main__":
    main()
