"""eval_robust.py — Güvenilir, Çeşitlendirilmiş ve İstatistiksel Değerlendirme Aracı.

Özellikler:
  1. Rastgele Açılış Kitaplığı: Her maçta ilk 2-4 hamle rastgele geçerli hamlelerle oynanır.
  2. Çift Renkli Test: Her rastgele açılış pozisyonu hem Model=P1 hem Model=P2 olarak iki kez oynanır.
  3. Wilson Skor Güven Aralığı: %95 Güven Aralığı (Wilson score interval) ile istatistiksel kesinlik.
  4. Açılış Bazında Teşhis: Modelin hangi açılış varyantlarında zorlandığı ve kaybettiği raporlanır.
"""
import math
import random
import argparse
from collections import defaultdict
import numpy as np
import torch

from network import Connect4Net
from network_v2 import Connect4NetV2
from games.connect4 import Connect4Game, PLAYER, AI
from core.mcts import GenericMCTS
from minimax_ai import get_ai_move
import board_fast as bf


def wilson_score_interval(successes: int, total: int, confidence: float = 0.95):
    """Wilson Score Güven Aralığı hesabı."""
    if total == 0:
        return 0.0, 0.0, 0.0

    p = successes / total
    # %95 güven aralığı için z ~ 1.96
    z = 1.959964
    denom = 1 + (z**2) / total
    center = (p + (z**2) / (2 * total)) / denom
    margin = (z * math.sqrt((p * (1 - p) / total) + (z**2) / (4 * total**2))) / denom

    low = max(0.0, center - margin)
    high = min(1.0, center + margin)
    return p, low, high


def load_model(ckpt_path: str, model_type: str = "auto") -> torch.nn.Module:
    """Checkpoint dosyasından doğru mimariyi yükler."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model_state", ckpt)

    if model_type == "v1":
        model = Connect4Net()
    elif model_type == "v2":
        model = Connect4NetV2()
    else:
        # Otomatik algılama
        if "policy_conv.weight" in state or "bn1.weight" in state:
            model = Connect4NetV2()
        else:
            model = Connect4Net()

    model.load_state_dict(state)
    model.eval()
    return model


def generate_random_openings(game: Connect4Game, num_openings: int, min_ply: int = 2, max_ply: int = 4, seed: int = 42):
    """Her biri oyun bitmemiş ve geçerli rastgele açılış hamle dizileri üretir."""
    random.seed(seed)
    openings = []
    seen = set()

    attempts = 0
    while len(openings) < num_openings and attempts < num_openings * 50:
        attempts += 1
        ply = random.randint(min_ply, max_ply)
        state = game.get_initial_state()
        cur_p = PLAYER
        moves = []
        terminal = False

        for _ in range(ply):
            valid_moves = [c for c in range(game.action_space_size) if bf.is_valid_location(state, c)]
            if not valid_moves:
                terminal = True
                break
            action = random.choice(valid_moves)
            state = game.get_next_state(state, action, cur_p)
            moves.append(action)
            if game.check_terminal_value(state, cur_p) is not None:
                terminal = True
                break
            cur_p = AI if cur_p == PLAYER else PLAYER

        moves_tuple = tuple(moves)
        if not terminal and moves_tuple not in seen:
            seen.add(moves_tuple)
            openings.append(moves)

    return openings


def play_from_opening(game: Connect4Game, model, opening_moves: list, model_player: int,
                      minimax_depth: int = 4, num_simulations: int = 300):
    """Belirli bir açılış pozisyonundan oyunu sonuna kadar oynatır."""
    state = game.get_initial_state()
    cur_p = PLAYER

    # Açılış hamlelerini oyna
    for action in opening_moves:
        state = game.get_next_state(state, action, cur_p)
        cur_p = AI if cur_p == PLAYER else PLAYER

    mcts = GenericMCTS(model, game=game, num_simulations=num_simulations)
    full_moves = list(opening_moves)

    # Kalan oyunu oyna
    while True:
        term = game.check_terminal_value(state, cur_p)
        if term is not None:
            # cur_p için oyun bitti
            break

        if cur_p == model_player:
            policy = mcts.search(state, cur_p, temperature=0.0, dirichlet_epsilon=0.0)
            move = int(np.argmax(policy))
        else:
            board_copy = game.clone_state(state)
            move = get_ai_move(board_copy, depth=minimax_depth, ai_player=cur_p)

        state = game.get_next_state(state, move, cur_p)
        full_moves.append(move)
        cur_p = AI if cur_p == PLAYER else PLAYER

    # Sonucu modelin perspektifinden değerlendir
    # Son hamleyi yapan oyuncu:
    last_p = AI if cur_p == PLAYER else PLAYER
    last_term = game.check_terminal_value(state, last_p)

    if last_term is None or last_term == 0.0:
        outcome = "draw"
    elif last_term > 0:
        outcome = "win" if last_p == model_player else "loss"
    else:
        outcome = "win" if last_p != model_player else "loss"

    return outcome, full_moves


def evaluate_robust(model_path: str, minimax_depth: int = 4, num_openings: int = 50,
                    num_simulations: int = 300, model_type: str = "auto", seed: int = 42):
    """50 açılış x 2 renk = 100 maçlık sağlam değerlendirme çalıştırır."""
    game = Connect4Game()
    model = load_model(model_path, model_type)
    model_name = type(model).__name__

    openings = generate_random_openings(game, num_openings, min_ply=2, max_ply=4, seed=seed)
    total_games = len(openings) * 2

    print(f"=== ROBUST EVALUATION: {model_path} ({model_name}) vs Minimax-Depth={minimax_depth} ===")
    print(f"Açılış Havuzu: {len(openings)} benzersiz varyant (2-4 hamlelik rastgele açılış)")
    print(f"Toplam Maç Sayısı: {total_games} (Her açılış 1 kez Sarı/P1, 1 kez Kırmızı/P2)")
    print(f"MCTS Simülasyon Sayısı: {num_simulations}\n")

    wins, losses, draws = 0, 0, 0
    p1_wins, p2_wins = 0, 0
    opening_stats = defaultdict(lambda: {"win": 0, "loss": 0, "draw": 0})

    for idx, op in enumerate(openings, 1):
        op_str = "-".join(map(str, op))

        # Maç 1: Model = P1 (Sarı, İlk Başlayan)
        out1, _ = play_from_opening(game, model, op, model_player=PLAYER,
                                    minimax_depth=minimax_depth, num_simulations=num_simulations)
        if out1 == "win":
            wins += 1
            p1_wins += 1
        elif out1 == "loss":
            losses += 1
        else:
            draws += 1
        opening_stats[op_str][out1] += 1

        # Maç 2: Model = P2 (Kırmızı, İkinci)
        out2, _ = play_from_opening(game, model, op, model_player=AI,
                                    minimax_depth=minimax_depth, num_simulations=num_simulations)
        if out2 == "win":
            wins += 1
            p2_wins += 1
        elif out2 == "loss":
            losses += 1
        else:
            draws += 1
        opening_stats[op_str][out2] += 1

        played_so_far = idx * 2
        win_r = (wins / played_so_far) * 100.0
        print(f"  [{played_so_far:2d}/{total_games}] Açılış {op_str} -> W={wins} L={losses} D={draws} (Kazanma: %{win_r:.1f})", flush=True)

    # İstatistikler ve Güven Aralığı
    total = wins + losses + draws
    win_rate, ci_low, ci_high = wilson_score_interval(wins, total)
    # Skor formülü: (W + 0.5 * D) / N
    score_rate, score_low, score_high = wilson_score_interval(int(wins + 0.5 * draws), total)

    print("\n" + "=" * 60)
    print("NİHAİ DEĞERLENDİRME VE GÜVEN ARALIĞI RAPORU")
    print("=" * 60)
    print(f"Model: {model_path} ({model_name}) vs Minimax-{minimax_depth}")
    print(f"Toplam Maç: {total} | Galibiyet (W): {wins} | Mağlubiyet (L): {losses} | Beraberlik (D): {draws}")
    print(f"Renk Dağılımı: P1 (Sarı) Galibiyeti: {p1_wins}/{len(openings)} | P2 (Kırmızı) Galibiyeti: {p2_wins}/{len(openings)}")
    print(f"Kazanma Oranı (Win Rate): %{win_rate*100:.1f} (95% CI: %{ci_low*100:.1f} - %{ci_high*100:.1f})")
    print(f"Turnuva Skoru ((W+0.5D)/N): %{score_rate*100:.1f} (95% CI: %{score_low*100:.1f} - %{score_high*100:.1f})")

    # En çok kayıp yaşanan açılışlar
    problematic = [op for op, st in opening_stats.items() if st["loss"] > 0]
    print(f"\nKayıp Yaşanan Açılış Varyantı Sayısı: {len(problematic)} / {len(openings)}")
    if problematic:
        print("Modelin Kaybettiği Açılışlar (Format: hamleler -> W/L/D):")
        for p_op in sorted(problematic, key=lambda x: opening_stats[x]["loss"], reverse=True)[:10]:
            st = opening_stats[p_op]
            print(f"  Açılış [{p_op}]: W={st['win']} L={st['loss']} D={st['draw']}")
    else:
        print("  Kayıp yaşanan hiçbir açılış varyantı bulunamadı (Tüm açılışlarda ezici üstünlük).")
    print("=" * 60)

    return {
        "wins": wins, "losses": losses, "draws": draws,
        "win_rate": win_rate, "ci_low": ci_low, "ci_high": ci_high,
        "problematic_openings": problematic
    }


def main():
    parser = argparse.ArgumentParser(description="eval_robust: İstatistiksel ve Rastgele Açılışlı Model Test Aracı")
    parser.add_argument("--a", type=str, required=True, help="Model checkpoint yolu")
    parser.add_argument("--minimax-depth", type=int, default=4, help="Minimax derinliği (4 veya 6)")
    parser.add_argument("--openings", type=int, default=50, help="Benzersiz rastgele açılış sayısı (toplam maç = 2x)")
    parser.add_argument("--sims", type=int, default=300, help="MCTS simülasyon sayısı")
    parser.add_argument("--model-type", type=str, default="auto", choices=["auto", "v1", "v2"])
    parser.add_argument("--seed", type=int, default=42, help="Rastgelelik tohumu")
    args = parser.parse_args()

    evaluate_robust(
        model_path=args.a,
        minimax_depth=args.minimax_depth,
        num_openings=args.openings,
        num_simulations=args.sims,
        model_type=args.model_type,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
