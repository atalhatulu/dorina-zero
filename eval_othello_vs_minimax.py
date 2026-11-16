"""eval_othello_vs_minimax.py — 14. İterasyon Othello Şampiyonunu Minimax Ustasına Karşı Sınama Scripti."""
import os
import time
import argparse
import numpy as np
import torch

from games.othello import OthelloGame, BOARD_SIZE
from network_othello import OthelloNet
from core.mcts import GenericMCTS
from games.othello_minimax import get_othello_ai_move


def play_duel(model, game, model_player, mcts_sims=200, minimax_depth=4, verbose=False):
    """Tek bir Model vs Minimax maçı oynatır."""
    mcts = GenericMCTS(model, game=game, num_simulations=mcts_sims)
    state = game.get_initial_state()
    current_player = 1
    move_count = 0

    while True:
        term = game.check_terminal_value(state, current_player)
        if term is not None:
            break

        if current_player == model_player:
            # AlphaZero (MCTS, deterministik temp=0)
            policy = mcts.search(state, current_player, temperature=0.0, dirichlet_epsilon=0.0)
            action = int(np.argmax(policy))
        else:
            # Minimax Usta
            action = get_othello_ai_move(state, depth=minimax_depth, ai_player=current_player)

        state = game.get_next_state(state, action, current_player)
        move_count += 1
        current_player = 3 - current_player

    p1_count = np.count_nonzero(state == 1)
    p2_count = np.count_nonzero(state == 2)
    model_score = p1_count if model_player == 1 else p2_count
    minimax_score = p2_count if model_player == 1 else p1_count

    if model_score > minimax_score:
        result = "model"
    elif minimax_score > model_score:
        result = "minimax"
    else:
        result = "draw"

    if verbose:
        role = "Siyah (P1)" if model_player == 1 else "Beyaz (P2)"
        print(f"  Maç Sonu: Model {role} | Skor: Model {model_score} - {minimax_score} Minimax | Kazanan: {result.upper()} ({move_count} hamle)")

    return result, model_score, minimax_score


def main():
    parser = argparse.ArgumentParser(description="Othello AlphaZero vs Heuristic Minimax")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/othello/current_best.pt")
    parser.add_argument("--sims", type=int, default=200, help="AlphaZero MCTS simülasyon adedi")
    parser.add_argument("--depth", type=int, default=4, help="Minimax arama derinliği")
    parser.add_argument("--games", type=int, default=10, help="Toplam maç adedi (yarı yarıya renk dağılımı)")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"Hata: Checkpoint bulunamadı: {args.checkpoint}")
        return

    game = OthelloGame()
    model = OthelloNet()
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    iter_num = ckpt.get("iteration", "Bilinmeyen")

    print(f"=== Othello Şampiyonu (İter {iter_num}) vs Heuristic Minimax (Depth {args.depth}) ===")
    print(f"Toplam {args.games} maç yapılacak (MCTS {args.sims} simülasyon)...\n")

    model_wins = 0
    minimax_wins = 0
    draws = 0
    total_model_disks = 0
    total_minimax_disks = 0

    t0 = time.time()
    for g_idx in range(args.games):
        model_player = 1 if (g_idx % 2 == 0) else 2
        res, m_pts, mm_pts = play_duel(model, game, model_player, mcts_sims=args.sims, minimax_depth=args.depth, verbose=True)
        if res == "model":
            model_wins += 1
        elif res == "minimax":
            minimax_wins += 1
        else:
            draws += 1
        total_model_disks += m_pts
        total_minimax_disks += mm_pts

    duration = time.time() - t0
    win_rate = (model_wins + 0.5 * draws) / args.games * 100.0

    print("\n" + "=" * 55)
    print("🏆 NİHAİ DÜELLO SKOR TABLOSU")
    print("=" * 55)
    print(f"AlphaZero Galibiyeti: {model_wins} / {args.games} (%{model_wins/args.games*100:.1f})")
    print(f"Minimax Galibiyeti  : {minimax_wins} / {args.games} (%{minimax_wins/args.games*100:.1f})")
    print(f"Beraberlik          : {draws} / {args.games}")
    print(f"Toplam Puan Oranı   : %{win_rate:.1f}")
    print(f"Ortalama Taş Sayısı : Model {total_model_disks/args.games:.1f} vs {total_minimax_disks/args.games:.1f} Minimax")
    print(f"Toplam Süre         : {duration:.1f} saniye")
    print("=" * 55)


if __name__ == "__main__":
    main()
