"""CLI: checkpoint karşılaştırma / minimax'a karşı test.

Otomatik model mimarisi tespiti (v1: 465K Connect4Net, v2: 116K Connect4NetV2)
veya elle `--model-type` parametresi ile seçim desteği.
"""
import os
import argparse
import torch

from network import Connect4Net
from network_v2 import Connect4NetV2
from evaluate import evaluate_vs_minimax, play_match
from train import load_checkpoint


def get_model_for_checkpoint(ckpt_path: str, model_type: str = "auto") -> torch.nn.Module:
    """Checkpoint dosyasını inceleyerek doğru mimariyi (v1 veya v2) döndürür."""
    if model_type == "v1":
        return Connect4Net()
    elif model_type == "v2":
        return Connect4NetV2()

    # Otomatik algılama ("auto")
    if not os.path.exists(ckpt_path):
        return Connect4NetV2()

    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = ckpt.get("model_state", ckpt)
        # v2'de 1x1 conv başlıkları ("policy_conv.weight") bulunur
        if "policy_conv.weight" in state or "bn1.weight" in state:
            return Connect4NetV2()
        return Connect4Net()
    except Exception:
        return Connect4NetV2()


def main():
    parser = argparse.ArgumentParser(description="AlphaZero Model Değerlendirme Aracı")
    parser.add_argument("--a", type=str, required=True, help="Model A checkpoint yolu")
    parser.add_argument("--b", type=str, default=None, help="Model B checkpoint yolu (verilmezse Minimax)")
    parser.add_argument(
        "--model-type",
        type=str,
        default="auto",
        choices=["auto", "v1", "v2"],
        help="Model mimarisi türü (auto, v1, v2). Varsayılan: auto",
    )
    parser.add_argument("--games", type=int, default=30, help="Oynanacak toplam oyun sayısı")
    parser.add_argument("--sims", type=int, default=150, help="MCTS simülasyon adedi")
    parser.add_argument("--minimax-depth", type=int, default=4, help="Minimax arama derinliği")
    args = parser.parse_args()

    model_a = get_model_for_checkpoint(args.a, args.model_type)
    load_checkpoint(model_a, args.a)
    model_a_name = type(model_a).__name__

    if args.b:
        model_b = get_model_for_checkpoint(args.b, args.model_type)
        load_checkpoint(model_b, args.b)
        model_b_name = type(model_b).__name__
        print(f"Karşılaştırma: {args.a} ({model_a_name}) vs {args.b} ({model_b_name})")
        a_wins, b_wins, draws = play_match(
            model_a, model_b, num_games=args.games, num_simulations=args.sims, verbose=True
        )
        total = a_wins + b_wins + draws
        print(f"\nNihai Sonuç: A={a_wins} B={b_wins} D={draws} (A Kazanma: {a_wins/total:.1%})")
    else:
        print(f"Model {args.a} ({model_a_name}) vs Minimax-Depth={args.minimax_depth}")
        print(f"Oyun sayısı: {args.games}, Simülasyon: {args.sims}")
        w, l, d = evaluate_vs_minimax(
            model_a,
            num_games=args.games,
            num_simulations=args.sims,
            minimax_depth=args.minimax_depth,
            verbose=True,
        )
        total = w + l + d
        win_rate = (w / total) * 100.0 if total else 0.0
        print(f"\nNihai Sonuç: {w}W / {l}L / {d}D (Kazanma Oranı: %{win_rate:.1f})")


if __name__ == "__main__":
    main()
