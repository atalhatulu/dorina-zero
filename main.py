"""dorina4 - Connect 4. Giriş noktası (Minimax veya AlphaZero)."""
import os
import argparse
from functools import partial
import torch

from board import AI
from minimax_ai import get_ai_move
from gui import run_game
from network import Connect4Net
from network_v2 import Connect4NetV2
from train import load_checkpoint
from mcts import MCTS


def load_model_auto(model_path: str):
    """Checkpoint yapısını inceleyerek v1 (465K) veya v2 (116K) modelini otomatik oluşturur ve yükler."""
    ckpt = torch.load(model_path, map_location="cpu")
    state = ckpt.get("model_state", ckpt)
    iteration = ckpt.get("iteration", 0)

    # v2 mimarisinde 1x1 conv başlıkları veya batchnorm vardır
    if "policy_conv.weight" in state or "bn1.weight" in state:
        model = Connect4NetV2()
    else:
        model = Connect4Net()

    model.load_state_dict(state)
    model.eval()
    return model, iteration


def main():
    parser = argparse.ArgumentParser(description="dorina4 - Connect 4 Oyunu")
    parser.add_argument("--model", type=str, default=None, help="AlphaZero checkpoint yolu (.pt)")
    parser.add_argument("--sims", type=int, default=100, help="MCTS simülasyon sayısı")
    parser.add_argument("--minimax-depth", type=int, default=4, help="Minimax derinliği (model verilmediğinde)")
    args = parser.parse_args()

    analysis_fn = None
    eval_fn = None

    if args.model and os.path.exists(args.model):
        model, iteration = load_model_auto(args.model)
        mcts = MCTS(model, num_simulations=args.sims)

        # Analiz ve değerlendirme fonksiyonları
        analysis_fn = mcts.analyze
        eval_fn = mcts.evaluate_board
        ai_fn = lambda board: mcts.best_move(board, current_player=AI)
        ai_name = f"AlphaZero (iter_{iteration:03d}, {args.sims} sim)"
        print(f"AlphaZero modeli yüklendi: {args.model} ({type(model).__name__}, İterasyon: {iteration})")
    else:
        if args.model:
            print(f"Uyarı: Model dosyası bulunamadı ({args.model}), varsayılan Minimax kullanılıyor.")
        ai_fn = partial(get_ai_move, depth=args.minimax_depth)
        ai_name = f"Minimax (depth {args.minimax_depth})"
        print(f"Minimax AI aktif (derinlik: {args.minimax_depth})")

    run_game(
        ai_move_fn=ai_fn,
        ai_name=ai_name,
        analysis_fn=analysis_fn,
        eval_fn=eval_fn,
    )


if __name__ == "__main__":
    main()
