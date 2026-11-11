"""CLI: checkpoint karşılaştırma / minimax'a karşı test."""
import argparse
from network import Connect4Net
from evaluate import compare_checkpoints, evaluate_vs_minimax
from train import load_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", type=str, required=True, help="Model A checkpoint")
    parser.add_argument("--b", type=str, help="Model B checkpoint (yoksa minimax)")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--sims", type=int, default=50)
    parser.add_argument("--minimax-depth", type=int, default=4)
    args = parser.parse_args()

    if args.b:
        print(f"Karşılaştırma: {args.a} vs {args.b}")
        compare_checkpoints(args.a, args.b, num_games=args.games,
                            num_simulations=args.sims)
    else:
        print(f"Model {args.a} vs minimax depth={args.minimax_depth}")
        model = Connect4Net()
        load_checkpoint(model, args.a)
        w, l, d = evaluate_vs_minimax(
            model, num_games=args.games,
            num_simulations=args.sims,
            minimax_depth=args.minimax_depth, verbose=True,
        )
        total = w + l + d
        print(f"Sonuç: {w}W/{l}L/{d}D ({w/total:.1%} kazanma)")


if __name__ == "__main__":
    main()