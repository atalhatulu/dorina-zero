"""
AlphaZero Othello vs Edax Tournament Engine Benchmark Script
Dünya standartlarındaki Edax reversi motoru ile modelimizi GTP protokolü üzerinden karşılaştırır.
"""

import os
import sys
import argparse
import subprocess
import torch
import numpy as np

from games.othello import OthelloGame
from network_othello import OthelloNet
from core.mcts import GenericMCTS


def action_to_gtp(action: int) -> str:
    """Action indexini (0..64) GTP koordinatına ('f5', 'd6', 'pass') çevirir."""
    if action == 64:
        return "pass"
    r, c = divmod(action, 8)
    col_char = chr(ord('a') + c)
    row_char = str(r + 1)
    return f"{col_char}{row_char}"


def gtp_to_action(gtp_str: str) -> int:
    """GTP koordinatını ('f5', 'pass') action indexine (0..64) çevirir."""
    clean = gtp_str.strip().lower()
    if clean in ("pass", "resign", ""):
        return 64
    c = ord(clean[0]) - ord('a')
    r = int(clean[1]) - 1
    return r * 8 + c


class EdaxBot:
    def __init__(self, level: int = 4, edax_dir: str = "bin/edax"):
        self.edax_dir = os.path.abspath(edax_dir)
        self.level = level
        self.proc = None
        self._start_process()

    def _start_process(self):
        cmd = ["./edax", "-gtp", "-l", str(self.level), "-q"]
        self.proc = subprocess.Popen(
            cmd,
            cwd=self.edax_dir,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        self._send_command("boardsize 8")
        self._send_command("clear_board")

    def _send_command(self, cmd: str) -> str:
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()
        lines = []
        while True:
            line = self.proc.stdout.readline()
            if not line:
                break
            line = line.strip()
            if line == "":
                if lines:
                    break
                continue
            lines.append(line)
        res = " ".join(lines)
        if res.startswith("= "):
            return res[2:].strip()
        return res

    def reset(self):
        self._send_command("clear_board")

    def play_move(self, color_str: str, move_str: str):
        self._send_command(f"play {color_str} {move_str}")

    def genmove(self, color_str: str) -> str:
        return self._send_command(f"genmove {color_str}")

    def close(self):
        if self.proc:
            try:
                self._send_command("quit")
            except Exception:
                pass
            self.proc.terminate()


def play_match(game, mcts, edax, model_player: int, verbose: bool = True):
    """
    Tek maç oynatır.
    model_player: 1 (Siyah / black) veya 2 (Beyaz / white)
    """
    state = game.get_initial_state()
    current_player = 1
    move_count = 0
    edax.reset()

    while True:
        term = game.check_terminal_value(state, current_player)
        if term is not None:
            break

        color_str = "black" if current_player == 1 else "white"

        if current_player == model_player:
            # AlphaZero Hamlesi
            probs = mcts.search(state, current_player, temperature=0.0, dirichlet_epsilon=0.0)
            action = int(np.argmax(probs))
            gtp_move = action_to_gtp(action)
            # Edax tahtasını senkronize et
            edax.play_move(color_str, gtp_move)
        else:
            # Edax Hamlesi
            gtp_move = edax.genmove(color_str)
            action = gtp_to_action(gtp_move)

        state = game.get_next_state(state, action, current_player)
        current_player = 3 - current_player
        move_count += 1

    p1_count = np.count_nonzero(state == 1)
    p2_count = np.count_nonzero(state == 2)
    model_score = p1_count if model_player == 1 else p2_count
    edax_score = p2_count if model_player == 1 else p1_count

    if model_score > edax_score:
        result = "model"
    elif edax_score > model_score:
        result = "edax"
    else:
        result = "draw"

    if verbose:
        role = "Siyah (P1)" if model_player == 1 else "Beyaz (P2)"
        print(f"  Maç Sonu: Model {role} | Skor: Model {model_score} - {edax_score} Edax | Kazanan: {result.upper()} ({move_count} hamle)")

    return result, model_score, edax_score


def main():
    default_ckpt = "models/othello/checkpoints/current_best.pt"
    parser = argparse.ArgumentParser(description="AlphaZero vs Edax Tournament Engine Düellosu")
    parser.add_argument("--checkpoint", type=str, default=default_ckpt)
    parser.add_argument("--sims", type=int, default=200, help="MCTS simülasyon adedi")
    parser.add_argument("--level", type=int, default=4, help="Edax arama seviyesi (level 1-10)")
    parser.add_argument("--games", type=int, default=4, help="Toplam maç adedi (renk değişmeli)")
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

    mcts = GenericMCTS(model=model, game=game, num_simulations=args.sims, c_puct=1.5)
    edax = EdaxBot(level=args.level)

    print(f"\n=== Othello Şampiyonu (İter {iter_num}) vs Edax Tournament Engine (Level {args.level}) ===")
    print(f"Toplam {args.games} maç yapılacak (MCTS {args.sims} simülasyon)...")

    results = {"model": 0, "edax": 0, "draw": 0}
    total_model_discs = 0
    total_edax_discs = 0

    import time
    t0 = time.time()

    try:
        for g in range(args.games):
            model_player = 1 if (g % 2 == 0) else 2
            res, m_score, e_score = play_match(game, mcts, edax, model_player, verbose=True)
            results[res] += 1
            total_model_discs += m_score
            total_edax_discs += e_score
    finally:
        edax.close()

    dt = time.time() - t0
    win_pct = results["model"] / args.games * 100.0
    score_rate = (results["model"] + 0.5 * results["draw"]) / args.games * 100.0

    print("\n" + "=" * 55)
    print("🏆 NİHAİ DÜELLO SKOR TABLOSU (ALPHAZERO vs EDAX)")
    print("=" * 55)
    print(f"AlphaZero Galibiyeti: {results['model']} / {args.games} (%{win_pct:.1f})")
    print(f"Edax Galibiyeti     : {results['edax']} / {args.games} (%{results['edax']/args.games*100:.1f})")
    print(f"Beraberlik          : {results['draw']} / {args.games}")
    print(f"Toplam Puan Oranı   : %{score_rate:.1f}")
    print(f"Ortalama Taş Sayısı : Model {total_model_discs/args.games:.1f} vs {total_edax_discs/args.games:.1f} Edax")
    print(f"Toplam Süre         : {dt:.1f} saniye")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
