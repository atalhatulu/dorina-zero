"""Log altyapısı. games.jsonl, iterations.csv, eval.csv yazar."""
import csv
import json
import os
from datetime import datetime

LOGS_DIR = "logs"


def _ensure_dir():
    os.makedirs(LOGS_DIR, exist_ok=True)


# --- Oyun logu (JSONL) ---

GAMES_PATH = os.path.join(LOGS_DIR, "games.jsonl")


def log_game(game_id, iteration, moves, winner, duration_sec, extra=None):
    """
    Tek oyunu games.jsonl'e yazar.
    winner: 'player1' | 'player2' | 'draw'
    """
    _ensure_dir()
    record = {
        "game_id": game_id,
        "iteration": iteration,
        "moves": moves,
        "winner": winner,
        "duration_sec": round(duration_sec, 3),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    if extra:
        record.update(extra)
    with open(GAMES_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


# --- İterasyon özeti (CSV) ---

ITER_PATH = os.path.join(LOGS_DIR, "iterations.csv")
ITER_FIELDS = [
    "iteration", "games", "p1_wins", "p2_wins", "draws",
    "avg_moves", "avg_duration_sec",
    "policy_loss", "value_loss",
    "elo_vs_prev", "timestamp",
]


def log_iteration(iteration, games, p1_wins, p2_wins, draws,
                  avg_moves, avg_duration_sec,
                  policy_loss=None, value_loss=None,
                  elo_vs_prev=None):
    _ensure_dir()
    file_exists = os.path.exists(ITER_PATH)
    with open(ITER_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ITER_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "iteration": iteration,
            "games": games,
            "p1_wins": p1_wins,
            "p2_wins": p2_wins,
            "draws": draws,
            "avg_moves": round(avg_moves, 2),
            "avg_duration_sec": round(avg_duration_sec, 3),
            "policy_loss": round(policy_loss, 4) if policy_loss is not None else "",
            "value_loss": round(value_loss, 4) if value_loss is not None else "",
            "elo_vs_prev": round(elo_vs_prev, 1) if elo_vs_prev is not None else "",
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        })


# --- Değerlendirme logu (CSV) ---

EVAL_PATH = os.path.join(LOGS_DIR, "eval.csv")
EVAL_FIELDS = [
    "iteration", "vs_iteration", "wins", "losses", "draws",
    "win_rate", "elo_diff", "timestamp",
]


def log_eval(iteration, vs_iteration, wins, losses, draws, elo_diff=None):
    _ensure_dir()
    total = wins + losses + draws
    win_rate = wins / total if total else 0.0
    file_exists = os.path.exists(EVAL_PATH)
    with open(EVAL_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EVAL_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "iteration": iteration,
            "vs_iteration": vs_iteration,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "win_rate": round(win_rate, 3),
            "elo_diff": round(elo_diff, 1) if elo_diff is not None else "",
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        })


# --- Yardımcı ---

def elo_diff(win_rate, k=400.0):
    """Galibiyet oranından Elo farkı tahmini (0.5 → 0, 0.75 → ~+190)."""
    import math
    if win_rate <= 0.0:
        return -float("inf")
    if win_rate >= 1.0:
        return float("inf")
    return k * math.log10(win_rate / (1 - win_rate))