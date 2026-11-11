import os
import re
import pandas as pd

log_path = "training_20iter_stable.log"
out_dir = "logs_v3"
out_csv = os.path.join(out_dir, "iterations_v3.csv")
os.makedirs(out_dir, exist_ok=True)

if not os.path.exists(log_path):
    exit(0)

with open(log_path, "r", encoding="utf-8") as f:
    text = f.read()

# Pattern matching
# [Iter 001]
# Self-play bitti: 128.1s | p1=174 p2=119 draw=7 (Draw: %2.3)
# Eğitim bitti: policy_loss=1.7694, value_loss=0.3130
# Kaydedildi: checkpoints/connect4/iter_001.pt | Toplam tur: 142.3s

pattern = re.compile(
    r"\[Iter\s+(\d+)\][\s\S]*?"
    r"Self-play bitti:\s*([\d\.]+)s\s*\|\s*p1=(\d+)\s+p2=(\d+)\s+draw=(\d+)\s+\(Draw:\s*%([\d\.]+)\)[\s\S]*?"
    r"Eğitim bitti:\s*policy_loss=([\d\.]+),\s*value_loss=([\d\.]+)[\s\S]*?"
    r"Toplam tur:\s*([\d\.]+)s"
)

rows = []
for m in pattern.finditer(text):
    it = int(m.group(1))
    sp_time = float(m.group(2))
    p1 = int(m.group(3))
    p2 = int(m.group(4))
    draw = int(m.group(5))
    draw_pct = float(m.group(6))
    pol_loss = float(m.group(7))
    val_loss = float(m.group(8))
    total_time = float(m.group(9))

    rows.append({
        "iteration": it,
        "policy_loss": pol_loss,
        "value_loss": val_loss,
        "total_loss": pol_loss + val_loss,
        "self_play_time": sp_time,
        "train_time": total_time - sp_time,
        "player1_wins": p1,
        "player2_wins": p2,
        "draws": draw,
        "draw_pct": draw_pct,
    })

if rows:
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
