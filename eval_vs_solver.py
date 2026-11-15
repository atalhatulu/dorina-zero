"""Nihai Karşılaşma: AlphaZero (v5/current_best) vs Kusursuz Çözücü (GameSolver)."""
import time
import numpy as np
import torch
from games.connect4 import Connect4Game
from network_v2 import Connect4NetV2
from core.self_play import GenericMCTS
from train import load_checkpoint
from perfect_solver import get_solver_best_move

def play_game_vs_solver(model_player=1, num_simulations=400):
    game = Connect4Game()
    model = Connect4NetV2()
    load_checkpoint(model, "checkpoints_v5/current_best.pt")
    model.eval()

    mcts = GenericMCTS(model, game=game, num_simulations=num_simulations)

    state = game.get_initial_state()
    current_player = 1
    move_history = []
    
    role_str = "Oyuncu 1 (Kırmızı / İlk Başlayan)" if model_player == 1 else "Oyuncu 2 (Sarı / Savunan)"
    print(f"\n======================================================================")
    print(f"⚔️ MAÇ BAŞLADI: Model Rolü: {role_str} | Simülasyon: {num_simulations}")
    print(f"Rakip: Pascal Pons Kusursuz Çözücü (GameSolver Perfect Oracle)")
    print(f"======================================================================\n")

    accuracy_hits = 0
    model_turns = 0

    for ply in range(1, 43):
        # 1. Hamle sırası
        is_model = (current_player == model_player)
        
        # Çözücünün bu pozisyondaki mutlak skoru ve en iyi hamlesi
        solver_best, scores, stat = get_solver_best_move(move_history)
        
        if is_model:
            model_turns += 1
            t0 = time.time()
            policy = mcts.search(state, current_player, temperature=0.0, dirichlet_epsilon=0.0)
            action = int(np.argmax(policy))
            dt = time.time() - t0
            
            # Çözücüye göre en iyi hamleyi mi yaptı?
            is_perfect = (action == solver_best)
            if is_perfect:
                accuracy_hits += 1
            star = "⭐️ KUSURSUZ (En İyi Hamle)" if is_perfect else f"⚠️ Alt-Optimal (Çözücü Tercihi: Kolon {solver_best+1})"
            
            print(f"Hamle {ply:02d} [Model - P{current_player}]: Kolon {action+1} ({dt:.2f}s) | Çözücü Durumu: {stat} | {star}")
        else:
            action = solver_best
            print(f"Hamle {ply:02d} [Çözücü - P{current_player}]: Kolon {action+1} | Çözücü Durumu: {stat}")

        move_history.append(action)
        state = game.get_next_state(state, action, current_player)

        # Bitiş kontrolü
        term = game.check_terminal_value(state, current_player)
        if term is not None:
            print(f"\n--- OYUN BİTTİ (Hamle {ply}) ---")
            if term == 0.0:
                print("🏁 SONUÇ: BERABERLİK! Model kusursuz çözücüye karşı beraberliği kopardı!")
            elif (term > 0 and is_model) or (term < 0 and not is_model):
                print("🏆 SONUÇ: MODEL KAZANDI! (Tarihi Başarı)")
            else:
                print(f"❌ SONUÇ: KUSURSUZ ÇÖZÜCÜ KAZANDI. (Model {ply} hamle dayandı)")
            break

        current_player = 2 if current_player == 1 else 1

    acc_pct = (accuracy_hits / max(1, model_turns)) * 100.0
    print(f"\n📊 Model Doğruluk Oranı: %{acc_pct:.1f} ({accuracy_hits} / {model_turns} hamlede çözücüyle birebir aynı oynadı)")
    print(f"Hamle Dizilimi: {''.join(str(m+1) for m in move_history)}\n")

if __name__ == "__main__":
    # 1. Maç: Model P1 (Kırmızı / İlk oynayan)
    play_game_vs_solver(model_player=1, num_simulations=400)
    # 2. Maç: Model P2 (Sarı / Savunan)
    play_game_vs_solver(model_player=2, num_simulations=400)
