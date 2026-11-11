"""Tek self-play oyunu oynat, eğitim verisi üret, logla."""
import time
import numpy as np
import torch

from board import (
    ROWS, COLS, EMPTY, PLAYER, AI,
    create_board, drop_piece, get_next_open_row,
    get_valid_locations, winning_move, is_full, clone_board,
)
from network import board_to_tensor
from mcts import MCTS, sample_move
import logger


def play_one_game(model, num_simulations=200, temperature=1.0,
                  temp_threshold=6, game_id=None, iteration=0,
                  log=True):
    """
    Bir self-play oyunu oynat.

    temperature: ilk `temp_threshold` hamlede 1.0 (keşif), sonra 0 (en iyi)
    Dönüş: (history, winner_str, num_moves)
      history: [(board, current_player, mcts_policy), ...]
    """
    mcts = MCTS(model, num_simulations=num_simulations)
    board = create_board()
    current_player = PLAYER
    history = []
    start = time.time()
    move_count = 0

    while True:
        valid = get_valid_locations(board)
        if not valid:
            winner = "draw"
            break

        # Sıcaklık: ilk N hamlede keşif, sonra deterministik
        temp = temperature if move_count < temp_threshold else 0.0

        # MCTS ile policy al
        policy = mcts.search(board, current_player, temperature=temp)

        # Eğitim için sakla: tahtayı o anki oyuncunun perspektifinden
        history.append((clone_board(board), current_player, policy.copy()))

        # Hamleyi örnekle (sıcaklık zaten policy'ye uygulandı)
        move = sample_move(policy)
        row = get_next_open_row(board, move)
        drop_piece(board, row, move, current_player)
        move_count += 1

        # Kazanan var mı?
        if winning_move(board, current_player):
            winner = "player1" if current_player == PLAYER else "player2"
            break

        if is_full(board):
            winner = "draw"
            break

        current_player = AI if current_player == PLAYER else PLAYER

    duration = time.time() - start

    if log and game_id is not None:
        logger.log_game(
            game_id=game_id,
            iteration=iteration,
            moves=move_count,
            winner=winner,
            duration_sec=duration,
            extra={"simulations": num_simulations},
        )

    return history, winner, move_count


def history_to_training_data(history, winner):
    """history'yi (state, policy_target, value_target) listesine çevir.
    Symmetry augmentation: her örnek 2x (orijinal + yatay flip).
    """
    if winner == "draw":
        winner_player = None
    elif winner == "player1":
        winner_player = PLAYER
    else:
        winner_player = AI

    data = []
    for board, player, policy in history:
        if winner == "draw":
            v = 0.0
        else:
            v = 1.0 if player == winner_player else -1.0

        # Orijinal
        data.append((board, player, policy, v))

        # Yatay flip (Connect4 dikey simetrik)
        flipped_board = [row[::-1] for row in board]
        flipped_policy = policy[::-1].copy()
        data.append((flipped_board, player, flipped_policy, v))

    return data


def play_games(model, num_games, num_simulations=200,
               iteration=0, start_game_id=0, log=True, verbose=False):
    """
    Birden çok oyun oynat (seri — paralel versiyon run_self_play.py'de).
    Dönüş: (all_data, stats)
    """
    all_data = []
    stats = {"player1": 0, "player2": 0, "draw": 0,
             "total_moves": 0, "total_duration": 0.0}

    for i in range(num_games):
        gid = start_game_id + i
        history, winner, moves = play_one_game(
            model, num_simulations=num_simulations,
            game_id=gid, iteration=iteration, log=log,
        )
        stats[winner] += 1
        stats["total_moves"] += moves
        if log:
            # duration log_game içinde yazıldı, ama stats için tekrar hesaplama
            pass
        all_data.extend(history_to_training_data(history, winner))

        if verbose and (i + 1) % 10 == 0:
            print(f"  [{i+1}/{num_games}] p1={stats['player1']} "
                  f"p2={stats['player2']} draw={stats['draw']}")

    if num_games > 0:
        stats["avg_moves"] = stats["total_moves"] / num_games
    else:
        stats["avg_moves"] = 0.0
    return all_data, stats
